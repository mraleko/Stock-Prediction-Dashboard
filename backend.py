import torch
import torch.nn as nn
from flask import Flask, jsonify, request, send_from_directory
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import json
from datetime import datetime, date

app = Flask(__name__, static_url_path='', static_folder='.')

def add_technical_indicators(df):
    df = df.copy()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    df = df.dropna()
    return df

class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=5, dropout=0.2):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def preprocess_data(data, scaler):
    # Add technical indicators
    data = add_technical_indicators(data)
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA20', 'RSI']
    scaled_data = scaler.transform(data[features].values)
    return scaled_data, data

def train_model(data, sequence_length=60, epochs=50):
    # Train/validation split (last 10% for validation)
    n = len(data)
    train_size = int(n * 0.9)
    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:]

    # Fit scaler only on training data
    train_data_with_ind = add_technical_indicators(train_data)
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA20', 'RSI']
    scaler = MinMaxScaler()
    scaler.fit(train_data_with_ind[features].values)

    scaled_train, _ = preprocess_data(train_data, scaler)
    scaled_val, _ = preprocess_data(val_data, scaler)

    # Prepare sequences for training (multi-output: next 5 closes)
    def create_sequences(scaled, raw_data):
        sequences, targets = [], []
        for i in range(len(scaled) - sequence_length - 4):
            sequences.append(scaled[i:i+sequence_length])
            # Predict next 5 closes (index 3 is 'Close')
            targets.append([scaled[i+sequence_length+j][3] for j in range(5)])
        return torch.FloatTensor(sequences), torch.FloatTensor(targets)

    X_train, y_train = create_sequences(scaled_train, train_data)
    X_val, y_val = create_sequences(scaled_val, val_data)

    model = StockLSTM(input_size=8, output_size=5)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        # Optionally add validation loss monitoring here

    return model, scaler

# Initialize model storage
MODELS = {}
SCALERS = {}
MODELS_DIR = 'stock_models'

# Create models directory if it doesn't exist
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def get_model_path(ticker):
    return os.path.join(MODELS_DIR, f'{ticker}_model.pth')

def get_meta_path(ticker):
    return os.path.join(MODELS_DIR, f'{ticker}_meta.json')

def get_current_week_str():
    # Returns a string like '2024-W27'
    today = date.today()
    year, week, _ = today.isocalendar()
    return f"{year}-W{week:02d}"

def load_last_retrain_week(ticker):
    meta_path = get_meta_path(ticker)
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            # Backward compatibility: support both 'last_retrain_date' and 'last_retrain_week'
            if 'last_retrain_week' in meta:
                return meta.get('last_retrain_week')
            elif 'last_retrain_date' in meta:
                # Convert old date to week string
                dt = datetime.strptime(meta['last_retrain_date'], '%Y-%m-%d').date()
                year, week, _ = dt.isocalendar()
                return f"{year}-W{week:02d}"
        except Exception:
            return None
    return None

def save_last_retrain_week(ticker, week_str):
    meta_path = get_meta_path(ticker)
    with open(meta_path, 'w') as f:
        json.dump({'last_retrain_week': week_str}, f)

def yfinance_symbol(ticker):
    # yfinance expects BRK.B as BRK-B, etc.
    return ticker.replace('.', '-')

def initialize_model(ticker):
    model_path = get_model_path(ticker)
    scaler_path = os.path.join(MODELS_DIR, f'{ticker}_scaler.npy')
    current_week_str = get_current_week_str()
    last_retrain_week = load_last_retrain_week(ticker)
    retrain_needed = (last_retrain_week != current_week_str)
    if os.path.exists(model_path) and os.path.exists(scaler_path) and not retrain_needed:
        model = StockLSTM(input_size=8, output_size=5)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        scaler = np.load(scaler_path, allow_pickle=True).item()
        return model, scaler, last_retrain_week
    yf_ticker = yfinance_symbol(ticker)
    stock = yf.Ticker(yf_ticker)
    # Try to get up to 10 years of data, fallback to max if not enough
    hist = stock.history(period="10y")
    if hist.empty or len(hist) < 250:  # Less than ~1 year of trading days
        hist = stock.history(period="max")
    if hist.empty:
        raise ValueError(f"No historical data available for {ticker}")
    model, scaler = train_model(hist)
    torch.save(model.state_dict(), model_path)
    np.save(scaler_path, scaler)
    save_last_retrain_week(ticker, current_week_str)
    model.eval()
    return model, scaler, current_week_str

@app.route('/')
def serve_index():
    # Try static/index.html first for backward compatibility, else index.html
    if os.path.exists(os.path.join(app.static_folder, 'static/index.html')):
        return send_from_directory(os.path.join(app.static_folder, 'static'), 'index.html')
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/stock-data/<ticker>')
def get_stock_data(ticker):
    # Map frontend period values to yfinance period format
    period_map = {
        '1w': '7d',
        '1m': '1mo',
        '6m': '6mo',
        '1y': '1y',
        '5y': '5y',
        'max': 'max'
    }
    
    period = request.args.get('period', '6mo')
    yf_period = period_map.get(period, '6mo')  # Default to 6mo if period not found

    start = request.args.get('start')
    end = request.args.get('end')
    
    try:
        yf_ticker = yfinance_symbol(ticker)
        stock = yf.Ticker(yf_ticker)
        if start or end:
            hist = stock.history(start=start, end=end)
        else:
            hist = stock.history(period=yf_period)
        if hist.empty:
            return jsonify({'error': 'No data available for this period'}), 400
            
        hist.index = hist.index.strftime('%Y-%m-%d')
        return jsonify(hist.reset_index().to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/<ticker>')
def predict_stock(ticker):
    try:
        # Get or initialize model and scaler for this specific stock
        if ticker not in MODELS or ticker not in SCALERS:
            model, scaler, last_retrain_week = initialize_model(ticker)
            MODELS[ticker] = model
            SCALERS[ticker] = scaler
        else:
            last_retrain_week = load_last_retrain_week(ticker)

        yf_ticker = yfinance_symbol(ticker)
        stock = yf.Ticker(yf_ticker)
        hist = stock.history(period='1y')
        if hist.empty or len(hist) < 65:
            return jsonify({'error': 'Not enough data to predict'}), 400

        # Add technical indicators and scale using training scaler
        scaled_data, hist_with_ind = preprocess_data(hist, scaler=SCALERS[ticker])
        sequence_length = 60
        last_sequence = scaled_data[-(sequence_length):]
        x = torch.FloatTensor(last_sequence).unsqueeze(0)
        with torch.no_grad():
            pred_scaled = MODELS[ticker](x).numpy()[0]  # shape: (5,)

        # Prepare dummy input for inverse transform
        predictions = []
        for i in range(5):
            dummy_input = np.zeros((1, 8))
            dummy_input[0, 3] = pred_scaled[i]  # 'Close' is at index 3
            pred_close = SCALERS[ticker].inverse_transform(dummy_input)[0, 3]
            predictions.append(float(pred_close))

        last_date = hist_with_ind.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=5, freq='B')

        prediction_data = [{
            'Date': date.strftime('%Y-%m-%d'),
            'Close': round(price, 2)
        } for date, price in zip(future_dates, predictions)]

        return jsonify({
            'predictions': prediction_data,
            'timestamp': pd.Timestamp.now().isoformat(),
            'last_retrain_week': last_retrain_week
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5001)