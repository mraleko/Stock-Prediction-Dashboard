# Stock Prediction Dashboard

This project is a web application for visualizing historical stock prices and predicting the next 5 days of closing prices using a machine learning model (LSTM neural network). It supports both stocks (including S&P 500 constituents) and major cryptocurrencies.

## Features

- **Historical Price Charts:** Interactive charts for stocks and cryptocurrencies.
- **5-Day Price Prediction:** LSTM-based neural network forecasts for the next 5 business days.
- **Technical Indicators:** Uses moving averages (MA10, MA20) and RSI for improved predictions.
- **S&P 500 & Crypto Lists:** Quick access to top S&P 500 stocks and major cryptocurrencies.
- **Dark/Light Mode:** Toggleable theme for better viewing experience.

## Technology Stack

- **Backend:** Python, Flask, PyTorch, yfinance, scikit-learn, pandas, numpy
- **Frontend:** HTML, CSS, JavaScript, Chart.js

## Setup Instructions

### Prerequisites

- Python 3.8+
- pip

### Installation

1. **Clone the repository:**
    ```bash
    git clone <repo-url>
    cd Project\ 2
    ```

2. **Install dependencies:**
    ```bash
    pip install flask torch yfinance pandas numpy scikit-learn
    ```

3. **Run the backend server:**
    ```bash
    python backend.py
    ```
    The server will start at [http://localhost:5001](http://localhost:5001).

4. **Open the app:**
    - Open `http://localhost:5001` in your browser.

## Usage

- **Search for a stock:** Enter a ticker (e.g., `AAPL`) and click "Go".
- **View S&P 500 or Crypto:** Use the dashboard lists for quick access.
- **Change time period:** Use the time filter buttons above each chart.
- **View predictions:** See the next 5-day price forecast and percentage change.

## Model Details

- **Architecture:** 2-layer LSTM (64 hidden units, dropout 0.2), Dense output for 5-day prediction.
- **Inputs:** Last 60 days of OHLCV + MA10, MA20, RSI.
- **Training:** Trained on up to 10 years of historical data, retrained weekly per ticker.
- **Loss:** SmoothL1Loss.

## Disclaimer

This dashboard is for informational and educational purposes only. Predictions are not financial advice.

## File Structure

```
backend.py           # Flask backend and ML logic
static/
  style.css          # Main CSS
  index.html         # Dashboard homepage
stock.html           # Stock details page
stock.js             # Stock page JS
app.js               # Dashboard JS
stock_models/        # Saved models and scalers
```