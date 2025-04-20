let sp500Chart = null;

async function searchSP500() {
    const ticker = "^GSPC"; // Use ^GSPC for S&P 500 index (Yahoo Finance)
    const period = document.querySelector('.sp500-time-button.active')?.dataset.period || '1y';

    let url = `/stock-data/${encodeURIComponent(ticker)}?period=${period}`;

    try {
        const [stockResponse, predictionResponse] = await Promise.all([
            fetch(url),
            fetch(`/predict/${encodeURIComponent(ticker)}`)
        ]);
        const stockData = await stockResponse.json();
        const predictionData = await predictionResponse.json();
        updateSP500Chart(stockData, predictionData);
        updateSP500PredictionInfo(predictionData);
    } catch (error) {
        console.error('Error:', error);
        alert('Error fetching S&P 500 data. Please try again.');
    }
}

function updateSP500Chart(data, predictionData) {
    const ctx = document.getElementById('sp500Chart').getContext('2d');
    if (sp500Chart) {
        sp500Chart.destroy();
    }
    const historicalData = data.map(d => ({
        x: d.Date,
        y: d.Close
    }));
    let predictionPoints = predictionData.predictions.map(d => ({
        x: d.Date,
        y: d.Close
    }));
    if (historicalData.length > 0 && predictionPoints.length > 0) {
        predictionPoints = [
            { x: historicalData[historicalData.length - 1].x, y: historicalData[historicalData.length - 1].y },
            ...predictionPoints
        ];
    }
    sp500Chart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [{
                label: 'Historical Price',
                data: historicalData,
                borderColor: '#28a745', // green
                tension: 0.1,
                fill: false,
                pointRadius: 0
            },
            {
                label: 'Predicted Price',
                data: predictionPoints,
                borderColor: '#007bff', // blue
                borderDash: [5, 5],
                tension: 0.1,
                fill: false,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: false,
                    text: 'S&P 500 Price History and Prediction'
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'day'
                    },
                    title: {
                        display: true,
                        text: 'Date'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Price ($)'
                    }
                }
            }
        }
    });

    // Set current price next to S&P 500 name
    const priceSpan = document.getElementById('sp500-current-price');
    if (Array.isArray(data) && data.length > 0) {
        const last = data[data.length - 1];
        priceSpan.textContent = `($${last.Close.toFixed(2)})`;
        priceSpan.style.color = '#28a745';
        priceSpan.style.marginLeft = '18px';
        priceSpan.style.fontSize = 'inherit';
        priceSpan.style.verticalAlign = 'middle';
    } else {
        priceSpan.textContent = '';
    }
}

function updateSP500PredictionInfo(predictionData) {
    const predictionElement = document.getElementById('sp500-prediction-value');
    const predictions = predictionData.predictions;
    let latestHistoricalPrice = null;
    if (sp500Chart && sp500Chart.data && sp500Chart.data.datasets.length > 0) {
        const histData = sp500Chart.data.datasets[0].data;
        if (histData.length > 0) {
            latestHistoricalPrice = histData[histData.length - 1].y;
        }
    }
    predictionElement.innerHTML = `
        <p>5-Day Price Predictions:</p>
        ${predictions.map(pred => {
            let pct = '';
            if (latestHistoricalPrice !== null && latestHistoricalPrice !== 0) {
                const diff = pred.Close - latestHistoricalPrice;
                const pctDiff = (diff / latestHistoricalPrice) * 100;
                pct = ` <span style="color:${pctDiff >= 0 ? 'green' : 'red'}">(${pctDiff >= 0 ? '+' : ''}${pctDiff.toFixed(2)}%)</span>`;
            }
            return `<div>${pred.Date}: $${pred.Close.toFixed(2)}${pct}</div>`;
        }).join('')}
    `;
}

// S&P 500 time button listeners
document.querySelectorAll('.sp500-time-button').forEach(button => {
    button.addEventListener('click', (e) => {
        document.querySelectorAll('.sp500-time-button').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        const period = e.target.dataset.period;
        searchSP500();
    });
});

// Initial S&P 500 chart load
window.addEventListener('DOMContentLoaded', () => {
    searchSP500();
});

// Redirect logic for search bar
document.getElementById('go-stock-page').addEventListener('click', () => {
    const ticker = document.getElementById('ticker-search').value.trim().toUpperCase();
    if (ticker) {
        window.location.href = `/stock.html?ticker=${encodeURIComponent(ticker)}`;
    }
});
document.getElementById('ticker-search').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        const ticker = document.getElementById('ticker-search').value.trim().toUpperCase();
        if (ticker) {
            window.location.href = `/stock.html?ticker=${encodeURIComponent(ticker)}`;
        }
    }
});

// S&P 500 stock list logic
const sp500Stocks = [
    // Top 30 S&P 500 stocks by market cap (symbol, name, and approximate weight %)
    { symbol: "AAPL", name: "Apple Inc.", weight: 7.2 },
    { symbol: "MSFT", name: "Microsoft Corp.", weight: 6.8 },
    { symbol: "AMZN", name: "Amazon.com Inc.", weight: 3.3 },
    { symbol: "NVDA", name: "NVIDIA Corp.", weight: 3.2 },
    { symbol: "GOOGL", name: "Alphabet Inc. (Class A)", weight: 2.0 },
    { symbol: "GOOG", name: "Alphabet Inc. (Class C)", weight: 1.7 },
    { symbol: "META", name: "Meta Platforms Inc.", weight: 2.1 },
    { symbol: "BRK.B", name: "Berkshire Hathaway Inc.", weight: 1.7 },
    { symbol: "TSLA", name: "Tesla Inc.", weight: 1.6 },
    { symbol: "UNH", name: "UnitedHealth Group Inc.", weight: 1.2 },
    { symbol: "LLY", name: "Eli Lilly and Co.", weight: 1.1 },
    { symbol: "JPM", name: "JPMorgan Chase & Co.", weight: 1.2 },
    { symbol: "V", name: "Visa Inc.", weight: 1.1 },
    { symbol: "XOM", name: "Exxon Mobil Corp.", weight: 1.0 },
    { symbol: "JNJ", name: "Johnson & Johnson", weight: 1.0 },
    { symbol: "MA", name: "Mastercard Inc.", weight: 0.9 },
    { symbol: "AVGO", name: "Broadcom Inc.", weight: 0.9 },
    { symbol: "PG", name: "Procter & Gamble Co.", weight: 0.9 },
    { symbol: "HD", name: "Home Depot Inc.", weight: 0.8 },
    { symbol: "MRK", name: "Merck & Co. Inc.", weight: 0.8 },
    { symbol: "COST", name: "Costco Wholesale Corp.", weight: 0.8 },
    { symbol: "ABBV", name: "AbbVie Inc.", weight: 0.7 },
    { symbol: "CVX", name: "Chevron Corp.", weight: 0.7 },
    { symbol: "ADBE", name: "Adobe Inc.", weight: 0.7 },
    { symbol: "PEP", name: "PepsiCo Inc.", weight: 0.7 },
    { symbol: "WMT", name: "Walmart Inc.", weight: 0.7 },
    { symbol: "MCD", name: "McDonald's Corp.", weight: 0.6 },
    { symbol: "BAC", name: "Bank of America Corp.", weight: 0.6 },
    { symbol: "KO", name: "Coca-Cola Co.", weight: 0.6 },
    { symbol: "ACN", name: "Accenture Plc", weight: 0.6 }
    // ...add more if desired
];

let sp500ListOffset = 0;
const sp500ListPageSize = 10;

function renderSP500StockList() {
    const listDiv = document.getElementById('sp500-stock-list');
    // Clear listDiv and add header row
    listDiv.innerHTML = '';
    const header = document.createElement('div');
    header.style.display = 'flex';
    header.style.fontWeight = 'bold';
    header.style.gap = '8px';
    header.style.padding = '4px 0';
    header.style.borderBottom = '1px solid #eee';

    // Columns: Name, Symbol, Weight, Price
    const nameCol = document.createElement('div');
    nameCol.textContent = 'Company';
    nameCol.style.flex = '2';
    const symbolCol = document.createElement('div');
    symbolCol.textContent = 'Symbol';
    symbolCol.style.flex = '1';
    const weightCol = document.createElement('div');
    weightCol.textContent = 'Weight';
    weightCol.style.flex = '1';
    const priceCol = document.createElement('div');
    priceCol.textContent = 'Price';
    priceCol.style.flex = '1';

    header.appendChild(nameCol);
    header.appendChild(symbolCol);
    header.appendChild(weightCol);
    header.appendChild(priceCol);
    listDiv.appendChild(header);

    const slice = sp500Stocks.slice(sp500ListOffset, sp500ListOffset + sp500ListPageSize);
    slice.forEach(stock => {
        const row = document.createElement('div');
        row.style.display = 'flex';
        row.style.alignItems = 'center';
        row.style.gap = '8px';
        row.style.padding = '2px 0';

        // Company Name (with link)
        const nameDiv = document.createElement('div');
        nameDiv.style.flex = '2';
        const a = document.createElement('a');
        a.href = `/stock.html?ticker=${encodeURIComponent(stock.symbol)}`;
        a.textContent = stock.name;
        a.title = stock.name;
        nameDiv.appendChild(a);

        // Symbol
        const symbolDiv = document.createElement('div');
        symbolDiv.textContent = stock.symbol;
        symbolDiv.style.flex = '1';

        // Weight
        const weightDiv = document.createElement('div');
        weightDiv.textContent = stock.weight ? `${stock.weight.toFixed(2)}%` : '';
        weightDiv.style.flex = '1';

        // Price
        const priceDiv = document.createElement('div');
        priceDiv.style.flex = '1';
        const priceSpan = document.createElement('span');
        priceSpan.textContent = '...';
        priceSpan.style.color = '#28a745';
        priceSpan.style.fontSize = 'inherit';
        priceSpan.style.verticalAlign = 'middle';
        priceDiv.appendChild(priceSpan);

        // Fetch current price
        fetch(`/stock-data/${encodeURIComponent(stock.symbol)}?period=1w`)
            .then(res => res.json())
            .then(data => {
                if (Array.isArray(data) && data.length > 0) {
                    const last = data[data.length - 1];
                    priceSpan.textContent = `$${last.Close.toFixed(2)}`;
                } else {
                    priceSpan.textContent = '';
                }
            })
            .catch(() => {
                priceSpan.textContent = '';
            });

        row.appendChild(nameDiv);
        row.appendChild(symbolDiv);
        row.appendChild(weightDiv);
        row.appendChild(priceDiv);
        listDiv.appendChild(row);
    });
    // Always show next/previous buttons, but disable if not available
    const nextBtn = document.getElementById('sp500-next-btn');
    const prevBtn = document.getElementById('sp500-prev-btn');
    nextBtn.style.display = '';
    prevBtn.style.display = '';
    if (sp500ListOffset + sp500ListPageSize >= sp500Stocks.length) {
        nextBtn.disabled = true;
    } else {
        nextBtn.disabled = false;
    }
    if (sp500ListOffset === 0) {
        prevBtn.disabled = true;
    } else {
        prevBtn.disabled = false;
    }
}

// --- CRYPTOCURRENCY LIST LOGIC ---

const topCryptos = [
    { symbol: "BTC-USD", name: "Bitcoin" },
    { symbol: "ETH-USD", name: "Ethereum" },
    { symbol: "USDT-USD", name: "Tether" },
    { symbol: "BNB-USD", name: "BNB" },
    { symbol: "SOL-USD", name: "Solana" },
    { symbol: "XRP-USD", name: "XRP" },
    { symbol: "USDC-USD", name: "USD Coin" },
    { symbol: "DOGE-USD", name: "Dogecoin" },
    { symbol: "TON-USD", name: "Toncoin" },
    { symbol: "ADA-USD", name: "Cardano" }
];

function renderCryptoList() {
    // Create or get the crypto list container
    let cryptoSection = document.getElementById('crypto-list-section');
    if (!cryptoSection) {
        cryptoSection = document.createElement('div');
        cryptoSection.id = 'crypto-list-section';
        cryptoSection.className = 'sp500-stock-list-fullwidth';
        // Insert after S&P 500 stock list section
        const sp500Section = document.getElementById('sp500-stock-list-section');
        if (sp500Section && sp500Section.parentNode) {
            sp500Section.parentNode.insertBefore(cryptoSection, sp500Section.nextSibling);
        } else {
            document.body.appendChild(cryptoSection);
        }
    }
    // Main list container
    let listDiv = document.getElementById('crypto-list');
    if (!listDiv) {
        listDiv = document.createElement('div');
        listDiv.id = 'crypto-list';
        listDiv.className = 'sp500-stock-list';
        cryptoSection.appendChild(listDiv);
    }
    listDiv.innerHTML = '';

    // Header row (same style as S&P 500)
    const header = document.createElement('div');
    header.style.display = 'flex';
    header.style.fontWeight = 'bold';
    header.style.gap = '8px';
    header.style.padding = '4px 0';
    header.style.borderBottom = '1px solid #eee';

    const nameCol = document.createElement('div');
    nameCol.textContent = 'Cryptocurrency';
    nameCol.style.flex = '2';
    const symbolCol = document.createElement('div');
    symbolCol.textContent = 'Symbol';
    symbolCol.style.flex = '1';
    const priceCol = document.createElement('div');
    priceCol.textContent = 'Price';
    priceCol.style.flex = '1';

    header.appendChild(nameCol);
    header.appendChild(symbolCol);
    header.appendChild(priceCol);
    listDiv.appendChild(header);

    topCryptos.forEach(crypto => {
        const row = document.createElement('div');
        row.style.display = 'flex';
        row.style.alignItems = 'center';
        row.style.gap = '8px';
        row.style.padding = '2px 0';

        // Name (with link)
        const nameDiv = document.createElement('div');
        nameDiv.style.flex = '2';
        const a = document.createElement('a');
        a.href = `/stock.html?ticker=${encodeURIComponent(crypto.symbol)}`;
        a.textContent = crypto.name;
        a.title = crypto.name;
        nameDiv.appendChild(a);

        // Symbol
        const symbolDiv = document.createElement('div');
        symbolDiv.textContent = crypto.symbol.replace('-USD', '');
        symbolDiv.style.flex = '1';

        // Price
        const priceDiv = document.createElement('div');
        priceDiv.style.flex = '1';
        const priceSpan = document.createElement('span');
        priceSpan.textContent = '...';
        priceSpan.style.color = '#28a745';
        priceSpan.style.fontSize = 'inherit';
        priceSpan.style.verticalAlign = 'middle';
        priceDiv.appendChild(priceSpan);

        // Fetch current price
        fetch(`/stock-data/${encodeURIComponent(crypto.symbol)}?period=1d`)
            .then(res => res.json())
            .then(data => {
                if (Array.isArray(data) && data.length > 0) {
                    const last = data[data.length - 1];
                    priceSpan.textContent = `$${last.Close.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
                } else {
                    priceSpan.textContent = '';
                }
            })
            .catch(() => {
                priceSpan.textContent = '';
            });

        row.appendChild(nameDiv);
        row.appendChild(symbolDiv);
        row.appendChild(priceDiv);
        listDiv.appendChild(row);
    });
}

document.addEventListener('DOMContentLoaded', () => {
    renderSP500StockList();
    renderCryptoList();
    document.getElementById('sp500-next-btn').addEventListener('click', () => {
        sp500ListOffset += sp500ListPageSize;
        renderSP500StockList();
    });
    // Add previous button event
    document.getElementById('sp500-prev-btn').addEventListener('click', () => {
        sp500ListOffset = Math.max(0, sp500ListOffset - sp500ListPageSize);
        renderSP500StockList();
    });
});
