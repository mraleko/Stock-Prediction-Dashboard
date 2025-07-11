let stockChart = null;

function getTickerFromURL() {
    const params = new URLSearchParams(window.location.search);
    return params.get('ticker') || '';
}

async function searchStock() {
    const ticker = document.getElementById('ticker-search').value.toUpperCase();
    const period = document.querySelector('.time-button.active')?.dataset.period || '1y';

    let url = `/stock-data/${ticker}?period=${period}`;

    try {
        const [stockResponse, predictionResponse] = await Promise.all([
            fetch(url),
            fetch(`/predict/${ticker}`)
        ]);
        
        const stockData = await stockResponse.json();
        const predictionData = await predictionResponse.json();
        
        updateChart(stockData, predictionData);
        updatePredictionInfo(predictionData);

        // Set stock title and current price
        document.getElementById('stock-title').textContent = `${ticker}`;
        let priceSpan = document.getElementById('current-price');
        if (!priceSpan) {
            priceSpan = document.createElement('span');
            priceSpan.id = 'current-price';
            // Set color to match historical price (#28a745)
            priceSpan.style.color = '#28a745';
            priceSpan.style.marginLeft = '18px';
            document.getElementById('stock-title').appendChild(priceSpan);
        } else {
            // Ensure color is set even if span already exists
            priceSpan.style.color = '#28a745';
        }
        if (Array.isArray(stockData) && stockData.length > 0) {
            const last = stockData[stockData.length - 1];
            priceSpan.textContent = `($${last.Close.toFixed(2)})`;
        } else {
            priceSpan.textContent = '';
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error fetching stock data. Please try again.');
    }
}

function updateChart(data, predictionData) {
    const ctx = document.getElementById('stockChart').getContext('2d');
    
    if (stockChart) {
        stockChart.destroy();
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
    
    stockChart = new Chart(ctx, {
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
                    text: 'Stock Price History and Prediction'
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
}

function updatePredictionInfo(predictionData) {
    const predictionElement = document.getElementById('prediction-value');
    const predictions = predictionData.predictions;
    let latestHistoricalPrice = null;
    if (stockChart && stockChart.data && stockChart.data.datasets.length > 0) {
        const histData = stockChart.data.datasets[0].data;
        if (histData.length > 0) {
            latestHistoricalPrice = histData[histData.length - 1].y;
        }
    }
    // Show current predictions
    let html = `
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

    // Show prediction history with actuals
    if (predictionData.prediction_history && predictionData.prediction_history.length > 0) {
        html += `<hr><p><strong>Past Predictions vs Actuals:</strong></p>`;
        html += `<table style="width:100%;font-size:13px;"><tr><th style="text-align:left">Date</th><th style="text-align:left">Predicted</th><th style="text-align:left">Actual</th><th style="text-align:left">Error</th></tr>`;
        // Deduplicate: keep only the latest prediction per date
        const deduped = {};
        for (const entry of predictionData.prediction_history) {
            // If not present or this entry is newer, keep it
            if (
                !deduped[entry.Date] ||
                (entry.timestamp && deduped[entry.Date].timestamp && entry.timestamp > deduped[entry.Date].timestamp)
            ) {
                deduped[entry.Date] = entry;
            }
        }
        // Sort by date ascending
        const sorted = Object.values(deduped).sort((a, b) => a.Date.localeCompare(b.Date));
        // Only keep the last 10 days of *history* (exclude future prediction dates)
        const today = new Date();
        const filtered = sorted.filter(entry => {
            const entryDate = new Date(entry.Date);
            return entryDate <= today;
        });
        let last10 = filtered;
        if (filtered.length > 10) {
            last10 = filtered.slice(-10);
        }
        let prevActual = null;
        for (const entry of last10) {
            let err = '';
            if (entry.actual_close !== null && entry.actual_close !== undefined) {
                const diff = entry.predicted_close - entry.actual_close;
                const pct = (diff / entry.actual_close) * 100;
                err = `${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%`;
            }
            // Compute % change from previous actual close to current actual close
            let actualPctChange = '';
            if (
                prevActual !== null &&
                entry.actual_close !== null &&
                entry.actual_close !== undefined &&
                prevActual !== undefined &&
                prevActual !== null
            ) {
                const pctVal = ((entry.actual_close - prevActual) / prevActual) * 100;
                actualPctChange = ` <span style="color:${pctVal >= 0 ? 'green' : 'red'}">${pctVal >= 0 ? '+' : ''}${pctVal.toFixed(2)}%</span>`;
            } else if (
                entry.pct_change_from_last !== null &&
                entry.pct_change_from_last !== undefined
            ) {
                const pctVal = entry.pct_change_from_last;
                actualPctChange = ` <span style="color:${pctVal >= 0 ? 'green' : 'red'}">${pctVal >= 0 ? '+' : ''}${pctVal.toFixed(2)}%</span>`;
            }
            // Predicted % change from previous day's historical close
            let predPctChange = '';
            if (
                entry.pct_change_from_last !== null &&
                entry.pct_change_from_last !== undefined
            ) {
                const pctVal = entry.pct_change_from_last;
                predPctChange = ` <span style="color:${pctVal >= 0 ? 'green' : 'red'}">${pctVal >= 0 ? '+' : ''}${pctVal.toFixed(2)}%</span>`;
            }
            html += `<tr>
                <td>${entry.Date}</td>
                <td>$${entry.predicted_close !== undefined ? entry.predicted_close.toFixed(2) : ''}${predPctChange}</td>
                <td>${entry.actual_close !== null && entry.actual_close !== undefined ? '$' + entry.actual_close.toFixed(2) : '-'}${actualPctChange}</td>
                <td>${err}</td>
            </tr>`;
            // Update prevActual for next row
            if (entry.actual_close !== null && entry.actual_close !== undefined) {
                prevActual = entry.actual_close;
            }
        }
        html += `</table>`;
    }

    predictionElement.innerHTML = html;
}

window.addEventListener('DOMContentLoaded', () => {
    const ticker = getTickerFromURL();
    if (ticker) {
        document.getElementById('ticker-search').value = ticker;
        searchStock();
    }
    document.getElementById('search-stock-btn').addEventListener('click', () => {
        const t = document.getElementById('ticker-search').value.trim().toUpperCase();
        if (t) {
            window.location.href = `/stock.html?ticker=${encodeURIComponent(t)}`;
        }
    });
    document.getElementById('ticker-search').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const t = document.getElementById('ticker-search').value.trim().toUpperCase();
            if (t) {
                window.location.href = `/stock.html?ticker=${encodeURIComponent(t)}`;
            }
        }
    });
    document.querySelectorAll('.time-button').forEach(button => {
        button.addEventListener('click', (e) => {
            document.querySelectorAll('.time-button').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            searchStock();
        });
    });
});
