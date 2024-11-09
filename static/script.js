async function getForecast() {
    const companyName = document.getElementById('companyName').value;
    if (!companyName) {
        alert("Please enter a company name.");
        return;
    }

    const response = await fetch('/forecast', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ company: companyName })
    });

    const data = await response.json();

    // Display forecasted returns
    document.getElementById('forecastResult').innerHTML = `
        <h3>Forecast for ${data.company}:</h3>
        <p><strong>1-Day Forecast:</strong> ${data.one_day_forecast.toFixed(2)}%</p>
        <p><strong>1-Week Forecast:</strong> ${data.one_week_forecast.toFixed(2)}%</p>
        <p><strong>1-Month Forecast:</strong> ${data.one_month_forecast.toFixed(2)}%</p>
    `;

    // Display chart
    displayChart(data.historical_dates, data.historical_prices, data.forecasted_prices);
}

function displayChart(dates, prices, forecastedPrices) {
    const ctx = document.getElementById('forecastChart').getContext('2d');

    if (window.myChart) {
        window.myChart.destroy();
    }

    const forecastLabels = Array.from({ length: forecastedPrices.length }, (_, i) => `Forecast Day ${i + 1}`);
    const allLabels = dates.concat(forecastLabels);
    const lastHistoricalPrice = prices[prices.length - 1];
    const finalForecastedPrice = forecastedPrices[forecastedPrices.length - 1];
    const forecastColor = (finalForecastedPrice >= lastHistoricalPrice) ? 'rgba(0, 200, 0, 1)' : 'rgba(255, 0, 0, 1)';

    window.myChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: allLabels,
            datasets: [
                {
                    label: 'Historical Price',
                    data: prices,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    fill: false,
                    tension: 0.1
                },
                {
                    label: '1-Month Forecasted Price',
                    data: [...Array(dates.length).fill(null), ...forecastedPrices],
                    borderColor: forecastColor,
                    fill: false,
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: false,  // Disable responsiveness to keep fixed size
            scales: {
                x: {
                    title: { display: true, text: 'Date' }
                },
                y: {
                    title: { display: true, text: 'Stock Price (USD)' }
                }
            }
        }
    });
    
}