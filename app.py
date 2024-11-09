from flask import Flask, request, jsonify, render_template
from newsapi import NewsApiClient
from textblob import TextBlob
import yfinance as yf
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load both pre-trained models and the scaler
model1 = load_model('forecast_model1.h5')
model2 = load_model('forecast_model2.h5')
scaler = joblib.load('scaler.save')

# Mapping from company names to ticker symbols
COMPANY_TICKER_MAPPING = {
    'apple': 'AAPL',
    'google': 'GOOGL',
    'goldman sachs': 'GS',
    'microsoft': 'MSFT',
    'amazon': 'AMZN',
    'tesla': 'TSLA',
    'meta': 'META',
    'nvidia': 'NVDA',
    'thermo fisher': 'TMO',
    'netapp': 'NTAP',
    'capital one': 'COF'
}

# Initialize News API client
newsapi = NewsApiClient(api_key='aa611326e3d24dc0b8fe7ab23fd7cb6c')  # Replace with your actual API key

# Function to fetch sentiment data
def fetch_news_sentiments(company_name):
    try:
        articles = newsapi.get_everything(q=company_name, language='en', sort_by='relevancy')
        sentiments = [TextBlob(article['title']).sentiment.polarity for article in articles['articles']]
        return np.mean(sentiments) if sentiments else 0
    except Exception as e:
        print(f"Error fetching news sentiments: {e}")
        return 0

# Function to get stock data
def get_stock_data(ticker, end_date=None):
    end_date = datetime.today() if end_date is None else datetime.strptime(end_date, '%Y-%m-%d')
    start_date = end_date - relativedelta(years=5)  # Adjust years as needed
    stock_data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    
    # Compute additional features
    stock_data['Return'] = stock_data['Close'].pct_change()
    stock_data['MA20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['Volatility'] = stock_data['Close'].rolling(window=20).std()
    stock_data.dropna(inplace=True)  # Drop any rows with NaN values
    return stock_data

# Prepare data for LSTM (same as in train_model.py)
def prepare_data_for_lstm(stock_data, scaler):
    # Scale the features
    scaled_data = scaler.transform(stock_data[['Return', 'Sentiment', 'MA20', 'MA50', 'Volatility']])
    X, y = [], []
    for i in range(60, len(scaled_data) - 20):
        X.append(scaled_data[i-60:i])  # Last 60 days as input sequence
        
        # Targets for 1-day, 1-week, and 1-month horizons
        one_day_target = stock_data['Return'].values[i + 1]
        one_week_target = np.sum(stock_data['Return'].values[i + 1:i + 6])  # Sum of returns over 5 days
        one_month_target = np.sum(stock_data['Return'].values[i + 1:i + 21])  # Sum of returns over 20 days
        y.append([one_day_target, one_week_target, one_month_target])  # Multiple targets per sample
    return np.array(X), np.array(y)


# Forecasting endpoint for real-time forecasts
@app.route('/forecast', methods=['POST'])
def forecast():
    data = request.get_json()
    company_name = data['company'].strip().lower()

    # Convert company name to ticker symbol
    ticker_symbol = COMPANY_TICKER_MAPPING.get(company_name)
    if not ticker_symbol:
        return jsonify({"error": f"Company '{company_name}' not found."}), 400

    # Fetch stock and sentiment data
    stock_data = get_stock_data(ticker_symbol)
    sentiment = fetch_news_sentiments(company_name)
    stock_data['Sentiment'] = sentiment

    # Ensure we have at least 60 rows for the look-back period
    if len(stock_data) < 60:
        print("Error: Not enough data for 60-day look-back window.")
        return jsonify({"error": "Not enough data available for forecasting"}), 400

    # Take the last 60 rows and scale them
    try:
        scaled_data = scaler.transform(stock_data[['Return', 'Sentiment', 'MA20', 'MA50', 'Volatility']].tail(60))
        X_forecast = np.array([scaled_data])  # Reshape directly into the required 3D shape
    except Exception as e:
        print(f"Error during scaling or data preparation: {e}")
        return jsonify({"error": "Data preparation failed"}), 500

    print("Shape of X_forecast after preparation:", X_forecast.shape)

    # Predictions from both models
    try:
        forecast1 = model1.predict(X_forecast)[0]
        forecast2 = model2.predict(X_forecast)[0]
        print("Forecast from model1:", forecast1)
        print("Forecast from model2:", forecast2)
    except Exception as e:
        print("Error during model prediction:", e)
        return jsonify({"error": "Error in model prediction"}), 500

    # Average the forecasts for final output
    forecast = (forecast1 + forecast2) / 2
    one_day_forecast = float(forecast[0] / 150 * 100)
    one_week_forecast = float(forecast[1] / 150 * 100)
    one_month_forecast = float(forecast[2] / 150 * 100)

    ## Ensure 'Close' column exists, and handle cases where it doesn’t
    historical_prices = stock_data.get('Close')
    if historical_prices is not None:
        # Use .iloc to safely extract as a Series in case of a multi-index or unexpected format
        historical_prices = stock_data['Close'].iloc[:, 0] if isinstance(stock_data['Close'], pd.DataFrame) else stock_data['Close']
        historical_prices = historical_prices.tolist()
    else:
        historical_prices = []  # Fallback if 'Close' column is missing

    # Fix for forecasted prices calculation
    if historical_prices:  # Check if there is at least one historical price
        if isinstance(historical_prices[-1], list):
            last_price = historical_prices[-1][0]  # Access the first element if it's a list
        else:
            last_price = historical_prices[-1]  # Access the last price if it's a scalar
    else:
        last_price = 0.0  # Default to 0 if no historical prices are available

    # Initialize forecasted prices with the last historical price
    forecasted_prices = [float(last_price)]
    growth_rate = 1 + (one_month_forecast / 100) / 20  # Approximate daily growth rate

    # Generate forecasted prices for 5 months (approximately 100 trading days)
    for i in range(1, 101):
        forecasted_prices.append(forecasted_prices[-1] * growth_rate)

    # Prepare response with historical data
    return jsonify({
        'company': company_name.title(),
        'one_day_forecast': one_day_forecast,
        'one_week_forecast': one_week_forecast,
        'one_month_forecast': one_month_forecast,
        'historical_dates': stock_data.index.strftime('%Y-%m-%d').tolist(),
        'historical_prices': historical_prices,
        'forecasted_prices': forecasted_prices[1:]  # Exclude the initial historical price
    })



# Add the backtesting function here as well, using `prepare_data_for_lstm`
def backtest_model(ticker_symbol, test_ratio=0.1):
    print("Starting backtest for:", ticker_symbol)

    # Fetch stock data for the past few years
    stock_data = get_stock_data(ticker_symbol)
    if stock_data.empty:
        print("No stock data found for", ticker_symbol)
        return

    sentiment = fetch_news_sentiments(ticker_symbol)
    stock_data['Sentiment'] = [sentiment] * len(stock_data)

    # Split data into training and testing sets
    split_point = int(len(stock_data) * (1 - test_ratio))
    train_data = stock_data[:split_point]
    test_data = stock_data[split_point:]

    # Prepare training data
    X_train, y_train = prepare_data_for_lstm(train_data, scaler)

    # Prepare testing data
    X_test, y_test = prepare_data_for_lstm(test_data, scaler)

    # Predict on the test set using both models and average predictions
    predictions1 = model1.predict(X_test).flatten()
    predictions2 = model2.predict(X_test).flatten()
    predictions = (predictions1 + predictions2) / 2

    # Calculate error metrics
    mae = mean_absolute_error(y_test.flatten(), predictions)
    mse = mean_squared_error(y_test.flatten(), predictions)
    rmse = np.sqrt(mse)

    # Print and return accuracy metrics
    accuracy_metrics = {'MAE': mae, 'MSE': mse, 'RMSE': rmse}
    print("Backtesting Results:", accuracy_metrics)
    return accuracy_metrics

@app.route('/')
def home():
    return render_template('index.html')

# Run the Flask app and backtest
if __name__ == '__main__':
    ticker = 'AAPL'
    backtest_model(ticker)
    app.run(debug=True)
