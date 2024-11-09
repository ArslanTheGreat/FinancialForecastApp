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

app = Flask(__name__)

# Load the pre-trained model and scaler
model = load_model('forecast_model.h5')
scaler = joblib.load('scaler.save')

# Initialize News API client
newsapi = NewsApiClient(api_key='aa611326e3d24dc0b8fe7ab23fd7cb6c')

# Function to fetch sentiment data
def fetch_news_sentiments(company_name):
    articles = newsapi.get_everything(q=company_name, language='en', sort_by='relevancy')
    sentiments = [TextBlob(article['title']).sentiment.polarity for article in articles['articles']]
    return np.mean(sentiments) if sentiments else 0

def get_stock_data(ticker, end_date=None):
    end_date = datetime.today() if end_date is None else datetime.strptime(end_date, '%Y-%m-%d')
    start_date = end_date - relativedelta(months=5)
    stock_data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    stock_data['Return'] = stock_data['Close'].pct_change()
    return stock_data.dropna()

def prepare_data_for_prediction(stock_data):
    scaled_data = scaler.transform(stock_data[['Return', 'Sentiment']])
    return np.array([scaled_data[-60:]])


def backtest_model(ticker_symbol, test_ratio=0.2):
    # Fetch stock data for the past few months
    stock_data = get_stock_data(ticker_symbol)
    sentiment = fetch_news_sentiments(ticker_symbol)

    # Add sentiment data
    stock_data[('Sentiment', '')] = [sentiment] * len(stock_data)

    # Split data into training and testing sets
    split_point = int(len(stock_data) * (1 - test_ratio))
    train_data = stock_data[:split_point]
    test_data = stock_data[split_point:]

    # Prepare training data
    X_train = prepare_data_for_prediction(train_data)
    y_train = train_data[('Close', ticker_symbol)].values

    # Prepare testing data
    X_test = prepare_data_for_prediction(test_data)
    y_test = test_data[('Close', ticker_symbol)].values

    # Train the model on the training set
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    # Predict on the test set
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(np.array([predictions.flatten(), np.zeros(len(predictions))]).T)[:, 0]

    # Calculate error metrics
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    # Print and return accuracy metrics
    accuracy_metrics = {'MAE': mae, 'MSE': mse, 'RMSE': rmse}
    print("Backtesting Results:", accuracy_metrics)
    return accuracy_metrics

@app.route('/forecast', methods=['POST'])
def forecast():
    data = request.get_json()
    company_name = data['company'].strip().lower()
    
    # Convert company name to ticker symbol using the mapping
    if company_name in COMPANY_TICKER_MAPPING:
        ticker_symbol = COMPANY_TICKER_MAPPING[company_name]
    else:
        return jsonify({"error": f"Company '{company_name}' not found."}), 400

    # Fetch stock data for the past 5 months
    stock_data = get_stock_data(ticker_symbol)
    
    # Access historical "Close" prices
    historical_prices = stock_data[('Close', ticker_symbol)].tolist() if ('Close', ticker_symbol) in stock_data.columns else []

    # Fetch sentiment and add it to stock_data
    sentiment = fetch_news_sentiments(company_name)
    stock_data[('Sentiment', '')] = [sentiment] * len(stock_data)
    
    # Prepare data and make predictions
    X = prepare_data_for_prediction(stock_data)
    one_day_forecast = scaler.inverse_transform([[model.predict(X)[0][0], 0]])[0][0]
    one_week_forecast = one_day_forecast * 5
    one_month_forecast = one_day_forecast * 20

    # Generate a 1-month forecast
    forecasted_prices = [historical_prices[-1]]
    for _ in range(20):
        forecasted_prices.append(forecasted_prices[-1] * (1 + one_day_forecast / 100))

    # Prepare JSON response
    return jsonify({
        'company': company_name.title(),
        'one_day_forecast': one_day_forecast,
        'one_week_forecast': one_week_forecast,
        'one_month_forecast': one_month_forecast,
        'historical_dates': stock_data.index.strftime('%Y-%m-%d').tolist(),
        'historical_prices': historical_prices,
        'forecasted_prices': forecasted_prices[1:]  # Exclude initial historical price
    })

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

    # Run Backtesting
    backtest_model('AAPL')