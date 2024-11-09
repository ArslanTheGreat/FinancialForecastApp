# train_model.py
import numpy as np
import yfinance as yf
from newsapi import NewsApiClient
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import joblib
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd

# Initialize News API client
newsapi = NewsApiClient(api_key='aa611326e3d24dc0b8fe7ab23fd7cb6c')  # Replace with your actual API key

# Function to fetch sentiment data
def fetch_news_sentiments(company_name):
    articles = newsapi.get_everything(q=company_name, language='en', sort_by='relevancy')
    sentiments = [TextBlob(article['title']).sentiment.polarity for article in articles['articles']]
    return np.mean(sentiments) if sentiments else 0

# Function to get stock data
def get_stock_data(ticker, start_date=None, end_date=None):
    end_date = datetime.today() if end_date is None else datetime.strptime(end_date, '%Y-%m-%d')
    start_date = end_date - relativedelta(years=5) if start_date is None else datetime.strptime(start_date, '%Y-%m-%d')
    stock_data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    stock_data['Return'] = stock_data['Close'].pct_change()
    stock_data['MA20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['Volatility'] = stock_data['Close'].rolling(window=20).std()
    return stock_data.dropna()

# Function to prepare data for LSTM
def prepare_data_for_lstm(stock_data, scaler):
    # Check if stock_data has at least 60 rows for the look-back window
    if len(stock_data) < 60:
        print("Error: Not enough data in stock_data for 60-day look-back window.")
        return np.array([]), np.array([])  # Return empty arrays if data is insufficient

    # Scale the features
    scaled_data = scaler.transform(stock_data[['Return', 'Sentiment', 'MA20', 'MA50', 'Volatility']])
    X, y = [], []
    for i in range(60, len(scaled_data) - 20):  # Stop 20 days before the end for the monthly target
        X.append(scaled_data[i-60:i])  # Last 60 days as input sequence
        
        # Targets for 1-day, 1-week, and 1-month horizons
        one_day_target = stock_data['Return'].values[i + 1]
        one_week_target = np.sum(stock_data['Return'].values[i + 1:i + 6])  # Sum of returns over 5 days
        one_month_target = np.sum(stock_data['Return'].values[i + 1:i + 21])  # Sum of returns over 20 days
        y.append([one_day_target, one_week_target, one_month_target])  # Multiple targets per sample

    # Convert lists to numpy arrays
    X, y = np.array(X), np.array(y)

    # Log shapes for debugging
    print("Shape of X in prepare_data_for_lstm:", X.shape)
    print("Shape of y in prepare_data_for_lstm:", y.shape)

    return X, y


# Function to build the two models for ensemble
def build_model_1(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dense(25),
        Dense(3)  # Outputs for 1-day, 1-week, 1-month forecasts
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')
    return model

def build_model_2(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(3)  # Outputs for 1-day, 1-week, 1-month forecasts
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')
    return model

# Main training function
def train_and_save_models():
    companies = ['GS', 'TMO', 'NTAP', 'COF', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    all_stock_data = []
    for company in companies:
        stock_data = get_stock_data(company, '2020-01-01', '2023-01-01')
        sentiment = fetch_news_sentiments(company)
        stock_data['Sentiment'] = [sentiment] * len(stock_data)
        all_stock_data.append(stock_data)

    # Combine all collected data into a single DataFrame for scaler fitting
    combined_stock_data = pd.concat(all_stock_data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(combined_stock_data[['Return', 'Sentiment', 'MA20', 'MA50', 'Volatility']])

    # Prepare training data
    all_X, all_y = [], []
    for stock_data in all_stock_data:
        X, y = prepare_data_for_lstm(stock_data, scaler)
        all_X.append(X)
        all_y.append(y)

    X_train = np.concatenate(all_X, axis=0)
    y_train = np.concatenate(all_y, axis=0)

    # Initialize both models and early stopping
    model1 = build_model_1((X_train.shape[1], X_train.shape[2]))
    model2 = build_model_2((X_train.shape[1], X_train.shape[2]))
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    # Train both models
    model1.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, callbacks=[early_stopping])
    model2.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, callbacks=[early_stopping])

    # Save both models and the scaler
    model1.save('forecast_model1.h5')
    model2.save('forecast_model2.h5')
    joblib.dump(scaler, 'scaler.save')
    print("Models and scaler saved successfully.")

if __name__ == '__main__':
    train_and_save_models()
