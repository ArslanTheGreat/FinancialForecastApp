# train_model.py
import numpy as np
import yfinance as yf
from newsapi import NewsApiClient
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

# Initialize News API client
newsapi = NewsApiClient(api_key='aa611326e3d24dc0b8fe7ab23fd7cb6c')

# Function to fetch sentiment data
def fetch_news_sentiments(company_name):
    articles = newsapi.get_everything(q=company_name, language='en', sort_by='relevancy')
    sentiments = []
    for article in articles['articles']:
        blob = TextBlob(article['title'])
        sentiments.append(blob.sentiment.polarity)
    avg_sentiment = np.mean(sentiments) if sentiments else 0
    return avg_sentiment

# Function to get stock data
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data['Return'] = stock_data['Close'].pct_change()
    return stock_data.dropna()

# Function to prepare data for LSTM
def prepare_data_for_lstm(stock_data, scaler):
    scaled_data = scaler.fit_transform(stock_data[['Return', 'Sentiment']])
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i, 0])  # Target is the 'Return' column
    return np.array(X), np.array(y)

# Main training function
def train_and_save_model():
    # List of companies to train on
    companies = ['GS', 'TMO', 'NTAP', 'COF']  # Add more companies as needed

    # Initialize data storage and scaler
    all_X, all_y = [], []
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Collect data for each company
    for company in companies:
        print(f"Processing {company}")
        stock_data = get_stock_data(company, '2020-01-01', '2023-01-01')
        sentiment = fetch_news_sentiments(company)
        stock_data['Sentiment'] = [sentiment] * len(stock_data)
        
        X, y = prepare_data_for_lstm(stock_data, scaler)
        all_X.append(X)
        all_y.append(y)
    
    # Combine data from all companies
    X_train = np.concatenate(all_X, axis=0)
    y_train = np.concatenate(all_y, axis=0)

    # Define and compile the LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Save model and scaler
    model.save('forecast_model.h5')
    joblib.dump(scaler, 'scaler.save')
    print("Model and scaler saved successfully.")
    

# Run the training function
if __name__ == '__main__':
    train_and_save_model()
