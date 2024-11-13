Financial Forecasting Application

This project is a machine learning-driven financial forecasting web application built with Flask. It leverages LSTM models, historical stock data, and sentiment analysis to predict short-term (1-day), medium-term (1-week), and longer-term (1-month) stock price movements for specified companies. The application provides predictions and backtests the model’s accuracy based on historical data, allowing users to compare forecasted prices against real historical data.

Project Structure
The project files include:

app.py: The main Flask application file that:
Serves as the web server.
Accepts user requests for stock forecasts.
Prepares input data, runs the forecasting model, and returns forecast results.
Integrates NewsAPI for sentiment analysis.

train_model.py: Script used for training the LSTM models:
Prepares historical stock and sentiment data.
Trains two LSTM models (forecast_model1.h5 and forecast_model2.h5) on different hyperparameters to create an ensemble for final predictions.
Saves the trained models and scaler for use in app.py.

Model Files (forecast_model.h5, forecast_model1.h5, forecast_model2.h5): Pre-trained LSTM models for forecasting stock returns.
Scaler (scaler.save): MinMaxScaler instance used to normalize input features during both training and inference.

required.txt: Lists dependencies for the project (Flask, NewsAPI, TensorFlow, etc.)​(required).

README.md: Project documentation.
Setup and Installation
To set up the project, follow these steps:

Clone the Repository:
bash
Copy code
git clone https://github.com/YourUsername/financial-forecasting-app.git
cd financial-forecasting-app


Install Dependencies: Use pip to install the dependencies listed in required.txt:
bash
Copy code
pip install -r required.txt


Set Up API Keys: Obtain an API key from NewsAPI and update app.py:
python
Copy code
newsapi = NewsApiClient(api_key='YOUR_NEWSAPI_KEY')


Run the Application:
bash
Copy code
python app.py


Access the Application: Open a browser and go to http://127.0.0.1:5000/.

Key Features and Workflow

Model Training (train_model.py)

Data Collection: Fetches five years of historical stock data using Yahoo Finance (yfinance library) and calculates features like moving averages (MA20, MA50) and volatility.
Sentiment Analysis: Extracts sentiment scores using NewsAPI and TextBlob on recent articles related to each company.
Data Preparation: Constructs training data with a look-back period of 60 days.
Model Building: Creates two LSTM models to capture different aspects of price movement and trains them with early stopping to avoid overfitting.
Saving Models: After training, saves the models and scaler for inference in the main application.

Forecasting (app.py)

Data Preparation:

Fetches the past 5 years of stock data.
Extracts recent sentiment scores using NewsAPI.
Prepares data for LSTM input with a 60-day look-back.
Ensemble Prediction:
Runs predictions with both trained LSTM models and averages their results to improve reliability.
Historical and Forecast Data:
Returns forecasted values (1-day, 1-week, and 1-month) and historical stock prices for comparison.
Backtesting: Compares predicted values with actual historical data to validate model accuracy.

Web Interface

The application allows users to enter a company name (mapped to its stock ticker symbol) and receive:
A 1-day, 1-week, and 1-month stock price forecast.
A line graph displaying historical prices alongside forecasted prices for comparison.

Future Improvements

This application could be enhanced by:
Expanding Predictive Horizons: Explore multi-month or annual predictions by adjusting the LSTM model’s architecture and input features.
Dynamic News Analysis: Increase the number of articles or integrate additional news sources to provide a more robust sentiment score.
More Sophisticated Feature Engineering: Implement technical indicators like RSI (Relative Strength Index) or MACD (Moving Average Convergence Divergence) to enhance predictive accuracy.
Alternative Models: Experiment with other neural network architectures (e.g., GRUs, Transformers) or ensemble methods.
Hyperparameter Tuning: Automate hyperparameter optimization to ensure optimal model configurations.
Limitations
Short Forecast Horizons: The LSTM model’s predictive power diminishes over long time frames, making it ideal for short-term forecasting only.
Reliance on NewsAPI: Sentiment analysis relies heavily on the frequency and relevance of news articles, which may not fully capture public sentiment.

Dependencies

This project requires the following libraries, as specified in required.txt:
Flask, NewsAPI, TextBlob, yfinance, TensorFlow, NumPy, scikit-learn.

Acknowledgments
Special thanks to NewsAPI for providing article data and to Yahoo Finance for stock price information.

