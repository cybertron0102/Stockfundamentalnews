import streamlit as st
import pandas as pd
import numpy as np
import csv
import re
import time
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import datetime
import warnings
import plotly.graph_objects as go
import os
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# Hide warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK data packages
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('vader_lexicon')

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'[^\w\s]','',text)
    words = nltk.word_tokenize(text)
    words = [w for w in words if not w in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def is_stock_mentioned(text, stock_name):
    return stock_name.lower() in text.lower()

# Streamlit app
st.title("Stock Analysis Dashboard")

# Get user input for ticker symbol and company name
tickersymbol = st.text_input("Enter the ticker symbol:")
company_name = st.text_input("Enter the company name for sentiment analysis:")

if not tickersymbol or not company_name:
    st.warning("Please provide both the ticker symbol and the company name.")
    st.stop()

try:
    # Initialize yfinance data
    data = yf.Ticker(tickersymbol)

    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=365)

    # Check if the data file exists
    filename = f"{tickersymbol}_data.csv"
    if os.path.exists(filename):
        # If the data file exists, load it
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
    else:
        # If the data file doesn't exist, download the data and save it to a file
        df = data.history(start=start_date, end=end_date)
        df.to_csv(filename)

    # Plot historical prices
    st.subheader("Historical Prices")
    st.plotly_chart(go.Figure(data=go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'),
                              layout=dict(title='Historic chart of the stock', xaxis_title='Date', yaxis_title='Price')))

    # Calculate returns
    df['High Returns'] = df['High'].pct_change().dropna()
    df['Low Returns'] = df['Low'].pct_change().dropna()
    df['Close Returns'] = df['Close'].pct_change().dropna()
    df['Open Returns'] = df['Open'].pct_change().dropna()
    df['Volume Changes'] = df['Volume'].pct_change().dropna()

    # Drop missing values
    df = df.dropna()

    # Create and fit ARIMA models
    models = {}
    for column in ['High Returns', 'Low Returns', 'Close Returns', 'Open Returns', 'Volume Changes']:
        try:
            model = ARIMA(df[column], order=(5, 1, 4))  # Adjust the order parameters as needed
            fitted = model.fit()
            models[column] = fitted
        except Exception as e:
            st.warning(f"Failed to fit ARIMA model for {column}: {e}")

    if not models:
        st.error("Failed to fit ARIMA models. Please check the data and try again.")
        st.stop()

    # Predict future prices
    def predict_future_days(models, days):
        future_returns = {column: models[column].predict(start=len(df), end=len(df)+days-1) for column in models}
        last_price = df['Close'][-1]  # Get the last price
        future_prices = [last_price]
        for i in range(days):
            future_price = future_prices[-1]*(1+future_returns['Close Returns'].iloc[i])
            future_prices.append(future_price)
        return future_prices

    # Define time frames
    time_frames = [
        (7, '1 week'),
        (14, '2 weeks'),
        # Add more time frames as needed
    ]

    # Get user choice for time frame
    time_frame_choice = st.selectbox("Select a time frame:", options=[label for _, label in time_frames])

    # Calculate support and resistance levels
    days = [days for days, label in time_frames if label == time_frame_choice][0]
    df_resampled = df.resample(f'{days}D').agg({'High': 'max', 'Low': 'min'})
    df['High_RollingMax'] = df_resampled['High'].reindex(df.index, method='ffill')
    df['Low_RollingMin'] = df_resampled['Low'].reindex(df.index, method='ffill')

    # Predict future prices for selected time frame
    predicted_prices = predict_future_days(models, days)

    # Print predicted prices
    st.subheader("Predicted Prices")
    st.write("The predicted prices are: ", predicted_prices[-3:])

    # Plot actual and predicted prices with support and resistance levels
    fig = go.Figure()

    # Actual prices
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Actual Price'))

    # Predicted prices
    predicted_dates = pd.date_range(start=df.index[-1], periods=len(predicted_prices)+1)[1:]
    fig.add_trace(go.Scatter(x=predicted_dates, y=predicted_prices, mode='lines', name='Predicted Price'))

    # Support and resistance levels
    future_support = [df['Low_RollingMin'][-1]] * len(predicted_dates)
    future_resistance = [df['High_RollingMax'][-1]] * len(predicted_dates)
    fig.add_trace(go.Scatter(x=predicted_dates, y=future_resistance, mode='lines', name='Resistance Level'))
    fig.add_trace(go.Scatter(x=predicted_dates, y=future_support, mode='lines', name='Support Level'))

    fig.update_layout(title='Stock Prices with Support and Resistance Levels', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

    # Sentiment Analysis
    url = "https://www.moneycontrol.com/news/business/stocks/"
    scraped_data = set()

    # Initialize the sentiment intensity analyzer
    sia = SentimentIntensityAnalyzer()

    # Define a function to classify sentiment
    def classify_sentiment(scores):
        if scores['compound'] > 0.05:
            return 'bullish'
        elif scores['compound'] < -0.05:
            return 'bearish'
        else:
            return 'neutral'

    # Streamlit section for sentiment analysis
    st.subheader("Sentiment Analysis")
    st.write("Analyzing sentiment for news related to", company_name)

    while True:
        webpage = requests.get(url)
        if webpage.status_code == 200:  # request successful
            soup = BeautifulSoup(webpage.content, 'html.parser')
            news_items = soup.find_all('li', class_='clearfix')

            for news_item in news_items:
                title = preprocess_text(news_item.find('h2').find('a').get('title'))
                paragraph = preprocess_text(news_item.find('p').text.strip())

                if is_stock_mentioned(title, company_name) or is_stock_mentioned(paragraph, company_name):
                    sentiment_scores = sia.polarity_scores(paragraph)
                    sentiment = classify_sentiment(sentiment_scores)

                    if (title, paragraph, str(sentiment_scores), sentiment) not in scraped_data:
                        st.write("Title:", title)
                        st.write("Paragraph:", paragraph)
                        st.write("Sentiment Scores:", sentiment_scores)
                        st.write("Sentiment:", sentiment)
                        st.write("\n")

                        scraped_data.add((title, paragraph, str(sentiment_scores), sentiment))
        else:
            st.write("Failed to retrieve the page. Status code:", webpage.status_code)

        time.sleep(60)

except Exception as e:
    st.error(f"An error occurred: {e}")