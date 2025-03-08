import numpy as np
import streamlit as st
import tensorflow as tf
import pandas as pd
import ta
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import plotly.graph_objects as go
from transformers import pipeline

# Load the trained model
model = tf.keras.models.load_model("stock_prediction_model.keras")

# Load TF-IDF Vectorizer & Technical Indicators Scaler used during training
vectorizer = joblib.load("tfidf_vectorizer.pkl")
tech_scaler = joblib.load("tech_scaler.pkl")  # Scaler fitted on 10 features

# Initialize Sentiment Analysis Pipeline (FinBERT)
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1)

# Load historical stock data from CSV
@st.cache_data
def load_stock_data():
    df = pd.read_csv("./Data/stock_prices.csv", parse_dates=["Date"])
    df["Stock_Price"] = pd.to_numeric(df["Stock_Price"], errors="coerce")
    df.dropna(subset=["Stock_Price"], inplace=True)
    return df

stock_data = load_stock_data()

# Compute technical indicators from the latest stock data and a provided current price,
# including the sentiment score as the first feature.
def compute_indicators(stock_data, current_price, sentiment):
    # Use the last 29 days and append a new row with the current price
    stock_data_recent = stock_data.tail(29).copy()
    new_data = pd.DataFrame({
        "Date": [stock_data_recent["Date"].iloc[-1] + pd.Timedelta(days=1)],
        "Stock_Price": [current_price]
    })
    stock_data_recent = pd.concat([stock_data_recent, new_data], ignore_index=True)

    # Calculate technical indicators for the new row
    stock_data_recent["RSI"] = ta.momentum.RSIIndicator(close=stock_data_recent["Stock_Price"], window=14).rsi()
    macd = ta.trend.MACD(close=stock_data_recent["Stock_Price"])
    stock_data_recent["MACD"] = macd.macd()
    stock_data_recent["MACD_Signal"] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(close=stock_data_recent["Stock_Price"])
    stock_data_recent["BB_High"] = bb.bollinger_hband()
    stock_data_recent["BB_Low"] = bb.bollinger_lband()
    adx_indicator = ta.trend.ADXIndicator(high=stock_data_recent["Stock_Price"],
                                          low=stock_data_recent["Stock_Price"],
                                          close=stock_data_recent["Stock_Price"])
    stock_data_recent["ADX"] = adx_indicator.adx()
    stock_data_recent["Volatility"] = stock_data_recent["BB_High"] - stock_data_recent["BB_Low"]
    stock_data_recent["Momentum"] = stock_data_recent["Stock_Price"].diff().fillna(0)
    stock_data_recent["Trend"] = stock_data_recent["Stock_Price"].rolling(window=5).mean() - stock_data_recent["Stock_Price"].rolling(window=20).mean()

    # Define the 9 numeric technical features
    numeric_features = ["RSI", "MACD", "MACD_Signal", "BB_High", "BB_Low", "ADX", "Volatility", "Momentum", "Trend"]
    stock_data_recent[numeric_features] = stock_data_recent[numeric_features].apply(pd.to_numeric, errors="coerce").fillna(0)

    # Get the latest row of technical indicators (9 features)
    indicators = stock_data_recent.iloc[-1][numeric_features].values  # shape (9,)
    # Prepend the sentiment value to form a 10-feature vector.
    tech_features_vector = np.hstack(([sentiment], indicators))  # shape (10,)
    tech_features_vector = tech_features_vector.reshape(1, -1)
    # Scale using the pre-fitted tech_scaler
    scaled = tech_scaler.transform(tech_features_vector)
    return scaled.astype(np.float32)

# Streamlit UI
st.title("📈 Stock Market Prediction")
st.write("ใส่ราคาตลาดล่าสุด และข่าวเพื่อพยากรณ์ว่าตลาดจะขึ้นหรือลง")

# User input for market price and news
market_price = st.number_input("💰 ใส่ราคาตลาดล่าสุด", min_value=0.0, 
                               value=float(stock_data["Stock_Price"].iloc[-1]))
latest_news = st.text_area("📰 ใส่ข่าวตลาด (ถ้ามี)", 
                           "Stock market is showing positive growth today...")

# Prediction function that returns the predicted label and probability
def predict(market_price, news):
    try:
        # Compute sentiment score from the news (using first 512 characters)
        sentiment_score = sentiment_pipeline(news[:512])[0]['score']
        
        # Convert news to TF-IDF (no additional scaling)
        news_tfidf = vectorizer.transform([news]).toarray().astype(np.float32)

        # Compute technical indicators (10 features) including the sentiment value
        indicators = compute_indicators(stock_data, market_price, sentiment_score).reshape(1, -1)

        # Verify feature counts: TF-IDF + technical indicators should match model's expected input
        expected_features = model.input_shape[1]  # e.g., TF-IDF features (e.g., 5000) + 10 = 5010
        current_features = news_tfidf.shape[1] + indicators.shape[1]
        if current_features != expected_features:
            return f"❌ Error: Expected {expected_features} features, but got {current_features}", None

        # Merge TF-IDF and technical indicators
        feature_array = np.hstack((news_tfidf, indicators)).astype(np.float32)
        st.write(f"✅ Final Feature Shape: {feature_array.shape}, dtype: {feature_array.dtype}")

        # Make prediction and return both label and probability
        prob = model.predict(feature_array)[0][0]
        label = "📈 Up" if prob > 0.5 else "📉 Down"
        return label, prob
    
    except Exception as e:
        return f"❌ Error: {e}", None

# Predict button
if st.button("🚀 Predict Market Movement"):
    result, probability = predict(market_price, latest_news)
    st.subheader(f"📊 Market Prediction: {result}")
    if probability is not None:
        st.write(f"Predicted probability of Up: {probability:.4f}")

    # Plot recent market trend using Plotly
    last_30_days = stock_data.tail(30)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=last_30_days["Date"],
        y=last_30_days["Stock_Price"],
        mode='lines',
        name="📊 Real Market Prices",
        line=dict(color="yellow", width=3, dash="solid"),
        hoverinfo="x+y"
    ))
    fig.add_trace(go.Scatter(
        x=[last_30_days["Date"].iloc[-1] + pd.Timedelta(days=5)],
        y=[market_price],
        mode='markers',
        name="📍 Current Market Price",
        marker=dict(color="red", size=12, symbol="star"),
        hoverinfo="x+y"
    ))
    fig.add_trace(go.Scatter(
        x=last_30_days["Date"],
        y=last_30_days["Stock_Price"].rolling(window=7).mean(),
        mode="lines",
        name="📈 7-Day Moving Avg",
        line=dict(color="green", width=2, dash="dot"),
        hoverinfo="x+y"
    ))
    fig.update_layout(
        title="📊 Real Market Trend vs Current Price",
        xaxis_title="📅 Date",
        yaxis_title="💰 Stock Price",
        font=dict(size=14, color="black"),
        xaxis=dict(showgrid=True, gridcolor="lightgray"),
        yaxis=dict(showgrid=True, gridcolor="lightgray")
    )
    st.plotly_chart(fig)

    # Create a prediction line chart: vary the market price and compute predicted probability
    price_range = np.linspace(market_price * 0.9, market_price * 1.1, 50)
    predicted_probs = []
    # Use the same news input and compute sentiment only once
    sentiment_score = sentiment_pipeline(latest_news[:512])[0]['score']
    news_tfidf = vectorizer.transform([latest_news]).toarray().astype(np.float32)
    for p in price_range:
        indicators = compute_indicators(stock_data, p, sentiment_score).reshape(1, -1)
        feature_array = np.hstack((news_tfidf, indicators)).astype(np.float32)
        prob = model.predict(feature_array)[0][0]
        predicted_probs.append(prob)
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=price_range, 
        y=predicted_probs, 
        mode='lines+markers', 
        name="Predicted Probability"
    ))
    fig2.add_trace(go.Scatter(
        x=[price_range[0], price_range[-1]], 
        y=[0.5, 0.5],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name="Threshold 0.5"
    ))
    fig2.update_layout(
        title="Predicted Probability vs Hypothetical Market Price",
        xaxis_title="Hypothetical Market Price",
        yaxis_title="Predicted Probability (Up)",
        font=dict(size=14),
        
    )
    st.plotly_chart(fig2)
