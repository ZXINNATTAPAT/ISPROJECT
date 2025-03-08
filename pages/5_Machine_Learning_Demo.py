import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb

# 📌 Page Title
st.title("🔮 Uniqlo (Fast Retailing) Stock Price Prediction")
st.write("This app trains a model using **2012-2016 stock data** and predicts 2017 stock prices.")

# 📌 Load Training Data (2012-2016)
@st.cache_data  # Caches the dataset to avoid reloading
def load_data():
    train_df = pd.read_csv("./Data/Uniqlo(FastRetailing) 2012-2016 Training - stocks2012-2016.csv")
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    train_df.set_index('Date', inplace=True)
    return train_df

train_df = load_data()

# 📌 Select Features & Target
features = ['Open', 'High', 'Low', 'Volume']
target = 'Close'

X_train = train_df[features]
y_train = train_df[target]

# 📌 Normalize Data (Scaling)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 📌 Train Models
@st.cache_resource  # Cache the models to avoid retraining on every input change
def train_models():
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
    
    return trained_models

models = train_models()

# 📌 User Input Fields
st.sidebar.header("Enter Stock Data for Prediction:")
open_price = st.sidebar.number_input("Open Price (¥)", min_value=10000, max_value=100000, value=62000)
high_price = st.sidebar.number_input("High Price (¥)", min_value=10000, max_value=100000, value=62500)
low_price = st.sidebar.number_input("Low Price (¥)", min_value=10000, max_value=100000, value=61500)
volume = st.sidebar.number_input("Volume", min_value=100000, max_value=10000000, value=1000000)

# 📌 Prediction
if st.sidebar.button("Predict Stock Price"):
    # Convert input to DataFrame
    input_data = pd.DataFrame([[open_price, high_price, low_price, volume]], columns=features)

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict with each model
    predictions = {name: model.predict(input_scaled)[0] for name, model in models.items()}

    # 📌 Display Results
    st.subheader("📈 Predicted Closing Prices:")
    for name, prediction in predictions.items():
        st.success(f"{name}: ¥{round(prediction, 2)}")

    # 📌 Plot Predictions
    st.subheader("📊 Model Predictions Comparison")
    fig, ax = plt.subplots()
    ax.bar(predictions.keys(), predictions.values(), color=['blue', 'green', 'red', 'purple'])
    ax.set_ylabel("Predicted Price (¥)")
    ax.set_title("Comparison of Model Predictions")
    st.pyplot(fig)

# 📌 Show Sample of Training Data
st.subheader("📊 Sample of Training Data (2012-2016)")
st.dataframe(train_df.head())
