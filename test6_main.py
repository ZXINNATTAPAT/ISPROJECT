import pandas as pd
import numpy as np
import ta  
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load datasets
news_df = pd.read_csv("./Data/Combined_News_DJIA.csv")
stock_df = pd.read_csv("./Data/stock_prices.csv")

# Initialize Sentiment Analysis Pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1)  # CPU mode

# Combine News Headlines
news_df['News_Headline'] = news_df[[f"Top{i}" for i in range(1, 26)]].apply(
    lambda x: ' '.join(x.dropna().astype(str)), axis=1)

# Sentiment Analysis Function
def get_sentiment(text):
    try:
        return sentiment_pipeline(text[:512])[0]['score']
    except Exception:
        return 0  

news_df['Sentiment'] = news_df['News_Headline'].apply(get_sentiment)

# Convert 'Date' columns to datetime format
news_df["Date"] = pd.to_datetime(news_df["Date"])
stock_df["Date"] = pd.to_datetime(stock_df["Date"])

# Merge news and stock data on Date
df = pd.merge(news_df, stock_df, on="Date")

# Convert stock price to float and create Market Movement target
df['Stock_Price'] = df['Stock_Price'].astype(float)
df['Market_Movement'] = df['Stock_Price'].diff().fillna(0).apply(lambda x: 1 if x > 0 else 0)

# Calculate Technical Indicators
df['RSI'] = ta.momentum.RSIIndicator(close=df['Stock_Price'], window=14).rsi()
macd = ta.trend.MACD(close=df['Stock_Price'])
df['MACD'], df['MACD_Signal'] = macd.macd(), macd.macd_signal()
bb = ta.volatility.BollingerBands(close=df['Stock_Price'])
df['BB_High'], df['BB_Low'] = bb.bollinger_hband(), bb.bollinger_lband()
df['ADX'] = ta.trend.ADXIndicator(high=df['Stock_Price'], low=df['Stock_Price'], close=df['Stock_Price']).adx()
df['Volatility'] = df['BB_High'] - df['BB_Low']
df['Momentum'] = df['Stock_Price'].diff().fillna(0)
df['Trend'] = df['Stock_Price'].rolling(window=5).mean() - df['Stock_Price'].rolling(window=20).mean()

# Handle missing values and drop duplicates
df.fillna(method='ffill', inplace=True)
df.fillna(0, inplace=True)
df.drop_duplicates(inplace=True)

# Convert News Headlines to TF-IDF vectors (reduced dimensionality)
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 3))
X_news = vectorizer.fit_transform(df['News_Headline']).toarray()
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Prepare Technical Indicator Features and scale them
tech_features = ['Sentiment', 'RSI', 'MACD', 'MACD_Signal', 'BB_High', 'BB_Low', 'ADX', 'Volatility', 'Momentum', 'Trend']
X_tech = df[tech_features].values
tech_scaler = StandardScaler()
X_tech_scaled = tech_scaler.fit_transform(X_tech)
joblib.dump(tech_scaler, "tech_scaler.pkl")

# Combine TF-IDF and Technical Indicator Features
# Note: We leave X_news as-is because TfidfVectorizer by default L2-normalizes the data.
X_final = np.hstack((X_news, X_tech_scaled))
X_final = np.nan_to_num(X_final)

# Align features with the target (remove first row to account for diff())
X_final = X_final[1:]
y = df['Market_Movement'][1:].values  

# Chronological split: use the first 80% for training and the last 20% for testing
split_index = int(0.8 * len(X_final))
X_train, X_test = X_final[:split_index], X_final[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Define a simpler Neural Network Model
model = Sequential([
    Dense(128, activation='relu', kernel_regularizer=l2(0.01), input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu', kernel_regularizer=l2(0.005)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile model with a lower learning rate
model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=128,
    callbacks=[early_stop],
    verbose=1
)

# Plot training vs validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs. Validation Loss')
plt.legend()
plt.show()

# Evaluate Model
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Neural Network Train Accuracy: {train_acc:.4f}")
print(f"Neural Network Test Accuracy: {test_acc:.4f}")

# Predict on test set and classify "Up" or "Down"
predictions = model.predict(X_test)
predicted_labels = ["Up" if pred > 0.5 else "Down" for pred in predictions.flatten()]

# For demonstration, print the first 20 predictions with their true labels
print("Sample predictions (Predicted vs. Actual):")
for i in range(20):
    print(f"Prediction: {predicted_labels[i]}, Actual: {'Up' if y_test[i]==1 else 'Down'}")

# Save model
model.save("stock_prediction_model.keras")
