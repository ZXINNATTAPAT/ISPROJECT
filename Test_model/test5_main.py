import pandas as pd
import numpy as np
import ta  
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2, l1_l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib

# ✅ Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ✅ Load datasets
news_df = pd.read_csv("./Data/Combined_News_DJIA.csv")
stock_df = pd.read_csv("./Data/stock_prices.csv")

# ✅ Initialize Sentiment Analysis Pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1)  # CPU mode

# ✅ Combine News Headlines
news_df['News_Headline'] = news_df[[f"Top{i}" for i in range(1, 26)]].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

# ✅ Sentiment Analysis with Error Handling
def get_sentiment(text):
    try:
        return sentiment_pipeline(text[:512])[0]['score']  # Limit to 512 characters
    except Exception:
        return 0  # Default sentiment score

news_df['Sentiment'] = news_df['News_Headline'].apply(get_sentiment)

# ✅ Convert 'Date' column to datetime format
news_df["Date"] = pd.to_datetime(news_df["Date"])
stock_df["Date"] = pd.to_datetime(stock_df["Date"])

# ✅ Merge news and stock data on Date
df = pd.merge(news_df, stock_df, on="Date")

# ✅ Convert stock price to float and create Market Movement target
df['Stock_Price'] = df['Stock_Price'].astype(float)
df['Market_Movement'] = df['Stock_Price'].diff().fillna(0).apply(lambda x: 1 if x > 0 else 0)

# ✅ Calculate Technical Indicators (RSI, MACD, Bollinger Bands, ADX)
df['RSI'] = ta.momentum.RSIIndicator(close=df['Stock_Price'], window=14).rsi()
macd = ta.trend.MACD(close=df['Stock_Price'])
df['MACD'], df['MACD_Signal'] = macd.macd(), macd.macd_signal()
bb = ta.volatility.BollingerBands(close=df['Stock_Price'])
df['BB_High'], df['BB_Low'] = bb.bollinger_hband(), bb.bollinger_lband()
df['ADX'] = ta.trend.ADXIndicator(high=df['Stock_Price'], low=df['Stock_Price'], close=df['Stock_Price']).adx()

# ✅ Create new features
df['Volatility'] = df['BB_High'] - df['BB_Low']
df['Momentum'] = df['Stock_Price'].diff().fillna(0)
df['Trend'] = df['Stock_Price'].rolling(window=5).mean() - df['Stock_Price'].rolling(window=20).mean()

# ✅ Handle missing values
df.fillna(method='ffill', inplace=True)
df.fillna(0, inplace=True)

# ✅ Remove duplicates
df.drop_duplicates(inplace=True)

# ✅ Convert News Headlines to TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
X_news = vectorizer.fit_transform(df['News_Headline']).toarray()

# ✅ บันทึก TF-IDF Vectorizer เพื่อใช้ตอนพยากรณ์
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("✅ Vectorizer saved as `tfidf_vectorizer.pkl`")

# ✅ Feature Scaling
scaler = StandardScaler()
X_news_scaled = scaler.fit_transform(X_news)

# ✅ Merge TF-IDF and Technical Indicators
features = ['Sentiment', 'RSI', 'MACD', 'MACD_Signal', 'BB_High', 'BB_Low', 'ADX', 'Volatility', 'Momentum', 'Trend']
X_final = np.hstack((X_news_scaled, df[features].values))
X_final = np.nan_to_num(X_final)  # Convert NaNs to 0

# ✅ Align features with the target
X_final = X_final[1:]
y = df['Market_Movement'][1:].values  # Ensure it's a NumPy array

# ✅ Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# ✅ Define Neural Network Model
model = Sequential([
    # Dense(256, activation='relu', kernel_regularizer=l2(0.005), input_shape=(X_train.shape[1],)),
    Dense(256, activation='relu', kernel_regularizer=l2(0.01), input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    # Dropout(0.3),
    Dropout(0.4),

    Dense(128, activation='relu', kernel_regularizer=l2(0.005)),
    BatchNormalization(),
    # Dropout(0.3),
    Dropout(0.4),

    Dense(64, activation='relu', kernel_regularizer=l2(0.005)),
    BatchNormalization(),
    # Dropout(0.3),
    Dropout(0.4),

    Dense(32, activation='relu', kernel_regularizer=l2(0.005)),
    # Dropout(0.3),
    Dropout(0.4),

    Dense(1, activation='sigmoid')
])

# ✅ Learning rate scheduler for gradual decay
lr_schedule = ExponentialDecay(
    initial_learning_rate=0.0001,  
    # initial_learning_rate=0.0003,  
    decay_steps=1000,
    decay_rate=0.9
)

# ✅ Compile model with Adam optimizer & Binary Crossentropy loss
model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='binary_crossentropy', metrics=['accuracy'])

# ✅ Set early stopping to avoid overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,  # Wait 10 epochs before stopping
    # patience=5,  # Wait 10 epochs before stopping
    restore_best_weights=True
)

# ✅ ReduceLROnPlateau for learning rate adjustments
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    patience=2,  
    factor=0.5,  
    verbose=1
)

# ✅ Train the model with callbacks
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=128,
    callbacks=[early_stopping]  
)

# ✅ Evaluate Model
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"Neural Network Train Accuracy: {train_acc:.4f}")
print(f"Neural Network Test Accuracy: {test_acc:.4f}")

# ✅ Test with a random sample
sample_index = np.random.randint(0, len(X_test))
sample_data = X_test[sample_index].reshape(1, -1)
nn_prediction = (model.predict(sample_data) > 0.5).astype(int).flatten()[0]
actual_value = y_test[sample_index]  # Fixed indexing issue

print(f"Neural Network Prediction: {'Up' if nn_prediction == 1 else 'Down'}")
print(f"Actual Market Movement: {'Up' if actual_value == 1 else 'Down'}")

# ✅ Plot Loss & Accuracy
plt.figure(figsize=(12, 5))

# 1️⃣ Training vs Validation Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()

# 2️⃣ Training vs Validation Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()

plt.show()

model.save("stock_prediction_model.keras")