import pandas as pd
import numpy as np
import ta  
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import tensorflow as tf
from tensorflow.keras.regularizers import l1_l2 # type: ignore

# Set random seeds for reproducibility
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
        return sentiment_pipeline(text[:512])[0]['score']  # Limit to 512 characters to avoid long processing
    except Exception as e:
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

# ✅ Feature Scaling
scaler = StandardScaler()
X_news_scaled = scaler.fit_transform(X_news)

# ✅ Merge TF-IDF and Technical Indicators
features = ['Sentiment', 'RSI', 'MACD', 'MACD_Signal', 'BB_High', 'BB_Low', 'ADX', 'Volatility', 'Momentum', 'Trend']

X_final = np.hstack((X_news_scaled, df[features].values))
X_final = np.nan_to_num(X_final)

# ✅ Align features with the target
X_final = X_final[1:]
y = df['Market_Movement'][1:]

# ✅ Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# ✅ Define Neural Network Model
# model = Sequential([
#     # Dense(256, activation='relu', kernel_regularizer=l2(0.01), input_shape=(X_train.shape[1],)),
#     # Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=0.005, l2=0.01)),
#     Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.005, l2=0.01)),
#     BatchNormalization(),
#     # Dropout(0.4),  # Increased dropout
#     # Dropout(0.5),  # Increased dropout
#     Dropout(0.6),  # Increased dropout

#     # Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
#     Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
#     BatchNormalization(),
#     # Dropout(0.4),  # Increased dropout
#     # Dropout(0.5),  # Increased dropout
#     Dropout(0.6),  # Increased dropout

#     # Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
#     Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
#     # Dropout(0.4),  # Increased dropout
#     # Dropout(0.5),  # Increased dropout
#     Dropout(0.6),  # Increased dropout

#     Dense(1, activation='sigmoid')
# ])
model = Sequential([
    Dense(256, activation='relu', kernel_regularizer=l2(0.005), input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),  # Reduced Dropout

    Dense(128, activation='relu', kernel_regularizer=l2(0.005)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu', kernel_regularizer=l2(0.005)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu', kernel_regularizer=l2(0.005)),
    Dropout(0.3),

    Dense(1, activation='sigmoid')
])

# ✅ Compile model with Adam optimizer and Binary Crossentropy loss
# model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=Adam(learning_rate=0.0003), loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer=Adam(learning_rate=0.00005), loss='binary_crossentropy', metrics=['accuracy'])

# ✅ Set early stopping to avoid overfitting
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

from tensorflow.keras.callbacks import ReduceLROnPlateau

# lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, verbose=1)
lr_schedule = ExponentialDecay(initial_learning_rate=0.0001, decay_steps=1000, decay_rate=0.9)
optimizer = Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])



import matplotlib.pyplot as plt

# ✅ Train the model with EarlyStopping
# history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])
# history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])
history = model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_test, y_test), callbacks=[early_stopping])

# ✅ Evaluate Model
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"Neural Network Train Accuracy: {train_acc:.4f}")
print(f"Neural Network Test Accuracy: {test_acc:.4f}")

# ✅ Test with a random sample
sample_index = np.random.randint(0, len(X_test))
sample_data = X_test[sample_index].reshape(1, -1)
nn_prediction = (model.predict(sample_data) > 0.5).astype(int).flatten()[0]
actual_value = y_test.iloc[sample_index]

print(f"Neural Network Prediction: {'Up' if nn_prediction == 1 else 'Down'}")
print(f"Actual Market Movement: {'Up' if actual_value == 1 else 'Down'}")

# ✅ Check for overfitting by plotting loss and accuracy

# 1️⃣ Plot Loss (Training vs Validation)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()

# 2️⃣ Plot Accuracy (Training vs Validation)
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()

plt.show()



