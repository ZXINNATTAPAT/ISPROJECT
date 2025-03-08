import pandas as pd
import numpy as np
import ta  
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

# ✅ ไลบรารีสำหรับ Technical Indicators
# ✅ โหลดข้อมูลข่าวและราคาหุ้น
news_df = pd.read_csv("./Data/Combined_News_DJIA.csv")
stock_df = pd.read_csv("./Data/stock_prices.csv")

# ✅ วิเคราะห์ Sentiment ด้วย FinBERT
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1)  # ใช้ CPU

# รวมข่าว Top1 - Top25 เป็น 'News_Headline'
news_df['News_Headline'] = news_df[[f"Top{i}" for i in range(1, 26)]].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

# ใช้ try-except ป้องกัน Error
def get_sentiment(text):
    try:
        return sentiment_pipeline(text)[0]['score']
    except:
        return 0

news_df['Sentiment'] = news_df['News_Headline'].apply(get_sentiment)

# ✅ แปลง 'Date' เป็น datetime และรวมข้อมูล
news_df["Date"] = pd.to_datetime(news_df["Date"])
stock_df["Date"] = pd.to_datetime(stock_df["Date"])
df = pd.merge(news_df, stock_df, on="Date")

# ✅ แปลง Stock_Price เป็น float และสร้าง Target (Market Movement)
df['Stock_Price'] = df['Stock_Price'].astype(float)
df['Market_Movement'] = (df['Stock_Price'].diff() > 0).astype(int)

# ✅ คำนวณ Indicators (RSI, MACD, Bollinger Bands, ADX)
df['RSI'] = ta.momentum.RSIIndicator(close=df['Stock_Price'], window=14).rsi()

df['MACD'] = ta.trend.MACD(close=df['Stock_Price']).macd()

df['MACD_Signal'] = ta.trend.MACD(close=df['Stock_Price']).macd_signal()

df['BB_High'] = ta.volatility.BollingerBands(close=df['Stock_Price']).bollinger_hband()

df['BB_Low'] = ta.volatility.BollingerBands(close=df['Stock_Price']).bollinger_lband()

df['ADX'] = ta.trend.ADXIndicator(high=df['Stock_Price'], low=df['Stock_Price'], close=df['Stock_Price']).adx()

# ✅ สร้าง Features ใหม่
df['Volatility'] = df['BB_High'] - df['BB_Low']

df['Momentum'] = df['Stock_Price'].diff()

df['Trend'] = df['Stock_Price'].rolling(window=5).mean() - df['Stock_Price'].rolling(window=20).mean()

# ✅ เติมค่า NaN
df.fillna(method='ffill', inplace=True)

df.fillna(0, inplace=True)

print(df.duplicated().sum())  # ดูว่ามีข้อมูลซ้ำกี่แถว

df.drop_duplicates(inplace=True) # ถ้ามีข้อมูลซ้ำมาก → ลบข้อมูลซ้ำออก

# ✅ ทำ Text Cleaning และแปลงเป็นเวกเตอร์
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))

X_news = vectorizer.fit_transform(df['News_Headline']).toarray()

# ✅ ทำ Feature Scaling
scaler = StandardScaler()

X_news_scaled = scaler.fit_transform(X_news)

# ✅ รวม TF-IDF และ Technical Indicators
X_final = np.hstack((X_news_scaled, df[['Sentiment', 'RSI', 'MACD', 'MACD_Signal', 'BB_High', 'BB_Low', 'ADX', 'Volatility', 'Momentum', 'Trend']].values))

X_final = np.nan_to_num(X_final)

# ✅ ตัดแถวแรกของ X_final ออกเพื่อให้ขนาดตรงกับ Market_Movement
X_final = X_final[1:]

# ✅ แบ่งข้อมูล Train/Test
X_train, X_test, y_train, y_test = train_test_split(X_final, df['Market_Movement'][1:], test_size=0.2, random_state=42)

# ✅ 3. Neural Network (Optimized)
model = Sequential([
    # Dense(256, activation='relu', kernel_regularizer=l2(0.001), input_shape=(X_train.shape[1],)),
    Dense(256, activation='relu', kernel_regularizer=l2(0.005), input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),

    # Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(128, activation='relu', kernel_regularizer=l2(0.005)),
    BatchNormalization(),
    Dropout(0.3),

    # Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(64, activation='relu', kernel_regularizer=l2(0.005)),
    Dropout(0.3),

    Dense(1, activation='sigmoid')
])

optimizer = Adam(learning_rate=0.005)  # ปรับ Learning Rate ให้สูงขึ้น
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


# history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])



loss, accuracy = model.evaluate(X_test, y_test)

print("Neural Network (MLP) Accuracy:", accuracy)

# xgb_preds = xgb_model.predict(X_test)
# nn_preds = (model.predict(X_test) > 0.5).astype(int).flatten()

# # ใช้ Voting ระหว่าง XGBoost และ Neural Network
# final_preds = (xgb_preds + nn_preds) // 2  # ถ้าเห็นตรงกันให้ใช้ค่านั้น

print("Ensemble Model Accuracy:", accuracy_score(y_test, final_preds))


train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"Neural Network Train Accuracy: {train_acc:.4f}")
print(f"Neural Network Test Accuracy: {test_acc:.4f}")


sample_index = np.random.randint(0, len(X_test))  # เลือกข้อมูลตัวอย่างแบบสุ่ม
sample_data = X_test[sample_index].reshape(1, -1)  # นำข้อมูลไปทดสอบ

nn_prediction = (model.predict(sample_data) > 0.5).astype(int).flatten()[0]
print(f"Neural Network Prediction: {'Up' if nn_prediction == 1 else 'Down'}")

actual_value = y_test.iloc[sample_index]
print(f"Actual Market Movement: {'Up' if actual_value == 1 else 'Down'}")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Train vs Validation Loss')
plt.show()


