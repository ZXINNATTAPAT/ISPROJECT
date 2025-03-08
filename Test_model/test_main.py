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

# ✅ 1. ใช้ Logistic Regression
# log_model = LogisticRegression(max_iter=10000)
# log_model.fit(X_train, y_train)
# log_preds = log_model.predict(X_test)

# train_acc = log_model.score(X_train, y_train)
# test_acc = log_model.score(X_test, y_test)
# print("Logistic Regression Accuracy:", accuracy_score(y_test, log_preds))
# print(f"Logistic Regression Train Accuracy: {train_acc:.2f}, Test Accuracy: {test_acc:.2f}")

# ✅ 2. ใช้ XGBoost (Optimized)
# xgb_model = xgb.XGBClassifier(
#     n_estimators=300,
#     learning_rate=0.01,
#     max_depth=6,
#     min_child_weight=5,  # ลด Overfitting
#     subsample=0.7,  # ใช้ข้อมูลแค่ 70% ในแต่ละรอบการเรียนรู้
#     colsample_bytree=0.7  # ใช้ Features แค่ 70% ในแต่ละรอบการเรียนรู้
# )
# xgb_model.fit(X_train, y_train)
# xgb_preds = xgb_model.predict(X_test)
# train_acc_xgb = xgb_model.score(X_train, y_train)
# test_acc_xgb = xgb_model.score(X_test, y_test)
# print(f"XGBoost Train Accuracy: {train_acc_xgb:.2f}, Test Accuracy: {test_acc_xgb:.2f}")
# print("XGBoost Accuracy:", accuracy_score(y_test, xgb_preds))

# ✅ 3. Neural Network (Optimized)
model = Sequential([
    Dense(512, activation='relu', kernel_regularizer=l2(0.04), input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),  # ลด Dropout เพื่อให้โมเดลเรียนรู้ดีขึ้น

    Dense(256, activation='relu', kernel_regularizer=l2(0.04)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu', kernel_regularizer=l2(0.04)),
    Dropout(0.3),

    Dense(1, activation='sigmoid')
])

optimizer = Adam(learning_rate=0.0005)  # เพิ่ม Learning Rate
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=60, batch_size=32, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)

print("Neural Network (MLP) Accuracy:", accuracy)

# ✅ 4. Stacking Model (Final Model)
# stack_model = StackingClassifier(
#     estimators=[
#         ('xgb', xgb.XGBClassifier(n_estimators=500, learning_rate=0.005, max_depth=10)),
#         ('rf', RandomForestClassifier(n_estimators=200, max_depth=20))
#     ],
#     final_estimator=xgb.XGBClassifier(n_estimators=100, learning_rate=0.001, max_depth=5)
# )

# stack_model.fit(X_train, y_train)
# stack_preds = stack_model.predict(X_test)

# train_acc_stack = stack_model.score(X_train, y_train)
# test_acc_stack = stack_model.score(X_test, y_test)

# print(f"Stacking Model Train Accuracy: {train_acc_stack:.2f}, Test Accuracy: {test_acc_stack:.2f}")
# print("Stacking Model Accuracy:", accuracy_score(y_test, stack_preds))
