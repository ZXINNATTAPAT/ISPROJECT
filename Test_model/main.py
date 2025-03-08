import pandas as pd
import numpy as np
import ta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import xgboost as xgb
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

# ✅ โหลดข้อมูล
news_df = pd.read_csv("./Data/Combined_News_DJIA.csv")
stock_df = pd.read_csv("./Data/stock_prices.csv")

# ✅ รวมข้อมูล
df = pd.merge(news_df, stock_df, on="Date")
df['Stock_Price'] = df['Stock_Price'].astype(float)
df['Market_Movement'] = (df['Stock_Price'].diff() > 0).astype(int)

# ✅ Feature Engineering
df['RSI'] = ta.momentum.RSIIndicator(close=df['Stock_Price'], window=14).rsi()
df['MACD'] = ta.trend.MACD(close=df['Stock_Price']).macd()
df['Daily_Return'] = df['Stock_Price'].pct_change()
df.fillna(method='ffill', inplace=True)

# ✅ Scaling Data
scaler = StandardScaler()
X_final = scaler.fit_transform(df[['RSI', 'MACD', 'Daily_Return']])
X_final = X_final[1:]

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_final, df['Market_Movement'][1:], test_size=0.2, random_state=42)

# ✅ SMOTE เพื่อเพิ่มข้อมูล
X_train_balanced = np.nan_to_num(X_train)  # แก้ปัญหา NaN
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# ✅ XGBoost Model
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.02,
    max_depth=5,
    tree_method='hist',
    subsample=0.8,  # ลด Overfitting
    colsample_bytree=0.8
)
xgb_model.fit(X_train_balanced, y_train_balanced)
xgb_preds = xgb_model.predict(X_test)
print("XGBoost Accuracy (CPU):", accuracy_score(y_test, xgb_preds))

# ✅ Neural Network Model (แก้ปัญหา NaN)
model = Sequential([
    Dense(128, kernel_regularizer=l2(0.0005), input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    LeakyReLU(),
    Dropout(0.3),

    Dense(64, kernel_regularizer=l2(0.0005)),
    BatchNormalization(),
    LeakyReLU(),
    Dropout(0.3),

    Dense(32, kernel_regularizer=l2(0.0005)),
    Dropout(0.3),

    Dense(1, activation='sigmoid')
])

optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# ✅ EarlyStopping & ReduceLROnPlateau
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

history = model.fit(X_train_balanced, y_train_balanced, epochs=30, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stop, reduce_lr])

# ✅ แสดงกราฟ Loss และ Accuracy
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Train vs Validation Loss')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Train vs Validation Accuracy')
plt.show()
