import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 📌 Load Training Data (2012-2016)
train_df = pd.read_csv("./Data/Uniqlo(FastRetailing) 2012-2016 Training - stocks2012-2016.csv")
train_df['Date'] = pd.to_datetime(train_df['Date'])
train_df.set_index('Date', inplace=True)

# 📌 Load Test Data (2017)
test_df = pd.read_csv("./Data/Uniqlo(FastRetailing) 2017 Test - stocks2017.csv")
test_df['Date'] = pd.to_datetime(test_df['Date'])
test_df.set_index('Date', inplace=True)

# 📌 Select Features & Target
features = ['Open', 'High', 'Low', 'Volume']  # Predict using these
target = 'Close'  # Predicting 'Close' price

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

# 📌 Normalize Data (Scaling)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 📌 Train Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

# 📌 Train Decision Tree Model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train_scaled, y_train)
y_pred_dt = dt_model.predict(X_test_scaled)

# 📌 Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)

# 📌 Evaluate Model Performance Function
def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) * 100  # Convert to percentage
    print(f"\n📊 {name} Model Performance:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Model Accuracy (R² Score): {r2:.2f}%")

# 📌 Print Accuracy of Each Model
evaluate_model("Linear Regression", y_test, y_pred_lr)
evaluate_model("Decision Tree", y_test, y_pred_dt)
evaluate_model("Random Forest", y_test, y_pred_rf)

# 📌 Visualizing Predictions
plt.figure(figsize=(14,6))
plt.plot(y_test.values, label="Actual Prices", color='black')
plt.plot(y_pred_lr, label="Linear Regression", linestyle='dashed', color='blue')
plt.plot(y_pred_dt, label="Decision Tree", linestyle='dashed', color='green')
plt.plot(y_pred_rf, label="Random Forest", linestyle='dashed', color='red')
plt.legend()
plt.title("Uniqlo (Fast Retailing) Stock Price Prediction (2017) - Model Comparison")
plt.show()
