import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 📌 ชื่อหน้า
st.title("📂 การเตรียมข้อมูลสำหรับการพยากรณ์ราคาหุ้น Uniqlo")
st.write("หน้านี้เน้นการโหลดข้อมูล ทำความสะอาด และการเตรียมข้อมูลก่อนนำไปใช้ในโมเดลพยากรณ์ราคาหุ้น")

# 📌 สร้างข้อมูลจำลองที่มีข้อผิดพลาด (Data Corruption Simulation)
st.subheader("⚠️ ข้อมูลดิบที่มีข้อผิดพลาด")
st.write("แสดงตัวอย่างข้อมูลที่มีปัญหา เช่น ค่าที่ขาดหาย และอักขระแปลก ๆ ที่อาจเกิดขึ้นในข้อมูลจริง")

def generate_corrupted_data():
    data = {
        'Date': pd.date_range(start='1/1/2016', periods=10, freq='D'),
        'Open': [25000, 25500, np.nan, '###', 26000, 25800, 25950, np.nan, 26200, '??'],
        'High': [25200, 25700, 25900, 26300, np.nan, '!!', 26150, 26300, np.nan, 26500],
        'Low': [24800, 25300, 25750, '***', 25800, np.nan, 25650, 25850, 26000, '###'],
        'Volume': [1200000, np.nan, 1150000, 1300000, 1255000, '???', 1280000, 1260000, np.nan, 1235000],
        'Close': [25100, 25600, 25800, 26050, 25900, np.nan, 'NaN', 26100, 26350, 26400]
    }
    df_corrupted = pd.DataFrame(data)
    df_corrupted.set_index('Date', inplace=True)
    return df_corrupted

corrupted_df = generate_corrupted_data()
st.dataframe(corrupted_df)

# 📌 ทำความสะอาดข้อมูล
st.subheader("🧼 การทำความสะอาดข้อมูล")
st.write("ทำการลบค่าที่ขาดหายและค่าที่มีอักขระแปลก ๆ ออกจากข้อมูล")

def clean_data(df):
    df_cleaned = df.replace(['###', '??', '***', '!!', '???', 'NaN'], np.nan)  # แทนค่าผิดพลาดด้วย NaN
    df_cleaned.dropna(inplace=True)  # ลบแถวที่มีค่า NaN
    df_cleaned = df_cleaned.apply(pd.to_numeric, errors='coerce')  # แปลงค่าข้อมูลให้เป็นตัวเลข
    return df_cleaned

df = clean_data(corrupted_df)
st.success("✅ ข้อมูลที่ผิดพลาดได้รับการทำความสะอาดเรียบร้อยแล้ว")

# 📌 การเลือกฟีเจอร์ที่สำคัญ
st.subheader("📌 การเลือกฟีเจอร์สำคัญ")
st.write("เลือกฟีเจอร์ที่มีความสำคัญต่อการพยากรณ์ราคาหุ้น")
features = ['Open', 'High', 'Low', 'Volume']
target = 'Close'

X = df[features]
y = df[target]

# 📌 การปรับขนาดข้อมูล (Normalization)
st.subheader("📏 การปรับขนาดข้อมูล")
st.write("ปรับขนาดข้อมูลด้วย MinMaxScaler เพื่อให้ค่าข้อมูลอยู่ในช่วงที่เหมาะสม")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

st.write("✅ ปรับขนาดข้อมูลเรียบร้อยแล้ว")

# 📌 แสดงตัวอย่างข้อมูลที่ถูกประมวลผล
processed_df = pd.DataFrame(X_scaled, columns=features, index=df.index)
st.subheader("📊 ตัวอย่างข้อมูลหลังประมวลผล")
st.dataframe(processed_df.head())

st.success("✅ การเตรียมข้อมูลเสร็จสมบูรณ์!")
