import streamlit as st
import pandas as pd

# ✅ ตั้งค่าหัวข้อของแอป
st.title("📊 การเตรียมข้อมูลและทฤษฎีของอัลกอริธึมสำหรับพยากรณ์ตลาดหุ้น")

st.write("""
แอปนี้แสดงรายละเอียดเกี่ยวกับการพัฒนาโมเดลพยากรณ์ตลาดหุ้นโดยใช้ **ข่าวสารรายวัน (Daily News for Stock Market Prediction)** และ **ข้อมูลทางเทคนิค (Technical Indicators)**  
กระบวนการนี้ประกอบไปด้วย 3 ขั้นตอนหลัก:
1. **การเตรียมข้อมูล (Data Preparation)**
2. **การทำความสะอาดข้อมูล (Data Cleaning)**
3. **การสร้างฟีเจอร์เพื่อใช้ในโมเดล (Feature Engineering)**
""")

# ✅ ส่วนที่ 1: การเตรียมข้อมูล
st.subheader("📂 1. การเตรียมข้อมูล (Data Preparation)")
st.write("""
โมเดลของเรานำเข้าข้อมูลจาก **2 แหล่งหลัก**:
1. **ข่าวสารตลาดหุ้นรายวัน** 📰  
   - มาจากชุดข้อมูล **"Daily News for Stock Market Prediction"** (Kaggle)
   - มีข่าวมากถึง **25 หัวข้อข่าวต่อวัน** ตั้งแต่ปี **2008 - 2016**
   - ใช้เทคนิค **TF-IDF (Term Frequency-Inverse Document Frequency)** เพื่อแปลงข้อความเป็นข้อมูลตัวเลข  

2. **ข้อมูลราคาหุ้นและอินดิเคเตอร์ทางเทคนิค** 📈  
   - ราคาหุ้นมาจาก **ดัชนี Dow Jones Industrial Average (DJIA)**
   - ใช้คำนวณ **อินดิเคเตอร์ทางเทคนิค** เช่น **RSI, MACD, Bollinger Bands, ADX, Volatility, Momentum, Trend**
""")

st.markdown("📌 **ที่มา:** [Daily News for Stock Market Prediction - Kaggle](https://www.kaggle.com/)")

# ✅ ส่วนที่ 2: การทำความสะอาดข้อมูล
st.subheader("🧼 2. การทำความสะอาดข้อมูล (Data Cleaning)")
st.write("""
ก่อนที่ข้อมูลจะถูกนำไปใช้พัฒนาโมเดล เราต้องทำการ **ทำความสะอาดข้อมูล** (Data Cleaning) เพื่อให้ได้ข้อมูลที่แม่นยำ  
""")

st.markdown("""
✅ **การทำความสะอาดข่าวสาร (News Cleaning)**  
- รวม **25 หัวข้อข่าวต่อวัน** เป็น **ข่าวเดียว**  
- แปลงข้อความเป็น **ตัวพิมพ์เล็กทั้งหมด**  
- ลบอักขระพิเศษ เช่น `! @ # $ % ^ & * ( )`  
- ใช้ **TF-IDF Vectorization** แปลงข้อความเป็นเวกเตอร์  

✅ **การทำความสะอาดข้อมูลราคาหุ้นและอินดิเคเตอร์**  
- แปลงคอลัมน์ **Date** ให้อยู่ในรูปแบบ `datetime`  
- ลบวันที่ไม่มีข้อมูลหุ้น **(Missing Dates Handling)**  
- ใช้ **Forward Fill Method (ffill)** เพื่อเติมค่าที่ขาดหาย  
- แปลงข้อมูลอินดิเคเตอร์ให้เป็นมาตรฐานโดยใช้ **StandardScaler**
""")

# ✅ ส่วนที่ 3: การสร้างฟีเจอร์ (Feature Engineering)
st.subheader("🔍 3. การสร้างฟีเจอร์เพื่อใช้ในโมเดล (Feature Engineering)")
st.write("""
หลังจากทำความสะอาดข้อมูลแล้ว จะต้องสร้างฟีเจอร์ที่โมเดลสามารถเรียนรู้ได้  
""")

st.markdown("""
### 📰 **ข่าวสารตลาดหุ้น → แปลงเป็นข้อมูลตัวเลข**
- ใช้ **TfidfVectorizer** แปลงข้อความข่าวเป็นเวกเตอร์
- ใช้ **N-Gram (1-3 คำติดกัน)** เพื่อให้โมเดลเข้าใจบริบทของข่าว
- กำหนดฟีเจอร์สูงสุดที่ **5,000 คำหลัก**
- ได้ข้อมูลเป็นเวกเตอร์ขนาด **(n_samples, 5000 features)**  

### 📈 **ข้อมูลราคาหุ้น → อินดิเคเตอร์ทางเทคนิค (Technical Indicators)**
- RSI (Relative Strength Index) → ตรวจจับภาวะ Overbought หรือ Oversold
- MACD (Moving Average Convergence Divergence) → คำนวณแนวโน้มของราคา
- Bollinger Bands → ดูช่วงราคาที่เหมาะสม
- ADX (Average Directional Index) → วัดความแข็งแกร่งของแนวโน้ม
- Volatility → วัดความผันผวนของราคา
- Momentum → คำนวณอัตราการเปลี่ยนแปลงของราคา
- Trend → ใช้ค่าเฉลี่ยเคลื่อนที่ (Moving Average)
- **ใช้ StandardScaler** ปรับขนาดอินดิเคเตอร์ให้เหมาะสม
""")

# ✅ ส่วนสุดท้าย: สรุปแนวทางการพัฒนา
st.subheader("✅ สรุปแนวทางการพัฒนาโมเดล")
st.write("""
1️⃣ **นำเข้าข้อมูลข่าวและราคาหุ้น**  
2️⃣ **ทำความสะอาดข้อมูล**  
3️⃣ **รวมข้อมูลข่าวและราคาหุ้นเข้าด้วยกัน**  
4️⃣ **ใช้ StandardScaler เพื่อทำให้ค่าของฟีเจอร์เป็นมาตรฐาน**  
5️⃣ **เตรียมข้อมูลสำหรับการพัฒนาโมเดล Neural Network**  
""")

# ✅ โหลดข้อมูล
@st.cache_data
def load_data():
    news_df = pd.read_csv("./Data/Combined_News_DJIA.csv", parse_dates=["Date"])
    stock_df = pd.read_csv("./Data/stock_prices.csv", parse_dates=["Date"])

    # แปลงราคาหุ้นให้เป็นตัวเลข (float)
    stock_df["Stock_Price"] = pd.to_numeric(stock_df["Stock_Price"], errors="coerce")
    stock_df.dropna(subset=["Stock_Price"], inplace=True)  # ลบค่าที่เป็น NaN

    # รวมพาดหัวข่าว 25 หัวข้อเป็นคอลัมน์เดียว
    news_df["Combined_News"] = news_df[[f"Top{i}" for i in range(1, 26)]].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
    
    return news_df, stock_df

news_df, stock_df = load_data()

# ✅ ส่วนหัวของแอป
st.title("📊 การแสดงข้อมูล Data Set สำหรับพยากรณ์ตลาดหุ้น")

st.write("""
แอปนี้แสดงตัวอย่างข้อมูลที่ใช้ในการพัฒนาโมเดลพยากรณ์ตลาดหุ้น ซึ่งประกอบไปด้วย:
- **ข้อมูลข่าวรายวัน** 📰 จากชุดข้อมูล "Daily News for Stock Market Prediction"
- **ข้อมูลราคาหุ้น** 📈 จากดัชนี Dow Jones Industrial Average (DJIA)
""")

# ✅ ตัวเลือกให้ผู้ใช้เลือกดูข้อมูลข่าวหรือข้อมูลราคาหุ้น
dataset_option = st.radio(
    "🔍 เลือกชุดข้อมูลที่ต้องการดู",
    ["ข่าวตลาดหุ้น (News Dataset)", "ราคาหุ้น (Stock Price Dataset)"]
)

if dataset_option == "ข่าวตลาดหุ้น (News Dataset)":
    st.subheader("📰 ตัวอย่างข้อมูลข่าวตลาดหุ้น")
    st.write(news_df[["Date", "Combined_News"]].head(10))

    # แสดงสถิติข่าว
    st.subheader("📊 สถิติของข้อมูลข่าว")
    st.write(f"- มีข้อมูลข่าวทั้งหมด: **{len(news_df)}** วัน")
    st.write(f"- ข้อมูลข่าวเริ่มต้นจาก: **{news_df['Date'].min().date()}** ถึง **{news_df['Date'].max().date()}**")

elif dataset_option == "ราคาหุ้น (Stock Price Dataset)":
    st.subheader("📈 ตัวอย่างข้อมูลราคาหุ้น DJIA")
    st.write(stock_df.head(10))

    # แสดงสถิติราคาหุ้น
    st.subheader("📊 สถิติของข้อมูลราคาหุ้น")
    st.write(f"- มีข้อมูลราคาหุ้นทั้งหมด: **{len(stock_df)}** วัน")
    st.write(f"- ข้อมูลราคาหุ้นเริ่มต้นจาก: **{stock_df['Date'].min().date()}** ถึง **{stock_df['Date'].max().date()}**")
    
    # ✅ ใช้ float ในการแสดงผลราคา
    st.write(f"- ราคาหุ้นต่ำสุด: **${float(stock_df['Stock_Price'].min()):.2f}**")
    st.write(f"- ราคาหุ้นสูงสุด: **${float(stock_df['Stock_Price'].max()):.2f}**")
    st.write(f"- ค่าเฉลี่ยของราคาหุ้น: **${float(stock_df['Stock_Price'].mean()):.2f}**")

# ✅ ส่วนท้าย - อ้างอิงแหล่งข้อมูล
st.subheader("📌 แหล่งที่มาของข้อมูล")

st.markdown("""
- ข้อมูลข่าวจากชุดข้อมูล **"Daily News for Stock Market Prediction"**  
  📌 [ที่มา: Kaggle](https://www.kaggle.com/)  
- ข้อมูลราคาหุ้นจากดัชนี **Dow Jones Industrial Average (DJIA)**  
""")

st.markdown("""
- ข้อมูลข่าวจากชุดข้อมูล **"Uniqlo (FastRetailing) Stock Price Prediction"**  
  📌 [ที่มา: Kaggle](https://www.kaggle.com/)  
""")

st.write("🚀 ข้อมูลนี้ใช้ในการพัฒนาโมเดลพยากรณ์ตลาดหุ้นโดยใช้ Neural Network")

