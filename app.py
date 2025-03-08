import streamlit as st
from streamlit_option_menu import option_menu

# ✅ ตั้งค่าเพจหลัก
st.set_page_config(page_title="แอปพยากรณ์ราคาหุ้น", page_icon="📈", layout="wide")

st.title("📊 แอปพยากรณ์ราคาหุ้น")
st.header("ณัฐภัทร พึ่งภักดี 6404062630333")

# ✅ แรงบันดาลใจในการพัฒนา
st.subheader("✨ แรงบันดาลใจในการพัฒนาแอปนี้")
st.write("""
ปัจจุบันตลาดหุ้นมีความผันผวนสูงและได้รับผลกระทบจากปัจจัยหลายด้าน ไม่ว่าจะเป็น **ข่าวสารเศรษฐกิจ, นโยบายภาครัฐ, แนวโน้มการลงทุนของนักลงทุนรายใหญ่** และ **อารมณ์ตลาด** การใช้ **Machine Learning และ Neural Networks (NNET)** ในการวิเคราะห์ข้อมูลสามารถช่วยให้นักลงทุนมองเห็นแนวโน้มของตลาดได้แม่นยำขึ้น ลดความเสี่ยงในการตัดสินใจ และใช้ข้อมูลเชิงลึกในการลงทุนได้อย่างมีประสิทธิภาพ

โครงการนี้ได้รับแรงบันดาลใจจากความต้องการ **เชื่อมโยงข้อมูลข่าวสารและข้อมูลทางการเงินเข้าด้วยกัน** เพื่อสร้าง **โมเดลที่สามารถพยากรณ์แนวโน้มตลาดหุ้นได้อัตโนมัติ** ซึ่งช่วยให้นักลงทุนและนักวิเคราะห์สามารถตัดสินใจได้อย่างแม่นยำมากขึ้น โดยใช้เทคนิค **Neural Networks (NNET)** ซึ่งเป็นอัลกอริทึมที่สามารถเรียนรู้จากข้อมูลขนาดใหญ่และซับซ้อนได้อย่างมีประสิทธิภาพ
""")

st.write("""
📌 **Datasetที่ใช้:**
- **📊 Daily News for Stock Market Prediction** (ใช้ **Neural Networks (NNET)** เพื่อวิเคราะห์ข่าวและพยากรณ์ราคาหุ้น)
- **📈 Uniqlo (FastRetailing) Stock Price Prediction** (ใช้ **Machine Learning** เพื่อทำนายราคาหุ้นของ Uniqlo)

📌 **คำแนะนำในการใช้งาน:**
- 📂 **การเตรียมข้อมูล & ทฤษฎีอัลกอริทึม**
- 🏗️ **การพัฒนาโมเดล (ML & Neural Network)**
- 🧑‍💻 **สาธิต: โมเดล Machine Learning**
- 🤖 **สาธิต: โมเดล Neural Network (NNET)**
""")

# ✅ ส่วนท้าย - อ้างอิงแหล่งข้อมูล
st.subheader("📌 แหล่งที่มาของข้อมูล")

st.markdown("""
- 🔗 [บทความเกี่ยวกับ Machine Learning](https://www.fusionsol.com/blog/machine-learning-algorithms/)  
- 📊 ข้อมูลข่าวจากชุดข้อมูล **"Daily News for Stock Market Prediction"**  
  📌 [ที่มา: Kaggle](https://www.kaggle.com/)  
- 📈 ข้อมูลราคาหุ้นจากดัชนี **Dow Jones Industrial Average (DJIA)**  
- 📉 ข้อมูลจากชุดข้อมูล **"Uniqlo (FastRetailing) Stock Price Prediction"**  
  📌 [ที่มา: Kaggle](https://www.kaggle.com/)  
- 📑 [Top 5 Stock Market Datasets for Machine Learning](https://www.kaggle.com/discussions/getting-started/167685)  
""")

st.subheader("📺 แหล่งเรียนรู้เพิ่มเติม (วิดีโอ YouTube)")

col1, col2, col3 = st.columns(3)

with col1:
    st.video("https://www.youtube.com/watch?v=Gv9_4yMHFhI")
with col2:
    st.video("https://www.youtube.com/watch?v=J4Wdy0Wc_xQ")
with col3:
    st.video("https://www.youtube.com/watch?v=OtD8wVaFm6E")