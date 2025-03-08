import streamlit as st

# ✅ ตั้งค่าหัวข้อหลัก
st.title("🤖 การพัฒนาโมเดลพยากรณ์ตลาดหุ้น")

st.write("""
หน้านี้อธิบายกระบวนการพัฒนาโมเดลพยากรณ์ตลาดหุ้น โดยใช้เทคนิค **Neural Networks**  
โดยจะครอบคลุมหัวข้อต่อไปนี้:
- **โครงสร้างโมเดล (Model Architecture)**
- **กระบวนการฝึกโมเดล (Training Process)**
- **การประเมินผล (Evaluation Metrics)**
- **ความท้าทายและข้อควรพิจารณา (Challenges & Considerations)**
- **แนวทางพัฒนาเพิ่มเติม (Future Improvements)**
""")

# ✅ ส่วนที่ 1: โครงสร้างของโมเดล
st.subheader("🧠 1. โครงสร้างของโมเดล (Model Architecture)")
st.write("""
โมเดลนี้เป็น **Neural Network แบบลึก (Deep Neural Network - DNN)** ซึ่งรับข้อมูลจาก 2 แหล่ง:
1. **ฟีเจอร์จากข่าวสารตลาดหุ้น** (5000 ฟีเจอร์ จาก TF-IDF)
2. **ฟีเจอร์จากอินดิเคเตอร์ทางเทคนิค** (9 ฟีเจอร์ เช่น RSI, MACD, Bollinger Bands ฯลฯ)

โดยข้อมูลทั้งหมดจะถูกใช้ร่วมกันเป็น **ฟีเจอร์รวม 5009 ฟีเจอร์** และโมเดลจะทำการพยากรณ์ว่า **ตลาดจะขึ้น (1) หรือ ลง (0)**
""")

st.markdown("""
### 📌 **โครงสร้างของ Neural Network**
- **ชั้นอินพุต (Input Layer: 5009 Neurons)** → รับค่าฟีเจอร์ทั้งหมด
- **ชั้นซ่อน (Hidden Layers)**
  - **ชั้นที่ 1**: 256 Neurons, ใช้ฟังก์ชัน `ReLU`, L2 Regularization
  - **ชั้นที่ 2**: 128 Neurons, `ReLU`, Batch Normalization
  - **ชั้นที่ 3**: 64 Neurons, `ReLU`, Dropout (40%)
  - **ชั้นที่ 4**: 32 Neurons, `ReLU`, Dropout (40%)
- **ชั้นเอาต์พุต (Output Layer: 1 Neuron, Sigmoid Activation)** → ใช้พยากรณ์ตลาดหุ้น (1 = ขึ้น, 0 = ลง)
""")

# ✅ ส่วนที่ 2: กระบวนการฝึกโมเดล
st.subheader("📊 2. กระบวนการฝึกโมเดล (Training Process)")
st.write("""
โมเดลนี้ถูกฝึกด้วยวิธีการ **Classification** โดยให้เรียนรู้ว่าตลาดหุ้นจะขึ้นหรือลง  
- **Loss Function**: `Binary Crossentropy`
- **Optimizer**: `Adam` + **Exponential Learning Rate Decay**
- **Batch Size**: 128
- **Epochs**: 100 (มี Early Stopping หากไม่ดีขึ้นใน 10 Epochs)
- **Validation Split**: ใช้ข้อมูล 20% สำหรับตรวจสอบความแม่นยำ
""")

st.markdown("""
### 🛠 **กลยุทธ์การฝึกโมเดล**
- **Dropout Regularization** → ป้องกัน Overfitting
- **Batch Normalization** → เพิ่มความเสถียรในการเรียนรู้
- **Early Stopping** → หยุดฝึกอัตโนมัติหากโมเดลไม่มีการพัฒนา
""")

# ✅ ส่วนที่ 3: การประเมินผล
st.subheader("📈 3. การประเมินผลโมเดล (Evaluation Metrics)")
st.write("""
หลังจากฝึกโมเดลเสร็จแล้ว จะมีการทดสอบกับข้อมูลที่โมเดลไม่เคยเห็นมาก่อน โดยใช้ตัวชี้วัดต่าง ๆ ได้แก่:
- **Accuracy** → วัดเปอร์เซ็นต์ที่โมเดลพยากรณ์ได้ถูกต้อง
- **Precision & Recall** → ตรวจสอบผลลัพธ์ของการพยากรณ์
- **Confusion Matrix** → วิเคราะห์ข้อผิดพลาดของโมเดล
""")

st.markdown("""
### 🔍 **ตัวชี้วัดหลักที่ใช้วัดประสิทธิภาพ**
| Metric | ความหมาย |
|--------|-------------|
| **Train Accuracy** | ความแม่นยำของโมเดลบนชุดข้อมูลฝึก |
| **Test Accuracy** | ความแม่นยำของโมเดลบนชุดข้อมูลทดสอบ |
| **Precision** | ตรวจสอบว่าโมเดลทำนาย "ขึ้น" แล้วถูกต้องกี่เปอร์เซ็นต์ |
| **Recall** | ตรวจสอบว่าตลาดขึ้นจริง แล้วโมเดลทำนายถูกต้องกี่เปอร์เซ็นต์ |
""")

# ✅ ส่วนที่ 4: ความท้าทายในการพัฒนาโมเดล
# ✅ ตั้งค่าหัวข้อหลัก
# ✅ แสดงหัวข้อหลัก
st.title("🤖 การพัฒนาโมเดลพยากรณ์ตลาดหุ้น")

# 🔹 ความท้าทายในการพยากรณ์ตลาดหุ้น
st.subheader("⚠️ 4. ความท้าทายและผลลัพธ์ของการฝึกโมเดล")
st.write("""
การพยากรณ์ตลาดหุ้นเป็นงานที่มีความซับซ้อนสูง เนื่องจากตลาดมีปัจจัยหลายอย่างที่ส่งผลกระทบ  
ทำให้การออกแบบโมเดลต้องคำนึงถึง **ความท้าทาย** หลายด้าน ได้แก่:
""")

st.markdown("""
- **📉 ความผันผวนของตลาดหุ้น:** ตลาดหุ้นเปลี่ยนแปลงขึ้นลงแบบคาดเดาได้ยาก อาจเกิดจากข่าวสารหรือปัจจัยอื่น ๆ  
- **⚠️ ปัญหา Overfitting:** โมเดลอาจจดจำข้อมูลในอดีตมากเกินไป และไม่สามารถปรับตัวกับข้อมูลใหม่ได้ดี  
- **📊 ความไม่สมดุลของข้อมูล (Imbalanced Data):** โมเดลอาจมี Bias หากจำนวนวันตลาดขึ้น/ลงไม่เท่ากัน  
""")

# ✅ แสดงผลลัพธ์ของการฝึกโมเดล
st.subheader("📊 ผลการฝึกโมเดล (Training Results)")

st.write("**กราฟแสดงผลการเรียนรู้ของโมเดล** โดยเปรียบเทียบค่า Loss และ Accuracy ระหว่าง Training และ Validation")

col1, col2 = st.columns(2)

# 📌 กราฟ Training vs Validation Loss
with col1:
    st.image("./assets/s1.png", caption="📉 กราฟ Loss ระหว่าง Training และ Validation", use_container_width=True)

# 📌 กราฟ Training vs Validation Accuracy
with col2:
    st.image("./assets/s2.png", caption="📈 กราฟ Accuracy ระหว่าง Training และ Validation", use_container_width=True)

st.write("""
🔍 **การวิเคราะห์กราฟ:**
- กราฟ **Loss** ควรลดลงอย่างต่อเนื่อง และไม่ควรแตกต่างจาก Validation มากเกินไป  
- กราฟ **Accuracy** ควรเพิ่มขึ้นอย่างมีเสถียรภาพ โดยไม่มี Overfitting  
""")

st.warning("⚠️ หากมีการ Overfitting อาจต้องใช้ **Dropout**, **L2 Regularization**, หรือ **เพิ่มข้อมูลฝึกโมเดล**")

# ✅ แสดงผลลัพธ์ Validation Loss และ Validation Accuracy
st.subheader("📊 การวิเคราะห์ Validation Loss และ Validation Accuracy")

col3, col4 = st.columns(2)

with col3:
    st.image("./assets/s3.png", caption="📊 Validation Loss", use_container_width=True)

with col4:
    st.image("./assets/s4.png", caption="📈 Validation Accuracy", use_container_width=True)

st.write("""
✅ **หากค่าความแม่นยำ (Accuracy) ของ Training และ Validation ใกล้เคียงกัน** แสดงว่าโมเดลมี **Generalization ที่ดี**  
⚠️ **แต่หาก Training Accuracy สูงกว่ามาก** แสดงว่าโมเดลอาจ **Overfit** ข้อมูล
""")

st.info("📌 **สามารถปรับค่า Hyperparameters หรือใช้ Transfer Learning เพื่อปรับปรุงโมเดลเพิ่มเติม**")

# ✅ แสดงค่าความแม่นยำของโมเดล
st.subheader("📈 ค่าความแม่นยำของโมเดล")

st.write("""
📌 **ค่าความแม่นยำจาก Terminal Logs**
- **Train Accuracy: 99.94%**
- **Validation Accuracy: 99.87 - 100%**
""")

st.write("""
📊 **วิเคราะห์ Accuracy**
- **โมเดลมีค่าความแม่นยำสูงมาก** (เกือบ 100%) แสดงว่าเรียนรู้ได้ดี
- **Training Accuracy และ Validation Accuracy ใกล้เคียงกัน** → หมายความว่าโมเดลไม่ Overfit มากนัก  
- แต่อาจต้อง **ทดสอบกับข้อมูลใหม่ (Unseen Data)** เพื่อดูว่าโมเดลไม่ได้จำข้อมูลเดิมมากเกินไป
""")

st.image("./assets/s5.png", caption="📊 กราฟความแม่นยำ Training และ Validation", use_container_width=True)

st.write("""
✅ **สรุป**
- **กราฟ Loss สวยงาม** → Training & Validation Loss ลดลงพร้อมกัน ไม่มี Overfitting  
- **Accuracy สูงมาก** → 99.87% - 100% (อาจ Overfit เล็กน้อย)  
- **ควรลองทดสอบกับข้อมูลใหม่** เพื่อดูว่าโมเดลสามารถใช้งานได้จริงหรือไม่  
""")

# ✅ แนวทางพัฒนาเพิ่มเติม
st.subheader("🚀 5. แนวทางพัฒนาเพิ่มเติม (Future Improvements)")
st.write("""
โมเดลสามารถพัฒนาเพิ่มเติมได้โดย:
""")


st.markdown("""
- **📌 ใช้ LSTM (Long Short-Term Memory)** เพื่อจับแนวโน้มระยะยาวจากข้อมูลข่าว  
- **📈 พัฒนาโมเดล Hybrid (ML + NN)** รวมข้อดีของ Machine Learning และ Neural Network  
- **🌏 เพิ่มข้อมูลปัจจัยภายนอก** เช่น ข่าวเศรษฐกิจ อัตราดอกเบี้ย ค่าเงิน  
- **🔧 Fine-tune ค่า Hyperparameters** เช่น Learning Rate, Dropout Rate  
""")

st.write("📌 **โมเดลนี้ยังอยู่ในช่วงพัฒนา และสามารถปรับปรุงให้ดีขึ้นได้ต่อไป! 🚀**")
