import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Настройка страницы
st.set_page_config(page_title="Оқушылар диагностикасы", layout="wide")

st.title("🧠 Мектеп оқушыларының оқу-психологиялық қиындықтарын ерте анықтаудың интеллектуалды жүйесі")

# Загрузка данных и обучение модели
@st.cache_resource
def get_trained_model():
    data = pd.read_csv('student_behavior_data.csv')
    le = LabelEncoder()
    data['Risk_Level_Encoded'] = le.fit_transform(data['Risk_Level'])
    
    X = data.drop(columns=['Student_ID', 'Risk_Level', 'Risk_Level_Encoded'])
    y = data['Risk_Level_Encoded']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler, le

try:
    model, scaler, le = get_trained_model()
except Exception as e:
    st.error("Қате: 'student_behavior_data.csv' файлы табылмады. Алдымен деректерді жасап алыңыз.")
    st.stop()

# Меню выбора режима работы
menu = st.sidebar.selectbox("Жұмыс режимі:", ["1. Бір оқушыны талдау", "2. Файл арқылы жаппай талдау (Массовый расчет)"])

if menu == "1. Бір оқушыны талдау":
    col1, col2 = st.columns()
    with col1:
        st.header("📊 Оқушы деректерін енгізу")
        lms_login = st.slider("LMS жүйесіне апталық кіру жиілігі:", 0, 25, 10)
        delay_hours = st.slider("Тапсырмаларды кешіктіру уақыты (сағатпен):", 0, 120, 24)
        screen_time = st.slider("Күнделікті экран уақыты (сағат):", 1.0, 12.0, 5.0, step=0.5)
        night_activity = st.slider("Түнгі белсенділік балы (1-10):", 1, 10, 4)
        math_grade = st.number_input("Математика бағасы (0-100):", 0, 100, 70)
        lang_grade = st.number_input("Тілдік пәндер бағасы (0-100):", 0, 100, 75)
        attendance = st.slider("Сабаққа қатысу көрсеткіші (%):", 40.0, 100.0, 90.0, step=1.0)

    with col2:
        st.header("🎯 Талдау нәтижесі")
        input_data = np.array([[lms_login, delay_hours, screen_time, night_activity, math_grade, lang_grade, attendance]])
        input_scaled = scaler.transform(input_data)
        
        if st.button("Интеллектуалды талдауды бастау", type="primary"):
            prediction = model.predict(input_scaled)
            predicted_class = le.inverse_transform(prediction)
            
            if predicted_class == 'Төмен':
                st.success(f"🟢 Қауіп деңгейі: {predicted_class}")
                st.info("Оқушының жағдайы тұрақты. Алдын алу шаралары жеткілікті.")
            elif predicted_class == 'Орташа':
                st.warning(f"🟡 Қауіп деңгейі: {predicted_class}")
                st.info("Ұсыныс: Сынып жетекшісі оқушымен әңгімелесіп, үлгерімінің төмендеу себебін анықтауы керек.")
            else:
                st.error(f"🔴 Қауіп деңгейі: {predicted_class}")
                st.markdown("**Шұғыл ұсыныс:** Мектеп психологының кеңесі қажет. Цифрлық тәуелділік немесе күйзеліс белгілері байқалады.")

else:
    st.header("📁 Мектеп бойынша жаппай талдау жасау")
    st.write("Мұнда сіз оқушылардың деректері бар дайын CSV файлды жүктей аласыз. ИИ әр оқушыны жеке талдап, қорытынды тізім береді.")
    
    uploaded_file = st.file_uploader("Оқушылар тізімі бар файлды таңдаңыз (.csv)", type=["csv"])
    
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        
        st.subheader("📋 Жүктелген деректердің алғашқы жолдары:")
        st.write(input_df.head())
        
        if st.button("Барлық оқушыларды есептеу", type="primary"):
            X_batch = input_df.drop(columns=['Student_ID'], errors='ignore')
            
            if 'Risk_Level' in X_batch.columns:
                X_batch = X_batch.drop(columns=['Risk_Level'])
                
            X_batch_scaled = scaler.transform(X_batch)
            batch_predictions = model.predict(X_batch_scaled)
            
            input_df['ИИ_Қорытындысы (Risk_Level)'] = le.inverse_transform(batch_predictions)
            
            # Перевод колонок на казахский язык
            ru_columns = {
                'Student_ID': 'Оқушы ID-і',
                'LMS_Login_Frequency': 'LMS-ке кіру жиілігі (аптасына)',
                'Avg_Assignment_Delay_Hours': 'Тапсырма кешігуі (сағатпен)',
                'Screen_Time_Hours_Daily': 'Экран уақыты (күнделікті, сағат)',
                'Night_Activity_Score': 'Түнгі белсенділік (1-10 балл)',
                'Math_Grade': 'Математика бағасы',
                'Language_Grade': 'Тілдік пәндер бағасы',
                'Attendance_Rate': 'Сабаққа қатысуы (%)',
                'Risk_Level': 'Шын мәніндегі қауіп (Настоящий риск)',
                'ИИ_Қорытындысы (Risk_Level)': 'ИИ Қорытындысы (Вердикт ИИ)'
            }
            display_df = input_df.rename(columns=ru_columns)
            
            st.subheader("🚀 Талдау нәтижесі (ИИ әр оқушыны есептеп шықты):")
            st.write(display_df)
            
            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Нәтижені жүктеп алу (Скачать отчет)",
                data=csv,
                file_name='мектеп_психологиялық_талдау_есеп.csv',
                mime='text/csv',
            )
