import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Настройка страницы в стиле полноценного портала BilimClass
st.set_page_config(page_title="BilimPredict - Мектеп платформасы", layout="wide")

# Инициализация баз данных в памяти (для демонстрации)
if 'real_students_db' not in st.session_state:
    st.session_state['real_students_db'] = pd.DataFrame(columns=[
        'Оқушы ID-і', 'Аты-жөні', 'Сыныбы', 
        'LMS-ке кіру жиілігі (аптасына)', 'Тапсырма кешігуі (сағатпен)', 
        'Экран уақыты (күнделікті, сағат)', 'Түнгі белсенділік (1-10 балл)', 
        'Математика бағасы', 'Тілдік пәндер бағасы', 'Сабаққа қатысуы (%)'
    ])

if 'homeworks' not in st.session_state:
    st.session_state['homeworks'] = [
        {"пән": "Алгебра", "тақырып": "Туындыны табу", "мерзімі": "Бүгін, 18:00"},
        {"пән": "Қазақ тілі", "тақырып": "Эссе жазу: Сандық әлем", "мерзімі": "Ертең, 15:00"}
    ]

# Обучение ИИ
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
except:
    st.error("Қате: Оқыту базасы табылмады.")
    st.stop()

# --- СИСТЕМА ВХОДА (ВЫБОР РОЛИ) ---
st.sidebar.image("https://flaticon.com", width=70)
st.sidebar.title("BilimPredict Порталы")

user_role = st.sidebar.radio("Жүйеге кіру (Роль):", ["👤 Оқушы кабинеті (Ученик)", "🧑‍🏫 Мұғалім/Психолог кабинеті"])

# ==================== 1. КАБИНЕТ УЧЕНИКА ====================
if user_role == "👤 Оқушы кабинеті (Ученик)":
    student_menu = st.sidebar.selectbox("Бөлімдер:", ["🪪 Менің Профилім", "📚 Сабақтар кестесі", "📝 Үй тапсырмалары (ДЗ)"])
    
    if student_menu == "🪪 Менің Профилім":
        st.title("🪪 Оқушының жеке профилі")
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image("https://flaticon.com", width=150)
        with col2:
            st.subheader("Алихан Сұлтанов")
            st.write("**Сыныбы:** 9-А сынып")
            st.write("**Мектеп:** №10 IT-Лицей")
            st.info("💡 ИИ ескертуі: Сіздің соңғы аптадағы экран уақытыңыз артқан. Ұйқы режимін сақтау ұсынылады.")
            
    elif student_menu == "📚 Сабақтар кестесі":
        st.title("📚 Бүгінгі сабақтар мен материалдар")
        st.write("Бүгінгі күнге арналған оқу материалдары мен онлайн силлабустар:")
        schedule = pd.DataFrame({
            "Уақыты": ["08:30 - 09:15", "09:25 - 10:10", "10:20 - 11:05"],
            "Пән атауы": ["Математика", "Информатика", "Қазақ тілі"],
            "Тақырып": ["Тригонометрия негіздері", "Python-да массивтер", "Сенімділік және цифрлық мәдениет"],
            "Сілтеме": ["Жүктеу (.pdf)", "Бейнебаянға сілтеме", "Оқулық беті-45"]
        })
        st.table(schedule)
        
    elif student_menu == "📝 Үй тапсырмалары (ДЗ)":
        st.title("📝 Белсенді үй тапсырмалары (ДЗ)")
        for hw in st.session_state['homeworks']:
            st.info(f"**Пән:** {hw['пән']} | **Тақырып:** {hw['тақырып']} | 📅 **Мерзімі:** {hw['мерзімі']}")
        
        st.subheader("📤 Жаңа тапсырма өткізу")
        uploaded_hw = st.file_uploader("Үй жұмысын бекіту (PDF, Имидж):", type=["pdf", "png", "jpg"])
        if uploaded_hw:
            st.success("Тапсырма сәтті жіберілді! Мұғалім тексереді.")

# ==================== 2. КАБИНЕТ УЧИТЕЛЯ ====================
else:
    teacher_menu = st.sidebar.selectbox("Бөлімдер:", ["📈 Басты бақылау панелі", "📝 Оқушыларды базаға енгізу", "🧠 ИИ Психологиялық талдау"])
    
    if teacher_menu == "📈 Басты бақылау панелі":
        st.title("🧑‍🏫 Мұғалім мен Психологтың жұмыс кабинеті")
        st.write("Қош келдіңіз! Бұл бөлімде сіз мектептегі оқушылардың сандық дамуын бақылай аласыз.")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Базадағы нақты оқушылар:", len(st.session_state['real_students_db']))
        c2.metric("Бүгін сабаққа қатысу көрсеткіші:", "94%")
        c3.metric("ИИ Модель дәлдігі:", "75.0%")
        
    elif teacher_menu == "📝 Оқушыларды базаға енгізу":
        st.title("📝 Жаңа оқушыны жүйеге тіркеу және деректер жинау")
        with st.form("teacher_add_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Оқушының аты-жөні (ФИО):")
                grade = st.selectbox("Сыныбы:", ["9-А", "9-Ә", "10-А"])
                lms = st.slider("LMS-ке апталық кіруі:", 0, 30, 10)
                delay = st.slider("Тапсырма кешігуі (сағат):", 0, 120, 24)
            with col2:
                screen = st.slider("Экран уақыты (сағат):", 1.0, 15.0, 5.0)
                night = st.slider("Түнгі белсенділік балы (1-10):", 1, 10, 4)
                math = st.number_input("Математика бағасы:", 0, 100, 75)
                lang = st.number_input("Тіл бағасы:", 0, 100, 75)
                att = st.slider("Қатысу пайызы (%):", 40.0, 100.0, 95.0)
            
            if st.form_submit_button("Базаға сақтау", type="primary"):
                if name:
                    new_student = {
                        'Оқушы ID-і': f"STU_{np.random.randint(1000, 9999)}", 'Аты-жөні': name, 'Сыныбы': grade,
                        'LMS-ке кіру жиілігі (аптасына)': lms, 'Тапсырма кешігуі (сағатпен)': delay,
                        'Экран уақыты (күнделікті, сағат)': screen, 'Түнгі белсенділік (1-10 балл)': night,
                        'Математика бағасы': math, 'Тілдік пәндер бағасы': lang, 'Сабаққа қатысуы (%)': att
                    }
                    st.session_state['real_students_db'] = pd.concat([st.session_state['real_students_db'], pd.DataFrame([new_student])], ignore_index=True)
                    st.success(f"{name} жүйеге қосылды!")
                else:
                    st.error("Аты-жөнін жазыңыз!")
                    
    elif teacher_menu == "🧠 ИИ Психологиялық талдау":
        st.title("🧠 Жасанды Интеллект арқылы қауіп топтарын анықтау")
        if len(st.session_state['real_students_db']) == 0:
            st.info("💡 Анализ жасау үшін алдымен оқушыларды тіркеңіз немесе жүйеге баға енгізіңіз.")
        else:
            df = st.session_state['real_students_db'].copy()
            X_batch = df.drop(columns=['Оқушы ID-і', 'Аты-жөні', 'Сыныбы'])
            X_batch_scaled = scaler.transform(X_batch)
            preds = model.predict(X_batch_scaled)
            df['ИИ Қорытындысы'] = le.inverse_transform(preds)
            
            def highlight(val):
                if val == 'Жоғары': return 'background-color: #ffcccc; color: black;'
                elif val == 'Орташа': return 'background-color: #ffe6cc; color: black;'
                return 'background-color: #e6ffcc; color: black;'
                
            st.dataframe(df.style.applymap(highlight, subset=['ИИ Қорытындысы']))

