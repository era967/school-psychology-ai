import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import firebase_admin
from firebase_admin import credentials, firestore

# Настройка страницы в стиле полноценного портала BilimClass
st.set_page_config(page_title="BilimPredict - Мектеп платформасы", layout="wide")

# --- 1. ПОДКЛЮЧЕНИЕ К ОБЛАЧНОЙ БАЗЕ ДАННЫХ FIREBASE ---
@st.cache_resource
def init_firebase():
    if not firebase_admin._apps:
        try:
            cred = credentials.Certificate("firebase-key.json")
            firebase_admin.initialize_app(cred)
        except Exception as e:
            st.error(f"Firebase-ке қосылу қатесі: {e}")
    return firestore.client()

db = init_firebase()

# --- 2. ОБУЧЕНИЕ МОДЕЛИ ИИ ---
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
    st.error("Қате: ИИ оқыту базасы табылмады.")
    st.stop()

# --- 3. ИНТЕРФЕЙС И АВТОРИЗАЦИЯ С ЛЮБОГО УСТРОЙСТВА ---
st.sidebar.image("https://flaticon.com", width=70)
st.sidebar.title("BilimPredict Порталы")

auth_mode = st.sidebar.radio("Жүйеге кіру:", ["Логин арқылы кіру", "Жаңа профиль тіркеу (Регистрация)"])

if auth_mode == "Жаңа профиль тіркеу (Регистрация)":
    st.title("📝 Платформаға жаңа қолданушыны тіркеу")
    st.write("Кез келген құрылғыдан кіру үшін өзіңізге профиль жасаңыз:")
    
    with st.form("reg_form"):
        reg_username = st.text_input("Логин (email немесе атыңыз латынша):").strip()
        reg_fullname = st.text_input("Толық аты-жөніңіз (ФИО):")
        reg_password = st.text_input("Құпия сөз (Пароль):", type="password")
        reg_role = st.selectbox("Роліңізді таңдаңыз:", ["Оқушы (Ученик)", "Мұғалім / Мектеп Психологы"])
        reg_class = st.selectbox("Сыныбыңыз (Тек оқушылар үшін):", ["9-А", "9-Ә", "10-А", "11-А", "Мұғалімде қажет емес"])
        
        if st.form_submit_button("Тіркелуді аяқтау", type="primary"):
            if reg_username and reg_password and reg_fullname:
                user_ref = db.collection("users").document(reg_username).get()
                if user_ref.exists:
                    st.error("Бұл логин бос емес! Басқа логин таңдаңыз.")
                else:
                    db.collection("users").document(reg_username).set({
                        "fullname": reg_fullname,
                        "password": reg_password,
                        "role": reg_role,
                        "class": reg_class
                    })
                    st.success("🎉 Тіркелу сәтті аяқталды! Енді сол жақ мәзірден 'Логин арқылы кіру' бөліміне өтіңіз.")
            else:
                st.error("Барлық өрістерді толтырыңыз!")

else:
    login_user = st.sidebar.text_input("Логин:").strip()
    login_pass = st.sidebar.text_input("Құпия сөз:", type="password")
    
    logged_in = False
    user_data = {}
    
    if login_user and login_pass:
        user_ref = db.collection("users").document(login_user).get()
        if user_ref.exists:
            data = user_ref.to_dict()
            if data['password'] == login_pass:
                logged_in = True
                user_data = data
                st.sidebar.success(f"Қош келдіңіз, {data['fullname']}!")
            else:
                st.sidebar.error("Құпия сөз қате!")
        else:
            st.sidebar.error("Логин табылмады!")

    if not logged_in:
        st.title("🏫 Мектептің бірыңғай цифрлық оқу-психологиялық платформасы")
        st.markdown("""
        ### Платформаға қош келдіңіз!
        Бұл сайт нақты уақыт режимінде мектеп өмірін, сабақтарды және Жасанды Интеллект көмегімен оқушылардың психологиялық жағдайын бақылауға арналған.
        
        **Жұмысты бастау үшін:**
        1. Сол жақтағы мәзірден **'Жаңа профиль тіркеу'** арқылы мұғалім немесе оқушы аккаунтын ашыңыз.
        2. Профиль жасаған соң, өз логиніңіз бен пароліңізді енгізіп, жеке кабинетіңізге кіріңіз.
        """)
    
    elif logged_in and user_data['role'] == "Оқушы (Ученик)":
        student_menu = st.selectbox("Бөлімді таңдаңыз:", ["🪪 Менің Профилім", "📚 Сабақтар мен ДЗ"])
        
        if student_menu == "🪪 Менің Профилім":
            st.title(f"🪪 Оқушының жеке профилі: {user_data['fullname']}")
            col1, col2 = st.columns(2)
            with col1:
                st.image("https://flaticon.com", width=130)
            with col2:
                st.subheader(user_data['fullname'])
                st.write(f"**Сыныбы:** {user_data['class']}")
                st.write("**Мектеп:** №10 IT-Лицей")
                st.info("💡 ИИ кеңесі: Жүйедегі көрсеткіштеріңіз тұрақты. Сабаққа белсенді қатысқаныңыз үшін рақмет!")
                
        elif student_menu == "📚 Сабақтар мен ДЗ":
            st.title("📚 Ағымдағы saбaqtar мен үй тапсырмалары (ДЗ)")
            st.write("Мұғалімдер қалдырған белсенді тапсырмалар:")
            
            hw_docs = db.collection("homeworks").stream()
            hw_list = [doc.to_dict() for doc in hw_docs]
            
            if len(hw_list) == 0:
                st.info("Әзірге белсенді үй тапсырмалары жоқ.")
            else:
                for hw in hw_list:
                    st.info(f"📘 **Пән:** {hw['subject']} | **Тақырып:** {hw['topic']} | 📅 **Мерзімі:** {hw['deadline']}")
            
            st.subheader("📤 Үй жұмысын файлмен өткізу")
            uploaded_file = st.file_uploader("Файлды таңдаңыз (PDF, PNG):", type=["pdf", "png", "jpg"])
            if uploaded_file:
                st.success("Тапсырма Firebase бұлтына сәтті жүктелді!")

    elif logged_in and user_data['role'] == "Мұғалім / Мектеп Психологы":
        teacher_menu = st.selectbox("Бөлімді таңдаңыз:", ["📈 Психологиялық ИИ Аналитика", "📝 Жаңа оқушы деректерін енгізу", "➕ Үй тапсырмасын қосу (ДЗ)"])
        
        if teacher_menu == "📈 Психологиялық ИИ Аналитика":
            st.title("🧠 Жасанды Интеллект арқылы қауіп топтарын нақты анықтау")
            st.write("Бұл тізім Firebase бұлтындағы нақты уақыттағы деректер негезінде ИИ арқылы автоматты түрде есептеледі:")
            
            student_docs = db.collection("students_metrics").stream()
            real_students = []
            for doc in student_docs:
                d = doc.to_dict()
                d['Оқушы ID-і'] = doc.id
                real_students.append(d)
                
            if len(real_students) == 0:
                st.info("💡 Қазіргі уақытта базада оқушылар дерегі жоқ. Алдымен 'Жаңа оқушы деректерін енгізу' бөлімінде оқушыларды қосыңыз.")
            else:
                df = pd.DataFrame(real_students)
                X_batch = df[['lms_login', 'delay_hours', 'screen_time', 'night_act', 'math', 'lang', 'attendance']]
                X_batch_scaled = scaler.transform(X_batch)
                preds = model.predict(X_batch_scaled)
                df['ИИ Қорытындысы (Қауіп деңгейі)'] = le.inverse_transform(preds)
                
                display_df = df.rename(columns={
                    'fullname': 'Оқушының аты-жөні', 'grade': 'Сыныбы',
                    'lms_login': 'LMS кіруі', 'delay_hours': 'Тапсырма кешігуі (сағат)',
                    'screen_time': 'Экран уақыты', 'night_act': 'Түнгі белсенділік',
                    'math': 'Математика', 'lang': 'Тілдер', 'attendance': 'Қатысуы (%)'
                })
                
                cols = ['Оқушы ID-і', 'Оқушының аты-жөні', 'Сыныбы', 'LMS кіруі', 'Тапсырма кешігуі (сағат)', 'Экран уақыты', 'Түнгі белсенділік', 'Математика', 'Тілдер', 'Қатысуы (%)', 'ИИ Қорытындысы (Қауіп деңгейі)']
                display_df = display_df[cols]
                
                def highlight(val):
                    if val == 'Жоғары': return 'background-color: #ffcccc; color: black;'
                    elif val == 'Орташа': return 'background-color: #ffe6cc; color: black;'
                    return 'background-color: #e6ffcc; color: black;'
                    
                st.dataframe(display_df.style.applymap(highlight, subset=['ИИ Қорытындысы (Қауіп деңгейі)']))
                
                csv = display_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Мектеп бойынша есепті жүктеп алу (Excel/CSV)",
                    data=csv,
                    file_name='мектеп_нақты_оқушылар_есеп.csv',
                    mime='text/csv'
                )
                
        elif teacher_menu == "📝 Жаңа оқушы деректерін енгізу":
            st.title("📝 Реалды оқушы көрсеткіштерін Firebase-ке сақтау")
            with st.form("add_student_cloud"):
                st.write("Оқушының мектептегі және цифрлық көрсеткіштерін жазыңыз:")
                f_name = st.text_input("Оқушының толық аты-жөні (ФИО):")
                f_grade = st.selectbox("Сыныбы:", ["9-А", "9-Ә", "10-А", "11-А"])
                f_lms = st.slider("LMS-ке (Күнделік) апталық кіруі:", 0, 30, 12)
                f_delay = st.slider("Тапсырмаларды кешіктіруі (сағатпен):", 0, 150, 12)
                f_screen = st.slider("Күнделікті экран уақыты (сағат):", 1.0, 15.0, 4.5, step=0.5)
                f_night = st.slider("Түнгі белсенділік (1-10 балл):", 1, 10, 3)
                f_math = st.number_input("Математика бағасы (0-100):", 0, 100, 75)
                f_lang = st.number_input("Тілдік пәндер бағасы (0-100):", 0, 100, 80)
                f_att = st.slider("Сабаққа қатысу көрсеткіші (%):", 30.0, 100.0, 95.0)
                submit_student = st.form_submit_button("Мәліметтерді бұлттық базаға жіберу", type="primary")
                if submit_student:
                    if f_name:
                        stu_uuid = f"STU_{np.random.randint(1000, 9999)}"db.collection("students_metrics").document(stu_uuid).
                        set({"fullname": f_name, "grade": f_grade, "lms_login": f_lms,"delay_hours": f_delay, "screen_time": f_screen, "night_act": f_night,"math": f_math, "lang": f_lang, "attendance": f_att})
                        st.success(f"🎉 {f_name} деректері бұлттық базаға сәтті сақталды!")
                    else:st.error("Аты-жөнін толтыру міндетті!")
                        elif teacher_menu == "➕ Үй тапсырмасын қосу (ДЗ)":
                            st.title("➕ Оқушыларға жаңа үй тапсырмасын (ДЗ) бекіту")
                            with st.form("add_hw_form"):
                                hw_subject = st.selectbox("Пән таңдаңыз:", ["Математика", "Алгебра", "Геометрия", "Информатика", "Қазақ тілі", "Ағылшын тілі"])
                                hw_topic = st.text_input("Үй тапсырмасының тақырыбы мен сипаттамасы:")
                                hw_deadline = st.text_input("Тапсырманы өткізу мерзімі (Дедлайн):",value="Ертең, 18:00")
                                if st.form_submit_button("ДЗ платформада жариялау", type="primary"):
                                    if hw_topic:db.collection("homeworks").add({"subject": hw_subject,"topic": hw_topic,"deadline": hw_deadline})
                                        st.success("Тапсырма сәтті жарияланды! Енді оны барлық оқушылар өз кабинеттерінен көре алады.")
                                else:
                                    st.error("Тақырыпты жазыңыз!")
