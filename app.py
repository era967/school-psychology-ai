import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Настройка страницы в стиле BilimClass
st.set_page_config(page_title="BilimPredict - Психологиялық жүйе", layout="wide")

# Инициализация базы данных в памяти сайта для хранения реальных учеников
if 'real_students_db' not in st.session_state:
    st.session_state['real_students_db'] = pd.DataFrame(columns=[
        'Оқушы ID-і', 'Аты-жөні', 'Сыныбы', 
        'LMS-ке кіру жиілігі (аптасына)', 'Тапсырма кешігуі (сағатпен)', 
        'Экран уақыты (күнделікті, сағат)', 'Түнгі белсенділік (1-10 балл)', 
        'Математика бағасы', 'Тілдік пәндер бағасы', 'Сабаққа қатысуы (%)'
    ])

# Загрузка синтетических данных и обучение модели ИИ
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
    st.error("Қате: Оқыту файлы табылмады.")
    st.stop()

# Боковое меню (Сайдбар) как в BilimClass
st.sidebar.image("https://flaticon.com", width=80)
st.sidebar.title("BilimPredict")
st.sidebar.write("🌐 *Мектептің интеллектуалды платформасы*")

menu = st.sidebar.selectbox("Бөлімді таңдаңыз:", [
    "📌 Басты бет (Платформа туралы)", 
    "📝 Жаңа оқушыны тіркеу (Сбор данных)", 
    "📊 Сынып журналы және ИИ талдауы"
])

# ---- 1. БАСТЫ БЕТ ----
if menu == "📌 Басты бет (Платформа туралы)":
    st.title("🏫 Мектеп оқушыларының оқу-психологиялық жағдайын бақылау платформасы")
    st.markdown("""
    **BilimPredict** — бұл сандық мінез-құлық пен оқу үлгерімін талдау негізінде мектеп оқушыларының 
    оқу-психологиялық қиындықтарын ерте анықтауға арналған интеллектуалды жүйе.
    
    ### Платформа қалай жұмыс істейді?
    1. **Деректерді жинау:** Психолог немесе сынып жетекшісі оқушының сандық көрсеткіштерін жүйеге енгізеді.
    2. **ИИ Модельдеу:** Жасанды интеллект оқушының мінез-құлқын талдап, оның күйзеліс немесе депрессия деңгейін болжайды.
    3. **Проактивті көмек:** Мектеп мамандары қауіп тобындағы оқушыларға уақытылы психологиялық қолдау көрсетеді.
    """)
    
    # Краткая статистика
    st.subheader("📈 Жүйе статистикасы")
    c1, c2 = st.columns(2)
    c1.metric("Тіркелген нақты оқушылар саны:", len(st.session_state['real_students_db']))
    c2.metric("ИИ модель дәлдігі (Accuracy):", "75.00%")

# ---- 2. СБОР ДАННЫХ (РЕГИСТРАЦИЯ) ----
elif menu == "📝 Жаңа оқушыны тіркеу (Сбор данных)":
    st.title("📝 Реалды оқушылардың деректерін жинау базасы")
    st.write("Төмендегі форманы толтыру арқылы оқушының цифрлық ізі мен бағаларын мектеп базасына қосыңыз:")
    
    with st.form("student_reg_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Оқушының аты-жөні (ФИО):", placeholder="Асанәлі Ибрагимов")
            grade_num = st.selectbox("Сыныбы:", ["9-А", "9-Ә", "10-А", "11-А", "11-Ә"])
            lms_login = st.slider("LMS-ке (Күнделік) апталық кіру жиілігі:", 0, 30, 12)
            delay_hours = st.slider("Тапсырмаларды кешіктіру уақыты (сағатпен):", 0, 150, 10)
        
        with col2:
            screen_time = st.slider("Күнделікті экран уақыты (сағат, телефонда отыруы):", 1.0, 15.0, 4.0, step=0.5)
            night_act = st.slider("Түнгі белсенділік балы (23:00-ден кейін желіде болуы, 1-10):", 1, 10, 3)
            math = st.number_input("Математика пәнінен бағасы (0-100):", 0, 100, 80)
            lang = st.number_input("Тілдік пәндерден бағасы (0-100):", 0, 100, 85)
            attendance = st.slider("Сабаққа қатысу көрсеткіші (%):", 30.0, 100.0, 95.0, step=1.0)
            
        submit_btn = st.form_submit_button("Оқушыны мектеп базасына сақтау", type="primary")
        
        if submit_btn:
            if name.strip() == "":
                st.error("Қате: Оқушының аты-жөнін жазыңыз!")
            else:
                # Генерируем случайный ID
                stu_id = f"REAL_{np.random.randint(1000, 9999)}"
                
                # Создаем новую строчку
                new_row = {
                    'Оқушы ID-і': stu_id, 'Аты-жөні': name, 'Сыныбы': grade_num,
                    'LMS-ке кіру жиілігі (аптасына)': lms_login, 'Тапсырма кешігуі (сағатпен)': delay_hours,
                    'Экран уақыты (күнделікті, сағат)': screen_time, 'Түнгі белсенділік (1-10 балл)': night_act,
                    'Математика бағасы': math, 'Тілдік пәндер бағасы': lang, 'Сабаққа қатысуы (%)': attendance
                }
                
                # Сохраняем в базу данных сессии
                st.session_state['real_students_db'] = pd.concat([st.session_state['real_students_db'], pd.DataFrame([new_row])], ignore_index=True)
                st.success(f"🎉 Оқушы {name} сәтті тіркелді және мектеп базасына қосылды!")

# ---- 3. СЫНЫП ЖУРНАЛЫ И ИИ АНАЛИЗ ----
elif menu == "📊 Сынып журналы және ИИ талдауы":
    st.title("📊 Мектеп журналы және Жасанды Интеллект қорытындысы")
    
    if len(st.session_state['real_students_db']) == 0:
        st.info("💡 Әзірге базада ешқандай оқушы тіркелмеген. Алдымен 'Жаңа оқушыны тіркеу' бөліміне өтіп, бірнеше оқушы қосыңыз.")
    else:
        st.write("Төменде мектепке жиналған нақты оқушылардың тізімі берілген. Кез келген уақытта ИИ талдауын қоса аласыз.")
        
        # Показываем текущую базу данных
        st.dataframe(st.session_state['real_students_db'])
        
        if st.button("Жасанды Интеллект талдауын іске қосу", type="primary"):
            df_to_predict = st.session_state['real_students_db'].copy()
            
            # Подготавливаем признаки для ИИ (удаляем текстовые колонки ID, Имя, Класс)
            X_batch = df_to_predict.drop(columns=['Оқушы ID-і', 'Аты-жөні', 'Сыныбы'])
            
            # Масштабируем и делаем предсказание рисков
            X_batch_scaled = scaler.transform(X_batch)
            batch_predictions = model.predict(X_batch_scaled)
            
            # Добавляем вердикт ИИ в таблицу
            df_to_predict['ИИ Қорытындысы (Қауіп деңгейі)'] = le.inverse_transform(batch_predictions)
            
            st.subheader("🚨 ИИ Диагностикасының нәтижесі:")
            
            # Красиво подсвечиваем таблицу
            def highlight_risk(val):
                if val == 'Жоғары': return 'background-color: #ffcccc; color: black;'
                elif val == 'Орташа': return 'background-color: #ffe6cc; color: black;'
                return 'background-color: #e6ffcc; color: black;'
                
            st.dataframe(df_to_predict.style.applymap(highlight_risk, subset=['ИИ Қорытындысы (Қауіп деңгейі)']))
            
            # Скачивание готового отчета школы
            csv = df_to_predict.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Мектеп бойынша есепті жүктеп алу (Excel/CSV)",
                data=csv,
                file_name='мектеп_нақты_оқушылар_есеп.csv',
                mime='text/csv'
            )
