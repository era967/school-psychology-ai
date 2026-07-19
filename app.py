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
            
            # --- ОСЫ БӨЛІК ЖАҢАДАН ҚОСЫЛДЫ (Перевод колонок на казахский) ---
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
            # Бағандардың атын ауыстырамыз
            display_df = input_df.rename(columns=ru_columns)
            # ---------------------------------------------------------------
            
            st.subheader("🚀 Талдау нәтижесі (ИИ әр оқушыны есептеп шықты):")
            st.write(display_df) # Енді экранға қазақшаланған кесте шығады
            
            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Нәтижені жүктеп алу (Скачать отчет)",
                data=csv,
                file_name='мектеп_психологиялық_талдау_есеп.csv',
                mime='text/csv',
            )
