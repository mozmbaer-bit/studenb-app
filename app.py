import streamlit as st
import pickle
import pandas as pd

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('🎓 Student Success Prediction with AI')
st.write('Enter student data and the AI model will predict Pass or Fail')

attendance = st.slider('Attendance %', 40, 100, 80)
midterm = st.slider('Midterm Score', 20, 100, 70)
homework = st.slider('Homework Score / 10', 0, 10, 7)
hours = st.slider('Study Hours per Day', 0, 10, 5)

if st.button('Predict Result'):
    data = pd.DataFrame([[attendance, midterm, homework, hours]], columns=['attendance','midterm_score','homework','study_hours'])
    pred = model.predict(data)[0]
    if pred == 1:
        st.success('✅ This student will PASS!')
    else:
        st.error('⚠️ This student is at RISK of failing - needs support!')