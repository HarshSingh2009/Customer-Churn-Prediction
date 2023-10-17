import tensorflow as tf
import streamlit as st
from streamlit_option_menu import option_menu
from pipeline import PredictionPipeline
import sklearn

# Credit Score ranges from 100 - 1000
# Tenure is between 0-10

with st.sidebar:
    st.image('tensorflow_developer_cert.png')
    selected = option_menu(
        menu_title='Customer Churn Prediction',
        menu_icon='🕵️',
        options=['Home']
    )

if selected == 'Home':
    st.title('Customer Churn prediction (DL)')
    st.text('')

    credit_score = st.slider('Credit Score', 1, 1000)
    gender = st.radio('Pick Gender: ', ['Male', 'Female'])
    age = st.slider('Age', 1, 100)
    tenure = st.slider('Tenure', 1, 10)
    balance = st.text_input('Balance of user: ')
    num_products = st.number_input('Number of Products bought by user: ', 1, 5)
    has_cred = st.radio('Does User have Credit card: ', ['Yes', 'No'])
    active_member = st.radio('Is User an active customer: ', ['Yes', 'No'])
    estimated_salary = st.text_input('Estimated Salary of User: ')
    living = st.selectbox('Where does user live: ', ['France', 'Germany', 'Spain'])


    if st.button('Predict if user will churn!!'):
        if gender == 'Male': gender = 0
        else: gender = 1
        balance = float(balance)
        if has_cred == 'Yes': has_cred = 1
        else: has_cred = 0
        if active_member == 'Yes': active_member = 1
        else: active_member = 0
        estimated_salary = float(estimated_salary)
        
        living_list = []
        if living == 'Spain': living_list = [0, 0, 1]
        elif living == 'France': living_list = [1, 0, 0]
        else: living_list = [0, 1, 0]

        inputs = [credit_score, gender, age, tenure, balance, num_products, has_cred, active_member, estimated_salary, living_list[0], living_list[1], living_list[2],]

        pipeline = PredictionPipeline()
        probs, status = pipeline.predict(inputs)

        st.text('')
        if status == 0:
            st.write(f'Customer will not churn with {probs*100:2f}% probablity')
        else: 
            st.balloons()
            st.success(f'Customer will churn with {int(probs*100)}% probablity')
        