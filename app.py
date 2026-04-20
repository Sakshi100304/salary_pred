
import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title('Salary Prediction App')

st.write("This app predicts salary based on various job attributes.")
st.write("Please enter numerical values for all features. For categorical features, use their previously encoded numerical representations.")

# Load the trained model
try:
    with open('linear_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Error: 'linear_regression_model.pkl' not found. Make sure the model was saved correctly.")
    st.stop()

# Create input fields for features
st.sidebar.header('Input Features')

def user_input_features():
    rating = st.sidebar.slider('Rating', 0.0, 5.0, 3.5)
    company_name = st.sidebar.number_input('Company Name (Encoded Number)', min_value=0, value=0)
    job_title = st.sidebar.number_input('Job Title (Encoded Number)', min_value=0, value=0)
    salaries_reported = st.sidebar.number_input('Salaries Reported', min_value=1, value=1)
    location = st.sidebar.number_input('Location (Encoded Number)', min_value=0, value=0)
    employment_status = st.sidebar.number_input('Employment Status (Encoded Number)', min_value=0, value=0)
    job_roles = st.sidebar.number_input('Job Roles (Encoded Number)', min_value=0, value=0)

    data = {
        'Rating': rating,
        'Company Name': company_name,
        'Job Title': job_title,
        'Salaries Reported': salaries_reported,
        'Location': location,
        'Employment Status': employment_status,
        'Job Roles': job_roles
    }
    features = pd.DataFrame(data, index=[0])
    return features

df_input = user_input_features()

st.subheader('User Input Features')
st.write(df_input)

if st.button('Predict Salary'):
    prediction = model.predict(df_input)
    st.subheader('Predicted Salary')
    st.write(f'The predicted salary is: ${prediction[0]:,.2f}')
