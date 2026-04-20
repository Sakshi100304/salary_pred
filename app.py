
import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title('Salary Prediction App')

st.write("This app predicts salary based on various job attributes. Categorical features can now be selected from dropdowns.")

# Load the trained model
try:
    with open('linear_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Error: 'linear_regression_model.pkl' not found. Make sure the model was saved correctly.")
    st.stop()

# Load the label encoders
try:
    with open('label_encoders.pkl', 'rb') as file:
        label_encoders = pickle.load(file)
except FileNotFoundError:
    st.error("Error: 'label_encoders.pkl' not found. Make sure the label encoders were saved correctly.")
    st.stop()

# Create input fields for features
st.sidebar.header('Input Features')

def user_input_features():
    rating = st.sidebar.slider('Rating', 0.0, 5.0, 3.5)
    
    # For categorical features, use selectbox and encode the selected value
    company_name_options = list(label_encoders['Company Name'].classes_)
    selected_company_name = st.sidebar.selectbox('Company Name', company_name_options)
    company_name_encoded = label_encoders['Company Name'].transform([selected_company_name])[0]

    job_title_options = list(label_encoders['Job Title'].classes_)
    selected_job_title = st.sidebar.selectbox('Job Title', job_title_options)
    job_title_encoded = label_encoders['Job Title'].transform([selected_job_title])[0]

    salaries_reported = st.sidebar.number_input('Salaries Reported', min_value=1, value=1)
    
    location_options = list(label_encoders['Location'].classes_)
    selected_location = st.sidebar.selectbox('Location', location_options)
    location_encoded = label_encoders['Location'].transform([selected_location])[0]

    employment_status_options = list(label_encoders['Employment Status'].classes_)
    selected_employment_status = st.sidebar.selectbox('Employment Status', employment_status_options)
    employment_status_encoded = label_encoders['Employment Status'].transform([selected_employment_status])[0]

    job_roles_options = list(label_encoders['Job Roles'].classes_)
    selected_job_roles = st.sidebar.selectbox('Job Roles', job_roles_options)
    job_roles_encoded = label_encoders['Job Roles'].transform([selected_job_roles])[0]

    data = {
        'Rating': rating,
        'Company Name': company_name_encoded,
        'Job Title': job_title_encoded,
        'Salaries Reported': salaries_reported,
        'Location': location_encoded,
        'Employment Status': employment_status_encoded,
        'Job Roles': job_roles_encoded
    }
    features = pd.DataFrame(data, index=[0])
    return features

df_input = user_input_features()

st.subheader('User Input Features (Encoded)')
st.write(df_input)

if st.button('Predict Salary'):
    prediction = model.predict(df_input.to_numpy())
    st.subheader('Predicted Salary')
    st.write(f'The predicted salary is: ${prediction[0]:,.2f}')
