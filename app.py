import streamlit as st
import pickle
import pandas as pd

st.set_page_config(layout="wide")
st.title('💼 Salary Prediction App')

st.write("Predict salary based on job details.")

# Load model and encoders
try:
    model = pickle.load(open('linear_regression_model.pkl', 'rb'))
    company_encoder = pickle.load(open('company_encoder.pkl', 'rb'))
    job_title_encoder = pickle.load(open('job_title_encoder.pkl', 'rb'))
    location_encoder = pickle.load(open('location_encoder.pkl', 'rb'))
    employment_encoder = pickle.load(open('employment_encoder.pkl', 'rb'))
    role_encoder = pickle.load(open('role_encoder.pkl', 'rb'))
except:
    st.error("Model or encoder files not found!")
    st.stop()

# Sidebar input
st.sidebar.header('Input Features')

def user_input_features():
    rating = st.sidebar.slider('Rating', 0.0, 5.0, 3.5)
    
    company = st.sidebar.selectbox(
        'Company Name', company_encoder.classes_
    )
    
    job_title = st.sidebar.selectbox(
        'Job Title', job_title_encoder.classes_
    )
    
    salaries_reported = st.sidebar.number_input(
        'Salaries Reported', min_value=1, value=1
    )
    
    location = st.sidebar.selectbox(
        'Location', location_encoder.classes_
    )
    
    employment = st.sidebar.selectbox(
        'Employment Status', employment_encoder.classes_
    )
    
    role = st.sidebar.selectbox(
        'Job Role', role_encoder.classes_
    )

    # Encoding
    data = {
        'Rating': rating,
        'Company Name': company_encoder.transform([company])[0],
        'Job Title': job_title_encoder.transform([job_title])[0],
        'Salaries Reported': salaries_reported,
        'Location': location_encoder.transform([location])[0],
        'Employment Status': employment_encoder.transform([employment])[0],
        'Job Roles': role_encoder.transform([role])[0]
    }

    return pd.DataFrame(data, index=[0])

df_input = user_input_features()

st.subheader('📊 User Input')
st.write(df_input)

# Prediction
if st.button('Predict Salary'):
    prediction = model.predict(df_input)
    st.subheader('💰 Predicted Salary')
    st.success(f"${prediction[0]:,.2f}")
