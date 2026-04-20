import streamlit as st
import pickle
import pandas as pd

st.set_page_config(layout="wide")
st.title("💼 Salary Prediction App")

st.write("Predict salary using job details.")

# Load model (pipeline)
try:
    model = pickle.load(open('linear_regression_model.pkl', 'rb'))
except:
    st.error("❌ Model file not found!")
    st.stop()

# Load dataset (for dropdown values)
try:
    df = pd.read_csv('Salary_Dataset_DataScienceLovers.csv')
except:
    st.error("❌ Dataset not found!")
    st.stop()

st.sidebar.header("Input Features")

def user_input():
    rating = st.sidebar.slider('Rating', 0.0, 5.0, 3.5)

    # ✅ Get unique values from dataset
    company = st.sidebar.selectbox('Company Name', sorted(df['Company Name'].dropna().unique()))
    job_title = st.sidebar.selectbox('Job Title', sorted(df['Job Title'].dropna().unique()))
    salaries_reported = st.sidebar.number_input('Salaries Reported', 1, 100, 1)
    location = st.sidebar.selectbox('Location', sorted(df['Location'].dropna().unique()))
    employment = st.sidebar.selectbox('Employment Status', sorted(df['Employment Status'].dropna().unique()))
    role = st.sidebar.selectbox('Job Role', sorted(df['Job Roles'].dropna().unique()))

    data = {
        'Rating': rating,
        'Company Name': [company],
        'Job Title': [job_title],
        'Salaries Reported': [salaries_reported],
        'Location': [location],
        'Employment Status': [employment],
        'Job Roles': [role]
    }

    return pd.DataFrame(data, index=[0])

df_input = user_input()

st.subheader("🧾 User Input")
st.write(df_input)

# Prediction
if st.button("Predict Salary"):
    prediction = model.predict(df_input)[0]

    st.subheader("💰 Predicted Salary")
    st.success(f"${prediction:,.2f}")
