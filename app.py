
import streamlit as st
import pickle
import pandas as pd

st.set_page_config(layout="wide")
st.title("💼 Salary Prediction App")

st.write("Enter job details to predict salary.")

# Load model
try:
    model = pickle.load(open('linear_regression_model.pkl', 'rb'))
except:
    st.error("❌ Model file not found!")
    st.stop()

# Load encoders
try:
    company_encoder = pickle.load(open('encoders/company_encoder.pkl', 'rb'))
    job_title_encoder = pickle.load(open('encoders/job_title_encoder.pkl', 'rb'))
    location_encoder = pickle.load(open('encoders/location_encoder.pkl', 'rb'))
    employment_encoder = pickle.load(open('encoders/employment_encoder.pkl', 'rb'))
    role_encoder = pickle.load(open('encoders/role_encoder.pkl', 'rb'))
except:
    st.error("❌ Encoder files not found!")
    st.stop()

# Sidebar inputs
st.sidebar.header("Input Features")

rating = st.sidebar.slider('Rating', 0.0, 5.0, 3.5)
company = st.sidebar.selectbox('Company Name', company_encoder.classes_)
job_title = st.sidebar.selectbox('Job Title', job_title_encoder.classes_)
salaries_reported = st.sidebar.number_input('Salaries Reported', 1, 100, 1)
location = st.sidebar.selectbox('Location', location_encoder.classes_)
employment = st.sidebar.selectbox('Employment Status', employment_encoder.classes_)
role = st.sidebar.selectbox('Job Role', role_encoder.classes_)

# Create input dataframe (ORIGINAL values)
data = {
    'Rating': [rating],
    'Company Name': [company],
    'Job Title': [job_title],
    'Salaries Reported': [salaries_reported],
    'Location': [location],
    'Employment Status': [employment],
    'Job Roles': [role]
}

df_input = pd.DataFrame(data)

# Show user input
st.subheader("🧾 User Input")
st.write(df_input)

# Predict
if st.button("Predict Salary"):
    try:
        df_encoded = df_input.copy()

        # Encode categorical columns
        df_encoded['Company Name'] = company_encoder.transform(df_encoded['Company Name'])
        df_encoded['Job Title'] = job_title_encoder.transform(df_encoded['Job Title'])
        df_encoded['Location'] = location_encoder.transform(df_encoded['Location'])
        df_encoded['Employment Status'] = employment_encoder.transform(df_encoded['Employment Status'])
        df_encoded['Job Roles'] = role_encoder.transform(df_encoded['Job Roles'])

        # Ensure correct column order
        df_encoded = df_encoded[
            ['Rating', 'Company Name', 'Job Title', 'Salaries Reported',
             'Location', 'Employment Status', 'Job Roles']
        ]

        # Prediction
        prediction = model.predict(df_encoded)[0]

        st.subheader("💰 Predicted Salary")
        st.success(f"${prediction:,.2f}")

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.write("Debug Info:")
        st.write(df_encoded)
        st.write(df_encoded.dtypes)
