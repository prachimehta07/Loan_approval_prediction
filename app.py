
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_option_menu import option_menu

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Function to make predictions
def predict_loan_approval(features):
    prediction = model.predict(features)
    return prediction

# Create a Streamlit app
st.title("Bank Loan Approval Prediction")

with st.sidebar:
    selected = option_menu("Main Menu", ["Form", "Dataset"], default_index=0)

if selected == "Form":
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
        married = st.selectbox("Married", ["Yes", "No"], key="married")
        Dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'], key="dependents")
        education = st.selectbox("Education", ["Graduate", "Not Graduate"], key="education")
        self_employed = st.selectbox("Self Employed", ["Yes", "No"], key="self_employed")

    with col2:
        applicant_income = st.number_input("Applicant Income", min_value=0, key="applicant_income")
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0, key="coapplicant_income")
        loan_amount = st.number_input("Loan Amount", min_value=0, key="loan_amount")
        loan_amount_term = st.number_input("Loan Amount Term", min_value=0, key="loan_amount_term")
        credit_history = st.selectbox("Credit History", [0, 1], key="credit_history")
        property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"], key="property_area")

    # Convert categorical variables to numerical variables
    gender_map = {"Male": 1, "Female": 0}
    married_map = {"Yes": 1, "No": 0}
    Dependents_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
    education_map = {"Graduate": 0, "Not Graduate": 1}
    self_employed_map = {"Yes": 1, "No": 0}
    property_area_map = {"Urban": 2, "Rural": 0, "Semiurban": 1}

    # Create a pandas DataFrame from user input
    features = pd.DataFrame({
        "Gender": [gender_map[gender]],
        "Married": [married_map[married]],
        "Dependents": [Dependents_map[Dependents]],
        "Education": [education_map[education]],
        "Self_Employed": [self_employed_map[self_employed]],
        "ApplicantIncome": [applicant_income],
        "CoapplicantIncome": [coapplicant_income],
        "LoanAmount": [loan_amount],
        "Loan_Amount_Term": [loan_amount_term],
        "Credit_History": [credit_history],
        "Property_Area": [property_area_map[property_area]]
    })

    # Apply log transformation
    features['ApplicantIncomelog'] = np.log(features['ApplicantIncome'] + 1)
    features['LoanAmountlog'] = np.log(features['LoanAmount'] + 1)
    features['Loan_Amount_Term_log'] = np.log(features['Loan_Amount_Term'] + 1)
    features['Total_Income_log'] = np.log(features['ApplicantIncome'] + features['CoapplicantIncome'] + 1)

    # Drop original features
    features = features.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)

    # Make predictions
    if st.button("Predict"):
        prediction = predict_loan_approval(features)
        if prediction == 1:
            st.success("Loan Approved!")
        else:
            st.error("Loan Not Approved")

elif selected == "Dataset":
    st.subheader("Loan Approval Dataset")
    try:
        data = pd.read_csv("LoanApprovalPrediction.csv")
        st.dataframe(data)
    except FileNotFoundError:
        st.error("Dataset not found.")
   
