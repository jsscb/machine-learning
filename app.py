import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the machine learning model and encode
model = joblib.load('XGB_churn.pkl')
gender_encode = joblib.load('gender_encode.pkl')
oneHot_encode_geo = joblib.load('oneHot_encode_geo.pkl')

def main():
    st.title('Churn Model Deployment')

    # Add user input components for each feature
    credit_score = st.number_input('Credit Score', min_value=0, max_value=2000, value=700)
    age = st.number_input('Age', min_value=0, max_value=100, value=30)
    gender = st.radio('Gender', ('Male', 'Female'))
    tenure = st.number_input('Tenure', min_value=0, max_value=100, value=5)
    balance = st.number_input('Balance', min_value=0.0, max_value=1000000.0, value=50000.0)
    num_of_products = st.number_input('Number of Products', min_value=1, max_value=4, value=2)
    has_cr_card = st.radio('Has Credit Card', ('Yes', 'No'))
    is_active_member = st.radio('Is Active Member', ('Yes', 'No'))
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0, max_value=1000000.0, value=50000.0)
    geography = st.selectbox('Geography', ('France', 'Germany', 'Spain'))

    # Prepare input data
    data = {'CreditScore': credit_score, 'Age': age, 'Gender': gender, 'Tenure': tenure, 'Balance': balance,
            'NumOfProducts': num_of_products, 'HasCrCard': has_cr_card,
            'IsActiveMember': is_active_member, 'EstimatedSalary': estimated_salary,
            'Geography': geography}

    # Preprocess input data
    input_data = preprocess_input(data)

    if st.button('Make Prediction'):
        # Use the model to make predictions
        prediction = make_prediction(input_data)
        if prediction == 1:
            st.success('The customer is likely to churn.')
        else:
            st.success('The customer is unlikely to churn.')

def preprocess_input(data):
    # Convert categorical variables to numeric
    data['Gender'] = 1 if data['Gender'] == 'Male' else 0
    data['HasCrCard'] = 1 if data['HasCrCard'] == 'Yes' else 0
    data['IsActiveMember'] = 1 if data['IsActiveMember'] == 'Yes' else 0
    # Map 'Geography' to appropriate encoding
    if data['Geography'] == 'France':
        data['Geography_France'] = 1
        data['Geography_Germany'] = 0
        data['Geography_Spain'] = 0
    elif data['Geography'] == 'Germany':
        data['Geography_France'] = 0
        data['Geography_Germany'] = 1
        data['Geography_Spain'] = 0
    else:  # Spain
        data['Geography_France'] = 0
        data['Geography_Germany'] = 0
        data['Geography_Spain'] = 1
    # Drop original 'Geography' column
    data.pop('Geography')
    return pd.DataFrame([data])

def make_prediction(input_data):
    # Use the loaded model to make predictions
    input_array = np.array(input_data)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()
