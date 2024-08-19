# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
from predict import *  # Import your preprocessing and prediction functions
import sys

# Define the Streamlit app
def main():
    # Set title and description
    st.title('CSAT Score Prediction App')
    st.write('This app predicts CSAT scores based on input data.')

    # Sample DataFrame for dropdown options (replace with your actual data loading method)
    data = pd.read_csv('data/eCommerce_Customer_support_data.csv')
    data.columns = data.columns.str.lower().str.replace(' ', '_')
    # Create input widgets for categorical columns
    st.subheader('Select Input Data')
    channel_name = st.selectbox('Channel Name', data['channel_name'].unique())
    category = st.selectbox('Category', data['category'].unique())
    sub_category = st.selectbox('Sub-category', data['sub-category'].unique())
    agent_name = st.selectbox('Agent Name', data['agent_name'].unique())
    supervisor = st.selectbox('Supervisor', data['supervisor'].unique())
    manager = st.selectbox('Manager', data['manager'].unique())
    tenure_bucket = st.selectbox('Tenure Bucket', data['tenure_bucket'].unique())
    agent_shift = st.selectbox('Agent Shift', data['agent_shift'].unique())
    # Date and time pickers for issue_responded
    responded_date = st.date_input('Responded Date')
    responded_time = st.time_input('Responded Time')
    
    # Date and time pickers for issue_reported_at
    reported_date = st.date_input('Reported Date')
    reported_time = st.time_input('Reported Time')

    # Input fields for calculated columns
    customer_review = st.text_area('Customer Review', 'Type here...')

    # Convert user inputs into DataFrame for processing
    input_data = pd.DataFrame({
        'channel_name': [channel_name],
        'category': [category],
        'sub-category': [sub_category],
        'agent_name': [agent_name],
        'supervisor': [supervisor],
        'manager': [manager],
        'tenure_bucket': [tenure_bucket],
        'agent_shift': [agent_shift],
        'customer_review': [customer_review],
        'issue_responded': [pd.to_datetime(str(responded_date) + ' ' + str(responded_time))],
        'issue_reported_at': [pd.to_datetime(str(reported_date) + ' ' + str(reported_time))]
    })

    # Predict CSAT score
    if st.button('Predict CSAT Score'):
        csat_score = real_time_predict(input_data)

        # Display predicted CSAT score
        st.write(f"Predicted CSAT score = {csat_score[0][0]}")

if __name__ == '__main__':
    main()
