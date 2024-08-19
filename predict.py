# Import Libraries
import numpy as np
import pandas as pd
import pickle
import os
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn import preprocessing
try:
    from category_encoders.target_encoder import TargetEncoder
except ModuleNotFoundError as e:
    print("ModuleNotFoundError:", e)
    import sys
    print("sys.path:", sys.path)
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

# Function to calculate sentiment score
def calculate_sentiment(remark):

    # Initialize VADER
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(remark)
    compound_score = scores['compound']
    return compound_score

def preprocess(input_data):
    '''Parameter input_data is in DataFrame format'''

    # Ensure date columns are in datetime format
    date_columns = ['issue_responded', 'issue_reported_at']
    for col in date_columns:
        input_data[col] = pd.to_datetime(input_data[col])
    
    # If customer review is null, fill with 'Good'
    input_data['customer_review'] = input_data['customer_review'].fillna('Good')
    
    # Calculate sentiment score (assuming a function calculate_sentiment exists)
    input_data['sentiment'] = input_data['customer_review'].apply(calculate_sentiment)  # Replace with actual sentiment calculation
    
    # Calculate wait_response_time
    input_data['wait_response_time'] = (input_data['issue_responded'] - input_data['issue_reported_at']).dt.total_seconds() / 60

    # Impute negative values with 60 minutes
    input_data['wait_response_time'] = input_data['wait_response_time'].apply(lambda x: 60 if x < 0 else x)

    input_data = input_data.drop(columns=['customer_review', 'issue_responded', 'issue_reported_at'])

    # Ensure categorical columns are consistent with those used during training
    categorical_columns = ['channel_name', 'category', 'tenure_bucket', 'agent_shift']
    
    # Preprocess for one-hot encoding
    with open('pkl_files/one_hot_encoder.pkl', 'rb') as f:
        one_hot_encoder = pickle.load(f)
    
    # Filter out any new categorical values that weren't in training
    input_data[categorical_columns] = input_data[categorical_columns].fillna('Unknown')
    
    input_data_one_hot = one_hot_encoder.transform(input_data[categorical_columns])
    
    # Preprocess for target encoding
    with open('pkl_files/target_encoders.pkl', 'rb') as f:
        target_encoders = pickle.load(f)
    
    for col in ['sub-category', 'agent_name', 'supervisor', 'manager']:
        input_data[col] = target_encoders[col].transform(input_data[col])
    
    # Drop original categorical columns and concatenate with one-hot encoded data
    input_data = input_data.drop(columns=categorical_columns)
    input_data = pd.concat([input_data, pd.DataFrame(input_data_one_hot, columns=one_hot_encoder.get_feature_names_out(categorical_columns))], axis=1)
    
    # Load and apply scaler_X
    with open('pkl_files/scaler_X.pkl', 'rb') as f:
        scaler_X = pickle.load(f)
    
    input_data = scaler_X.transform(input_data)
    
    return input_data

### Predict Function

def real_time_predict(input_data):
    # Preprocess the input data
    input_data = preprocess(input_data)
    
    # Load the best model
    best_model = load_model('models_saved/best_model.h5')
    
    # Predict with the best model
    y_pred = best_model.predict(input_data)
    
    # Reverse scale y_pred value to get the actual CSAT score
    min_y, max_y = 1, 5  
    csat_score = (y_pred * (max_y - min_y)) + min_y
    csat_score = csat_score.round()  # Round to the nearest integer if needed
    
    return csat_score