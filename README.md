# **Customer Satisfaction (CSAT) Prediction Model**
## Overview
This project focuses on predicting Customer Satisfaction (CSAT) scores using a Deep Learning Artificial Neural Network (ANN). The model is developed in the context of an e-commerce platform, aiming to understand and improve customer satisfaction through their interactions and feedback. By leveraging neural network model, we aim to accurately forecast CSAT scores based on various features, providing actionable insights for service improvement.

## Project Background
Customer satisfaction in e-commerce is a critical metric influencing customer loyalty, repeat business, and word-of-mouth marketing. Traditionally, companies have relied on direct surveys to gauge customer satisfaction, which can be time-consuming and may not always capture the full spectrum of customer experiences. With the advent of deep learning, it's now possible to predict customer satisfaction scores in real-time, offering a granular view of service performance and identifying areas for immediate improvement.

## Dataset Overview
The dataset encompasses customer satisfaction scores over a one-month period on an e-commerce platform named "Shopzilla." It consists of the following features:

Unique id: Unique identifier for each record (integer).
Channel name: Name of the customer service channel (object/string).
Category: Category of the interaction (object/string).
Sub-category: Sub-category of the interaction (object/string).
Customer Remarks: Feedback provided by the customer (object/string).
Order id: Identifier for the order associated with the interaction (integer).
Order date time: Date and time of the order (datetime).
Issue reported at: Timestamp when the issue was reported (datetime).
Issue responded: Timestamp when the issue was responded to (datetime).
Survey response date: Date of the customer survey response (datetime).
Customer city: City of the customer (object/string).
Product category: Category of the product (object/string).
Item price: Price of the item (float).
Connected handling time: Time taken to handle the interaction (float).
Agent name: Name of the customer service agent (object/string).
Supervisor: Name of the supervisor (object/string).
Manager: Name of the manager (object/string).
Tenure Bucket: Bucket categorizing agent tenure (object/string).
Agent Shift: Shift timing of the agent (object/string).
CSAT Score: Customer Satisfaction (CSAT) score (integer).

## **Steps Involved**
**1. Data Preparation**
Cleaning: Handle missing values, remove duplicates, and address inconsistencies in the dataset.
Encoding: Convert categorical variables into numerical values using techniques like one-hot encoding and target encoding.
**2. Feature Engineering & Data Transformation**
Interaction Timing: Calculate the time differences between various timestamps to create new features related to interaction speed.
Text Analysis: Perform sentiment analysis on customer remarks to extract sentiment scores as features.
Normalization: Normalize numerical features to ensure uniformity in the input data.
**3. Model Development & Evaluation**
Initial implementation :
![image](https://github.com/user-attachments/assets/1f732706-e37f-4e77-b0be-ae4ab47495b6)

Hyperparameter tuning and selecting the best model :
![image](https://github.com/user-attachments/assets/a7013ad0-1f6b-49cf-b668-54266b2c634c)

**4. Prediction**
Using best model, resulted in accuracy of 72%

**5.Insights**

**6.Conclusion**

