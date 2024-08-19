# **Customer Satisfaction (CSAT) Prediction Model**
## Overview
This project focuses on predicting Customer Satisfaction (CSAT) scores using a Deep Learning Artificial Neural Network (ANN). The model is developed in the context of an e-commerce platform, aiming to understand and improve customer satisfaction through their interactions and feedback. By leveraging neural network model, we aim to accurately forecast CSAT scores based on various features, providing actionable insights for service improvement.

## Project Background
Customer satisfaction in e-commerce is a critical metric influencing customer loyalty, repeat business, and word-of-mouth marketing. Traditionally, companies have relied on direct surveys to gauge customer satisfaction, which can be time-consuming and may not always capture the full spectrum of customer experiences. With the advent of deep learning, it's now possible to predict customer satisfaction scores in real-time, offering a granular view of service performance and identifying areas for immediate improvement.

## Dataset Overview
The dataset encompasses customer satisfaction scores over a one-month period on an e-commerce platform named "Shopzilla." It consists of the following features:

- **Unique id:** Unique identifier for each record (integer).
- **Channel name:** Name of the customer service channel (object/string).
- **Category:** Category of the interaction (object/string).
- **Sub-category:** Sub-category of the interaction (object/string).
- **Customer Remarks:** Feedback provided by the customer (object/string).
- **Order id:** Identifier for the order associated with the interaction (integer).
- **Order date time:** Date and time of the order (datetime).
- **Issue reported at:** Timestamp when the issue was reported (datetime).
- **Issue responded:** Timestamp when the issue was responded to (datetime).
- **Survey response date:** Date of the customer survey response (datetime).
- **Customer city:** City of the customer (object/string).
- **Product category:** Category of the product (object/string).
- **Item price:** Price of the item (float).
- **Connected handling time:** Time taken to handle the interaction (float).
- **Agent name:** Name of the customer service agent (object/string).
- **Supervisor:** Name of the supervisor (object/string).
- **Manager:** Name of the manager (object/string).
- **Tenure Bucket:** Bucket categorizing agent tenure (object/string).
- **Agent Shift:** Shift timing of the agent (object/string).
- **CSAT Score:** Customer Satisfaction (CSAT) score (integer).


## 1. Understanding and Data Preprocessing

The objective of this project is to develop a predictive model for Customer Satisfaction (CSAT) scores, enabling the business to anticipate customer satisfaction levels based on various features derived from customer interactions. The key steps in understanding and preparing the data are as follows:

### Data Preparation:
- **Data Collection and Cleaning:**  
  The dataset was first cleaned to remove any inconsistencies, missing values, and irrelevant data. Key date columns were converted to the appropriate datetime format to facilitate further analysis.

- **Feature Engineering:**  
  Important features were engineered from the raw data, such as wait response time, which was calculated by measuring the time difference between when an issue was reported and when it was responded to. Any negative values were imputed with a standard time (e.g., 60 minutes) to ensure consistency.

- **Sentiment Analysis:**  
  Sentiment scores were derived from customer reviews, providing additional insights into the customer’s experience. Reviews with missing values were filled with the term "Good" to maintain data integrity.

---

## 2. EDA INSIGHTS

### 1. CSAT Rating Distribution:

A significant 70% of customer calls receive a perfect rating of 5, indicating a strong overall satisfaction with customer support. However, 13% of calls are rated at 4, which is still positive but suggests room for improvement. Notably, another 13% of calls are rated as 1, reflecting severe dissatisfaction. To elevate customer support from "Excellent" to "Outstanding," it is crucial to minimize the number of calls receiving a rating of 1. Focused efforts on identifying and addressing the root causes of these low ratings will be essential.

### 2. CSAT Score Distribution by Category and Sub-Category:

Analyzing CSAT scores across different categories and sub-categories reveals that certain areas are prone to lower ratings. Categories like commission-related issues, account-related problems, and call disconnections have a concerning percentage of ratings at 1. In some sub-categories, more than 30% of interactions are rated as 1. On the other hand, sub-categories such as billing & payment, instant discount, non-order related issues, and app-related problems consistently receive ratings of 5, with over 80% of customers expressing high satisfaction. This highlights the need to target specific areas with tailored interventions to reduce dissatisfaction while continuing to reinforce the strengths of high-performing categories.

### 3. Customer Remarks and CSAT Scores:

Customer remarks are provided in about 50% of interactions, predominantly when customers are either highly satisfied or highly disappointed. This pattern suggests that remarks are a strong indicator of customer sentiment. Given that 'good' is the most common remark, it is recommended to impute 'good' in cases where no remark is provided. This approach will maintain data consistency and align with the observed mode in customer feedback.

### 4. Trend in CSAT Scores:

There is an observable upward trend in CSAT scores, indicating an overall improvement in customer satisfaction over time. This positive trajectory should be monitored closely, and efforts should continue to maintain and accelerate this growth, particularly by addressing areas of persistent dissatisfaction.

### 5. Manager Performance:

The analysis shows that all managers maintain a similar level of CSAT scores, suggesting a consistent performance across leadership. While this consistency is positive, it also indicates that there may be opportunities for innovation and differentiation in management practices to further enhance customer satisfaction.

### 6. Focus on Low-Performing Agents:

There are 20 agents with a median CSAT score of less than 3. This group represents a critical area for improvement. Implementing targeted training programs and conducting detailed performance reviews could help these agents address specific challenges and elevate their performance. Addressing the issues faced by these agents will be key to improving overall customer satisfaction.

### **Recommendations:**

1. **Targeted Interventions:** Focus on the categories and sub-categories with high percentages of low ratings (1s) to identify and resolve underlying issues.
2. **Training and Development:** Provide additional support and training to the 20 agents with a median CSAT score of less than 3 to improve their performance.
3. **Proactive Customer Engagement:** Leverage the upward trend in CSAT scores to explore new ways of exceeding customer expectations, especially in high-performing areas like billing and app-related issues.
4. **Continuous Monitoring:** Maintain a consistent review of CSAT scores across categories, sub-categories, and agents to ensure that improvements are sustained and areas of dissatisfaction are promptly addressed.

---

## 3. Data Transformations

To ensure that the model could effectively learn from the data, several transformations were applied:

- **Scaling:**  
  Both features (`X`) and the target variable (`y`) were scaled using MinMaxScaler to normalize the data within a consistent range. This helped in stabilizing the training process and improving model performance.

- **One-Hot Encoding:**  
  Categorical variables such as channel name, category, tenure bucket, and agent shift were encoded using one-hot encoding to convert them into a machine-readable format.

- **Target Encoding:**  
  Columns like sub-category, agent name, supervisor, and manager were target-encoded to capture the relationship between these categories and the CSAT scores.

- **Preprocessing Pipelines:**  
  The preprocessing steps were consolidated into a pipeline to streamline the transformation process for both training and test datasets, ensuring that the same procedures were applied consistently.

---

## 4. Modelling & Evaluations

A deep learning model was developed to predict CSAT scores based on the prepared dataset. The model architecture and training process are outlined below:

### Model Architecture:
- **Neural Network Design:**  
  The model was designed with two hidden layers, each consisting of ReLU activation functions and dropout for regularization. A linear activation function was used in the output layer to predict the continuous CSAT score.

- **Optimization:**  
  The Adam optimizer was used with a learning rate that was fine-tuned through hyperparameter tuning. Early stopping was implemented to prevent overfitting, ensuring that the model retained the best weights during training.

### Hyperparameter Tuning:
- **Randomized Search:**
  A RandomizedSearchCV approach was employed to optimize key hyperparameters, including the number of neurons, dropout rate, batch size, epochs, and learning rate. This approach helped in exploring a wide range of hyperparameters efficiently.

- **Keras Tuner:**
  Additionally, Keras Tuner was utilized to further refine the hyperparameter optimization process. Keras Tuner provides advanced search algorithms like RandomSearch, Hyperband, and BayesianOptimization to systematically explore hyperparameter space. By integrating Keras Tuner, the process identified the optimal combination of parameters that resulted in the highest model accuracy. This tool enabled fine-grained control over the tuning process and facilitated the discovery of improved hyperparameter configurations beyond initial random search.

### Model Evaluation:  
  The model’s performance was evaluated using metrics like mean squared error and mean absolute error. The distribution of the differences between actual and predicted CSAT scores was analyzed to understand the model's accuracy and identify areas for improvement.

---

## 5. Conclusion & Action Points

The CSAT prediction model provides valuable insights into customer satisfaction, enabling the business to make data-driven decisions to enhance customer service quality. Key findings and actionable recommendations are summarized below:

### Summary

The model effectively predicts CSAT scores by leveraging features such as response times and customer sentiment. It has identified critical drivers of customer satisfaction and highlighted areas where service improvements could have a significant impact. While the model performs well overall, further refinements can improve its accuracy, particularly in cases where prediction errors are larger.

### Action Points

1. **Optimize Customer Service Operations:**
   - **Reduce Response Times:** Implement strategies to minimize customer wait times, as this has been shown to significantly impact satisfaction. Consider increasing staff during peak periods or leveraging automation to improve response efficiency.
   - **Monitor and Improve Customer Sentiment:** Use the insights from sentiment analysis to proactively address negative feedback and reinforce positive experiences. This could involve targeted training for customer service representatives or adjusting service protocols.

2. **Enhance Data Quality and Model Accuracy:**
   - **Refine Feature Engineering:** Continuously improve the model by exploring additional features that may contribute to more accurate predictions. Regularly update the dataset to reflect the most recent customer interactions and trends.
   - **Address Prediction Outliers:** Investigate cases where the model's predictions deviate significantly from actual scores to identify potential service gaps or areas for process improvement.

3. **Deploy Model for Proactive Customer Management:**
   - **Real-Time Monitoring:** Integrate the model into customer service operations to provide real-time predictions that can guide decision-making and improve customer satisfaction proactively.
   - **Strategic Use of Insights:** Leverage the model’s predictions to inform broader business strategies, such as resource allocation, training programs, and service enhancements, to drive continuous improvement in customer satisfaction.

By implementing these recommendations, the business can leverage the predictive model to enhance customer satisfaction, improve service quality, and strengthen overall customer loyalty and retention.


