# Incident-Impact-Prediction
The objective is to predict the impact of the incidents/complaints raised by the customers from a service desk platform of an IT company to improve their service. A data set which contains the event log of an incident management process extracted from the service desk platform of an IT company was used to carry out this project.

Acceptance criteria:
To build the best model which gives the maximum performance, and need to deploy the model with suitable platform.

The following steps are carried out:

1. Importing the data, necessary libraries, & exploring the data to look for missing values.
2. Selecting the features for analysis, label encoding the ordinal column and splitting the data into test & train.
3. Training the data using algorithms like Support Vector Machine, Decision Tree, Random Forest, K-Nearest Neighbor, XGBoost Classifier and Artificial Neural Network and    checking the accuracy to find out which algorithm is the best.
4. Exporting the model with highest accuracy.  
5.Model is Deployed on Heroku Platform with Streamlit Framework.
  Deployment Link: https://impact-prediction-of-incident.herokuapp.com/


Results:
For predicting Ticket Priority, Random Forest gives almost 98.09% accuracy.  
