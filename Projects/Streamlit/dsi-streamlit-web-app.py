# title - what this app actually is 

# how to use
# 1. Age 
# 2. Gender 
# 3. credit score 
# Submit button 
import streamlit as st
import pandas as pd 
import joblib

# load our model pipeline object 
# model = joblib.load('C:/Users/Emanuel/Documents/GitHub/MachineLearningWork/Projects/Streamlit/model.joblib')
model = joblib.load('model.joblib')


# add title and instructions 
st.title('Purchase Prediction Model')
st.subheader('Enter customer information and submit for likelihood to purchase')

# age input form 

age = st.number_input(
    label="01. Enter the customer's age",
    min_value= 18,
    max_value=120,
    value = 35  
    )

# gender input form
gender = st.radio(
    label= "02. Enter the customer's gender",
    options= ['M','F']
)
# credit score input form 
credit_score = st.number_input(
    label="03. Enter the customer's credit score",
    min_value= 0,
    max_value=1000,
    value = 500 
    )

# Submit to model
if st.button("Submit For Prediction"):
    # store our data in a data frame for prediction 
    new_data = pd.DataFrame({"age": [age],
                            "gender":[gender],
                            "credit_score":[credit_score]})
    # apply model pipe line to the input data and extract probability prediction
    pred_proba = model.predict_proba(new_data)[0][1]

    # output prediction 
    st.subheader(f"Based on these customer's attributes our model predicts a purchase probability of {pred_proba:.0%}")
