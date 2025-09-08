import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tensorflow.keras.models import load_model
import tensorflow as tf

#load the trained model, scaler pickele file and label encoder pickle file
model=load_model('annclassification/ann_model.h5')


#load the encoder and scaler
with open('annclassification/scaler.pkl','rb') as f:
    scaler=pickle.load(f)

with open('annclassification/lable_encoder_gender.pkl','rb') as f:
    lable_encoder_gender=pickle.load(f)

with open('annclassification/onehot_encoder_geography.pkl','rb') as f:
    onehot_encoder_geography=pickle.load(f)


## steamlit app
st.title('Customer Churn Prediction')

geography = st.selectbox('Geography',onehot_encoder_geography.categories_[0])
gender = st.selectbox('Gender',lable_encoder_gender.classes_)
age = st.slider('Age',18,100,25)
balance = st.number_input('Balance',0.0,250000.0,1000.0)
credit_score = st.slider('Credit Score',300,850,600)
estimated_salary = st.number_input('Estimated Salary',0.0,200000.0,5000.0)
tenure = st.slider('Tenure (in years)',0,10,1)
num_of_products = st.slider('Number of Products',1,4,1)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

#preparing the input data for prediction
#encoding the categorical features
geo_encoded = onehot_encoder_geography.transform([[geography]]).toarray()
geo_encoder_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geography.get_feature_names_out(['Geography']))
input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[lable_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary],
})


#combining the encoded geography columns with the input data
input_df = pd.concat([input_data.reset_index(drop=True),geo_encoder_df],axis=1)
# input_df = input_df.drop(['Geography'],axis=1)

input_data_scaled = scaler.transform(input_df) #scaling the input data

pred = model.predict(input_data_scaled) #making prediction
pred_probablity = pred[0][0]

if pred_probablity > 0.5:
    st.write(f'The customer is likely to churn with a probability of {pred_probablity:.2f}')
else:
    st.write(f'The customer is unlikely to churn with a probability of {pred_probablity:.2f}')
             