# Customer Churn Prediction Web App

Live Demo: https://customer-churn-prediction-app-main.streamlit.app/

This project is a Streamlit-based web application for predicting customer churn using an Artificial Neural Network (ANN) model. The app allows users to input customer details and receive a prediction on whether the customer is likely to churn.

## Features
- User-friendly interface built with Streamlit
- Input fields for customer demographics and account details
- Encodes categorical features using pre-trained encoders
- Scales input data using a pre-trained scaler
- Loads a trained ANN model for prediction
- Displays churn probability and prediction result

## How It Works
1. User enters customer information (geography, gender, age, balance, credit score, etc.)
2. The app encodes and scales the input data using saved encoders and scaler
3. The ANN model predicts the probability of churn
4. The result is displayed to the user

## Files
- `app.py`: Main Streamlit app
- `ann_model.h5`: Trained ANN model
- `scaler.pkl`, `lable_encoder_gender.pkl`, `onehot_encoder_geography.pkl`: Pre-trained scaler and encoders

## Getting Started
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run annclassification/app.py
   ```

## Requirements
- Python 3.7+
- Streamlit
- Pandas
- Numpy
- Scikit-learn
- TensorFlow

## Usage
- Enter customer details in the web interface
- View the churn prediction and probability

## License
This project is for educational purposes.
