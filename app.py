import streamlit as st
import pickle
import pandas as pd

# Load the trained XGBoost model (or pipeline)
with open('xgb_model.pkl', 'rb') as file:
    xgb_model = pickle.load(file)

# Streamlit app title
st.title('Car Price Prediction App')

# Input fields for user to provide the vehicle's features
st.header('Enter the vehicle details:')

# Collecting the input features interactively from the user
manufacture_year = st.number_input('Manufacture Year', min_value=1980, max_value=2024, value=2024)
brand = st.text_input('Brand', '')
specific_model = st.text_input('Model', '')
additional_designation = st.text_input('Additional Designation', '')
body = st.selectbox('Body Type', ['Sedan', 'SUV', 'Truck', 'Coupe', 'Convertible', 'Wagon', 'Van'], index=0)
condition = st.slider('Condition (1-100)', min_value=1, max_value=100, value=50)
odometer = st.number_input('Odometer (miles)', min_value=0)
seller = st.text_input('Seller', '')

# Add a predict button
if st.button('Predict MMR'):
    # Create the input dictionary (as you did in the notebook)
    input_data = {
        'manufacture_year': [manufacture_year],
        'brand': [brand],
        'specific_model': [specific_model],
        'additional_designation': [additional_designation],
        'body': [body],
        'condition': [condition],
        'odometer': [odometer],
        'seller': [seller]
    }
    
    # Convert the input into a DataFrame
    input_df = pd.DataFrame(input_data)
    
    # Make a prediction with the trained model
    predicted_price = xgb_model.predict(input_df)
    
    # Display the prediction
    st.success(f'Predicted Vehicle Price (MMR): ${predicted_price[0]:,.2f}')
