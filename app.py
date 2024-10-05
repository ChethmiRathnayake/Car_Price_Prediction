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

# Collecting the input features interactively from the user with front-end validations
year = st.number_input('Manufactured Year', min_value=1980, max_value=2024, value=2024)
make = st.text_input('Brand', '')
model = st.text_input('Model', '')
trim = st.text_input('Additional Designation (e.g., LX, T5)', '')
body = st.selectbox('Body Type', ['Sedan', 'SUV', 'Truck', 'Coupe', 'Convertible', 'Wagon', 'Van'], index=0)
state = st.text_input('State (standard abbreviations used by the United States Postal Service)', '')

# Validation to ensure state code is exactly 2 letters
if state and (len(state) != 2 or not state.isalpha()):
    st.warning("Please enter a valid 2-letter state code.")

condition = st.slider('Condition (1-100)', min_value=1, max_value=100, value=50)
odometer = st.number_input('Odometer (miles)', min_value=0, value=0)

# Validation to ensure odometer has a reasonable maximum limit
if odometer > 500000:
    st.warning("Please enter an odometer value less than 500,000 miles.")

color = st.text_input('Color', '')
interior = st.text_input('Interior', '')
seller = st.text_input('Seller', '')

# Validate that required fields are filled in before allowing prediction
if st.button('Predict MMR'):
    if not make or not model or not state:
        st.warning("Please fill in all required fields (Brand, Model, State).")
    elif len(state) != 2 or not state.isalpha():
        st.warning("State must be a valid 2-letter code.")
    elif odometer > 500000:
        st.warning("Odometer should be less than 500,000 miles.")
    else:
        # Create the input dictionary (as you did in the notebook)
        input_data = {
            'year': [year],
            'make': [make],
            'model': [model],
            'trim': [trim],
            'body': [body],
            'state': [state],
            'condition': [condition],
            'odometer': [odometer],
            'color': [color],
            'interior': [interior],
            'seller': [seller]
        }
        
        # Convert the input into a DataFrame
        input_df = pd.DataFrame(input_data)
        
        # Make a prediction with the trained model
        predicted_price = xgb_model.predict(input_df)
        
        # Display the prediction
        st.success(f'Predicted Vehicle Price (MMR): ${predicted_price[0]:,.2f}')
