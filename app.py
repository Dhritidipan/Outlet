import streamlit as st
import joblib
import pandas as pd

# Load the trained model
# Assuming you saved the model as 'logistic_regression_model.joblib'
model = joblib.load('logistic_regression_model.joblib')

st.title('Outlet Quality Prediction')

st.write("Enter the features to predict the outlet quality:")

# Define the expected features based on the training data (X_train)
# You should get this list from your training process, e.g., by saving X_train.columns
# For now, I'll use the columns from your original df after one-hot encoding, excluding the target
expected_features = ['SKU_count', 'Avg_Sales', 'Willingness_Organic', 'Willingness_Atta',
                     'Willingness_Ghee', 'Willingness_Oil', 'Willingness_Beverage',
                     'Willingness_Bakery', 'Importance_of_AMUL', 'Reputation_of_Store',
                     'Self_Service_No', 'Self_Service_Yes', 'Customer_Can_Browse_Full View and Choice',
                     'Customer_Can_Browse_No Entry',  'Customer_Can_Browse_Partial View','Shop_Type_Convenience store',
                     'Shop_Type_General trade', 'Shop_Type_Modern trade', 'Shelf_Space_20-30ft',
                     'Shelf_Space_30-40ft', 'Shelf_Space_40-50ft', 'Shelf_Space_<20ft',
                     'Shelf_Space_>50ft', 'Cold_Storage_No', 'Cold_Storage_Yes']


# Create input fields for the features
input_data = {}
for feature in expected_features:
    if feature in ['SKU_count']:
        input_data[feature] = st.number_input(feature, min_value=0)
    elif feature in ['Avg_Sales']:
        input_data[feature] = st.number_input(feature, min_value=0.0)
    elif feature in ['Willingness_Organic', 'Willingness_Atta', 'Willingness_Ghee',
                     'Willingness_Oil', 'Willingness_Beverage', 'Willingness_Bakery',
                     'Importance_of_AMUL', 'Reputation_of_Store']:
        input_data[feature] = st.slider(feature, 1, 5)
    elif feature in ['Self_Service_No', 'Self_Service_Yes', 'Customer_Can_Browse_Full View and Choice',
                     'Customer_Can_Browse_No Entry',  'Customer_Can_Browse_Partial View''Shop_Type_Convenience store',
                     'Shop_Type_General trade', 'Shop_Type_Modern trade', 'Shelf_Space_20-30ft',
                     'Shelf_Space_30-40ft', 'Shelf_Space_40-50ft', 'Shelf_Space_<20ft',
                     'Shelf_Space_>50ft', 'Cold_Storage_No', 'Cold_Storage_Yes']:
        # For dummy variables, you might want a different input method,
        # or handle the original categorical inputs and then one-hot encode them here.
        # For simplicity, let's use checkboxes for now, assuming True/False corresponds to 1/0
        input_data[feature] = st.checkbox(feature)

# Convert input data to a DataFrame
input_df = pd.DataFrame([input_data])

# Ensure the input DataFrame has the same columns as the training data and in the same order
# Fill in missing dummy columns with 0
for feature in expected_features:
    if feature not in input_df.columns:
        input_df[feature] = False # Assuming False corresponds to 0 for checkboxes

# Convert boolean columns to integers (0 or 1)
for feature in input_df.columns:
    if input_df[feature].dtype == 'bool':
        input_df[feature] = input_df[feature].astype(int)

# Reorder columns to match the training data
input_df = input_df[expected_features]


# Make a prediction
if st.button('Predict'):
    prediction = model.predict(input_df)
    st.success(f"Predicted Outlet Quality: {prediction[0]}")
