import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("model.pkl")

# App title and description
st.title("KTM Ridership Prediction")
st.markdown("""
This app predicts **hourly ridership** at a selected KTM Komuter station based on:
- Whether it's a weekend
- Whether it's a peak hour
- Whether it's a public holiday
- The population near the station
""")

# Input features
is_weekend = st.selectbox("Is it a weekend?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
is_peak_hour = st.selectbox("Is it a peak hour?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
is_holiday = st.selectbox("Is it a public holiday?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
origin_encoded = st.number_input("Origin Station Code (encoded)", min_value=0, max_value=100, value=13)
destination_encoded = st.number_input("Destination Station Code (encoded)", min_value=0, max_value=100, value=7)

# Prediction button
if st.button("Predict Ridership"):
    # Prepare the feature vector
    input_data = np.array([[is_weekend, is_peak_hour, is_holiday, origin_encoded,destination_encoded]])
    
    # Scale the input
    # input_scaled = scaler.transform(input_data)

    # Predict using the model
    predicted_ridership = model.predict(input_scaled)

    # Display the result
    st.success(f"Estimated Hourly Ridership: **{int(predicted_ridership[0])} passengers**")
