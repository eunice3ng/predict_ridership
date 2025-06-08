import streamlit as st
import numpy as np
import pandas as pd
import joblib
import datetime
import gdown
import os

# Google Drive file IDs
MODEL_FILE_ID = "14FqiZG16TxzsUEsj9QCH5xy2FTbjZ3_y"
ENCODERS_FILE_ID = "1QhYvlNFxh6RBkJJ6Jf5KuUjY2_i-Yh-x"

# Download model files if they don't exist
if not os.path.exists("best_model.pkl"):
    gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", "best_model.pkl", quiet=False)

if not os.path.exists("label_encoders.json"):
    gdown.download(f"https://drive.google.com/uc?id={ENCODERS_FILE_ID}", "label_encoders.json", quiet=False)

# Load model
model = joblib.load("best_model.pkl")

with open("label_encoders.json", "r") as f:
    label_encoder_data = json.load(f)

label_encoders = {}
for key, classes in label_encoder_data.items():
    le = LabelEncoder()
    le.classes_ = np.array(classes)
    label_encoders[key] = le


# label_encoders = joblib.load("label_encoders.pkl")

# Set page
st.set_page_config(page_title="Komuter Ridership Predictor", layout="centered")
st.title("🚆 Komuter Ridership Prediction App")
st.markdown("Predict hourly Komuter ridership based on date, time, and trip details.")

# Form input
with st.form("prediction_form"):
    date_input = st.date_input("Select Date", datetime.date.today())
    time_input = st.time_input("Select Time", datetime.time(8, 0))

    origin_name = st.selectbox("Origin Station", label_encoders['origin'].classes_.tolist())
    destination_name = st.selectbox("Destination Station", label_encoders['destination'].classes_.tolist())

    submit = st.form_submit_button("Predict Ridership")

if submit:
    # Combine date and time
    datetime_combined = datetime.datetime.combine(date_input, time_input)
    hour = datetime_combined.hour
    day_of_week = datetime_combined.weekday()  # Monday=0, Sunday=6
    month = datetime_combined.month

    # Derived features
    is_weekend = 1 if day_of_week >= 5 else 0
    is_peak_hour = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
    malaysia_holidays = pd.Series(pd.date_range("2025-01-01", "2025-12-31")).map(lambda d: d in pd.Series(pd.date_range("2025-01-01", "2025-12-31")))  # Replace with actual holiday check
    is_holiday = 1 if date_input in malaysia_holidays else 0  # Placeholder

    # Encode stations
    origin_encoded = label_encoders['origin'].transform([origin_name])[0]
    destination_encoded = label_encoders['destination'].transform([destination_name])[0]

    # Prepare feature array
    features = np.array([[hour, is_peak_hour, is_weekend, is_holiday, origin_encoded, destination_encoded]])
    
    # Removed scaler transformation because scaler is no longer used
    # features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features)[0]

    # Output
    st.subheader("Prediction Result")
    st.info(f"Estimated Ridership: *{int(prediction):,} ridership(s)*")
