import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 载入模型与Scaler
model = joblib.load("random_forest_model.pkl")
# scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Komuter Ridership Predictor", layout="centered")

st.title("🚉 Komuter Hourly Ridership Predictor")
st.markdown("This app predicts the hourly ridership of a Komuter station based on input features.")

# 用户输入表单
with st.form("prediction_form"):
    hour = st.slider("Hour (0-23)", 0, 23, 8)
    day = st.slider("Day (1-31)", 1, 31, 15)
    weekday = st.selectbox("Weekday (0=Monday, 6=Sunday)", list(range(7)))
    origin_encoded = st.number_input("Origin Station Encoded", min_value=0, max_value=50, value=1)
    destination_encoded = st.number_input("Destination Station Encoded", min_value=0, max_value=50, value=3)
    is_peak_hour = st.selectbox("Is Peak Hour?", [0, 1])
    is_weekend = st.selectbox("Is Weekend?", [0, 1])
    is_busy_station = st.selectbox("Is Busy Station?", [0, 1])

    submitted = st.form_submit_button("Predict Ridership")

if submitted:
    input_data = pd.DataFrame([[
        hour, day, weekday, origin_encoded, destination_encoded,
        is_peak_hour, is_weekend, is_busy_station,
        hour*2, hour*3,
        hour * is_peak_hour,
        is_weekend * is_peak_hour,
        is_peak_hour * weekday,
        is_weekend * is_busy_station
    ]], columns=[
        'hour', 'day', 'weekday', 'origin_encoded', 'destination_encoded',
        'is_peak_hour', 'is_weekend', 'is_busy_station',
        'hour_squared', 'hour_cubed',
        'peak_hour_interaction', 'weekend_peak',
        'peak_weekday_interaction', 'weekend_busy_interaction'
    ])

    # 标准化（如适用）
    # input_scaled = scaler.transform(input_data)

    # 预测
    prediction = model.predict(input_data)

    st.success(f"📈 Predicted Ridership: {int(prediction[0])} passengers")
