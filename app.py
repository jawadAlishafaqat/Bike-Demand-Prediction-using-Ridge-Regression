import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

model = joblib.load("ridge_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Bike Demand Predictor", layout="wide")

st.title("🚲 Bike Demand Prediction Dashboard")
st.markdown("Predict bike rental demand using Ridge Regression")

# ================= SIDEBAR =================
st.sidebar.header("Enter Input Features")

season = st.sidebar.slider("Season", 1, 4, 1)
yr = st.sidebar.slider("Year", 0, 1, 1)
mnth = st.sidebar.slider("Month", 1, 12, 6)
hr = st.sidebar.slider("Hour", 0, 23, 12)

holiday = st.sidebar.slider("Holiday", 0, 1, 0)
weekday = st.sidebar.slider("Weekday", 0, 6, 3)
workingday = st.sidebar.slider("Working Day", 0, 1, 1)

weathersit = st.sidebar.slider("Weather Situation", 1, 4, 1)

temp = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)
atemp = st.sidebar.slider("Feels Temperature", 0.0, 1.0, 0.5)
hum = st.sidebar.slider("Humidity", 0.0, 1.0, 0.5)
windspeed = st.sidebar.slider("Windspeed", 0.0, 1.0, 0.2)

rush_hour = 1 if (7 <= hr <= 9) or (17 <= hr <= 19) else 0
is_weekend = 1 if weekday in [0, 6] else 0
temp_humidity = temp * hum

features = np.array([[
    season,
    yr,
    mnth,
    hr,
    holiday,
    weekday,
    workingday,
    weathersit,
    temp,
    atemp,
    hum,
    windspeed,
    rush_hour,
    is_weekend,
    temp_humidity
]])

scaled_features = scaler.transform(features)

prediction = model.predict(scaled_features)[0]
prediction = max(prediction, 0)

# ================= METRICS =================
col1, col2, col3 = st.columns(3)

col1.metric("🕒 Hour", hr)
col2.metric("🌡 Temperature", round(temp,2))
col3.metric("🚲 Predicted Demand", int(prediction))

st.success(f"Estimated Bike Rentals: {int(prediction)}")

# ================= CHART =================
st.subheader("📊 Demand Insights")

chart_data = pd.DataFrame({
    "Feature": ["Temperature", "Humidity", "Windspeed"],
    "Value": [temp, hum, windspeed]
})

fig, ax = plt.subplots()

ax.bar(chart_data["Feature"], chart_data["Value"])

st.pyplot(fig)
