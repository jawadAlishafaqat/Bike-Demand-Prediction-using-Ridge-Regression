import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Bike Demand AI",
    page_icon="🚲",
    layout="wide"
)

# ================= LOAD MODEL =================
model = joblib.load("ridge_model.pkl")
scaler = joblib.load("scaler.pkl")

# ================= CUSTOM CSS =================
st.markdown("""
<style>

.main {
    background-color: #0f1117;
}

h1, h2, h3 {
    color: white;
}

.metric-card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
}

.metric-value {
    font-size: 32px;
    color: #00ffcc;
    font-weight: bold;
}

.metric-label {
    color: #aaaaaa;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.title("🚲 Bike Demand Prediction Dashboard")
st.markdown(
    "### Predict hourly bike rental demand using Ridge Regression"
)

# ================= SIDEBAR =================
st.sidebar.title("⚙ Input Features")

season = st.sidebar.selectbox(
    "Season",
    [1,2,3,4]
)

yr = st.sidebar.selectbox(
    "Year",
    [0,1]
)

mnth = st.sidebar.slider(
    "Month",
    1,12,6
)

hr = st.sidebar.slider(
    "Hour",
    0,23,12
)

holiday = st.sidebar.selectbox(
    "Holiday",
    [0,1]
)

weekday = st.sidebar.slider(
    "Weekday",
    0,6,3
)

workingday = st.sidebar.selectbox(
    "Working Day",
    [0,1]
)

weathersit = st.sidebar.selectbox(
    "Weather Situation",
    [1,2,3,4]
)

temp = st.sidebar.slider(
    "Temperature",
    0.0,1.0,0.5
)

atemp = st.sidebar.slider(
    "Feels Like Temperature",
    0.0,1.0,0.5
)

hum = st.sidebar.slider(
    "Humidity",
    0.0,1.0,0.5
)

windspeed = st.sidebar.slider(
    "Windspeed",
    0.0,1.0,0.2
)

# ================= FEATURE ENGINEERING =================
rush_hour = 1 if (7 <= hr <= 9) or (17 <= hr <= 19) else 0

is_weekend = 1 if weekday in [0, 6] else 0

temp_humidity = temp * hum

# ================= DATA =================
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

# ================= METRIC CARDS =================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Predicted Rentals</div>
        <div class="metric-value">{int(prediction)}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Temperature</div>
        <div class="metric-value">{round(temp*100,1)}%</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Humidity</div>
        <div class="metric-value">{round(hum*100,1)}%</div>
    </div>
    """, unsafe_allow_html=True)

# ================= CHART SECTION =================
st.markdown("## 📊 Weather Analysis")

chart_data = pd.DataFrame({
    "Feature": ["Temperature", "Humidity", "Windspeed"],
    "Value": [temp, hum, windspeed]
})

fig, ax = plt.subplots(figsize=(7,4))

ax.bar(
    chart_data["Feature"],
    chart_data["Value"]
)

ax.set_title("Environmental Factors")

st.pyplot(fig)

# ================= INFO SECTION =================
st.markdown("## 🧠 Model Information")

st.info("""
This AI system predicts bike rental demand using Ridge Regression.
The model was trained on historical bike-sharing and weather data.
""")

# ================= FOOTER =================
st.markdown("---")
st.markdown(
    "Developed using Streamlit + Scikit-learn 🚀"
)
