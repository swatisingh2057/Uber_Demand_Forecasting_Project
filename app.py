import streamlit as st
import gdown
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler

@st.cache_resource
def download_project_files():
    # Replace this with your folder's actual shareable link
    folder_url = "https://drive.google.com/drive/folders/1PlaoigO4dysH_4pRe_Y1Tr2b3VVa69qj?usp=sharing"
    
    # Download the whole folder to a local directory called 'data'
    gdown.download_folder(folder_url, output="data", quiet=False)
    
    # Now your files are in the 'data' folder
    return "data"

# Call the function
data_dir = download_project_files()



# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Uber Demand Dashboard",
    layout="wide"
)


st.title("Uber Demand Forecasting Dashboard")


# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("uber_data.csv")
    data['Date/Time'] = pd.to_datetime(data['Date/Time'])
    return data


data = load_data()


# -------------------------------
# LOAD MODELS
# -------------------------------
df = pd.read_csv(os.path.join(data_dir, "uber_data.csv"))
rf_model = joblib.load(os.path.join(data_dir, "rf_model.pkl"))
lstm_model = load_model(
    os.path.join(data_dir, "lstm_model.h5"), 
    custom_objects={"mse": MeanSquaredError()}
)


# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
data['hour'] = data['Date/Time'].dt.hour
data['day'] = data['Date/Time'].dt.day
data['month'] = data['Date/Time'].dt.month
data['weekday'] = data['Date/Time'].dt.weekday

data['is_weekend'] = (data['weekday'] >= 5).astype(int)


# -------------------------------
# KPI CARDS
# -------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Trips Recorded", f"{len(data):,}")
col2.metric("Peak Demand Hour", f"{data['hour'].mode()[0]}:00")
busiest_day = data['Date/Time'].dt.day_name().mode()[0]
col3.metric("Busiest Day of Week", busiest_day)


# -------------------------------
# FILTER
# -------------------------------

month_map = {
    4:"April",
    5:"May",
    6:"June",
    7:"July",
    8:"August",
    9:"September"
}

unique_months = sorted(data['month'].unique())

month_labels = [month_map[m] for m in unique_months]

col1, col2 = st.columns([1,3])
with col1:

    selected_month_name = st.selectbox(
    "Select Month for Analysis:",
    month_labels
    )

selected_month = [m for m, name in month_map.items() if name == selected_month_name][0]

filtered_data = data[
    data['month'] == selected_month
]


# -------------------------------
# PLOTLY GRAPHS
# -------------------------------
st.subheader("Hourly Trip Distribution")
st.caption("Shows the average volume of trips throughout a standard 24-hour day in the selected month.")

hour_data = (
    filtered_data
    .groupby('hour')
    .size()
    .reset_index(name='Trips')
)

fig1 = px.line(
    hour_data,
    x='hour',
    y='Trips',
    markers=True
)

st.plotly_chart(fig1, use_container_width=True)


# Heatmap
st.subheader("Demand Density Heatmap")
st.caption("Identifies peak activity patterns by overlaying day of the week against the hour of the day of the selected month.")

heat = (
    filtered_data
    .groupby(['day','hour'])
    .size()
    .reset_index(name='Trips')
)

fig2 = px.density_heatmap(
    heat,
    x='hour',
    y='day',
    z='Trips'
)

st.plotly_chart(fig2, use_container_width=True)


# -------------------------------
# ML PREDICTION
# -------------------------------
st.subheader("Predict Trip Demand")
st.caption("Adjust the parameters below to estimate trip volume using our trained Random Forest model.")

with st.container():
    col1, col2, col3 = st.columns(3)

    hour = col1.slider("Hour", 0, 23, 12)
    day = col2.slider("Day", 1, 31, 15)
    month = col3.slider("Month", 4, 9, 6)

weekday = day % 7
is_weekend = 1 if weekday >= 5 else 0

input_data = pd.DataFrame([[
    hour, day, month, weekday, is_weekend
]], columns=[
    'hour','day','month','weekday','is_weekend'
])

prediction = rf_model.predict(input_data)[0]

st.success(f"Estimated Trips: **{int(prediction)} trips**")


# -------------------------------
# LSTM FORECAST
# -------------------------------
st.subheader("Demand Forecast")

# Prepare hourly series
ts = (
    data
    .set_index('Date/Time')
    .resample('h')
    .size()
)

ts = ts.to_frame(name="Trips")


# Scale
scaler = MinMaxScaler()
scaled = scaler.fit_transform(ts)


# Last 168 hours
seq_length = 168

last_seq = scaled[-seq_length:]
last_seq = last_seq.reshape(1, seq_length, 1)


# Predict next 24 hours
future_preds = []

current_seq = last_seq.copy()

for _ in range(24):
    pred = lstm_model.predict(current_seq, verbose=0)[0][0]
    future_preds.append(pred)

    current_seq = np.append(
        current_seq[:,1:,:],
        [[[pred]]],
        axis=1
    )


# Inverse scale
future_preds = scaler.inverse_transform(
    np.array(future_preds).reshape(-1,1)
)


# Create future dataframe
future_df = pd.DataFrame({
    "Hour": range(1,25),
    "Predicted Trips": future_preds.flatten()
})

last_date = data['Date/Time'].max()
forecast_start = last_date + timedelta(hours=1)
st.caption(f"LSTM Demand Forecast starting from {forecast_start.strftime('%Y-%m-%d %H:00')}")


fig3 = px.line(
    future_df,
    x="Hour",
    y="Predicted Trips",
    markers=True,
    title="Next 24 Hours Forecast"
)

st.plotly_chart(fig3, use_container_width=True)