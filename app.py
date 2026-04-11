import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime

# -------------------------
# 1. Page Setup
# -------------------------
st.set_page_config(page_title="NSE Stock Prediction", layout="centered")
st.title("📈 NSE Nifty Stock Prediction")
st.write("Predict next day's closing price using LSTM")

# -------------------------
# 2. Nifty 50 List
# -------------------------
nifty_50 = {
    'RELIANCE.NS': 'Reliance Industries',
    'TCS.NS': 'TCS',
    'INFY.NS': 'Infosys',
    'HDFCBANK.NS': 'HDFC Bank',
    'ICICIBANK.NS': 'ICICI Bank',
    'SBIN.NS': 'State Bank of India',
    'ITC.NS': 'ITC',
    'LT.NS': 'Larsen & Toubro',
    'AXISBANK.NS': 'Axis Bank',
    'KOTAKBANK.NS': 'Kotak Bank'
}

symbol = st.selectbox(
    "Choose a Company",
    list(nifty_50.keys()),
    format_func=lambda x: nifty_50[x]
)

# -------------------------
# 3. Fetch Data (Auto Refresh)
# -------------------------
@st.cache_data(ttl=3600)
def get_data(symbol):
    today = datetime.today().strftime('%Y-%m-%d')
    df = yf.download(symbol, start="2023-01-01", end=today)
    return df

df = get_data(symbol)

# -------------------------
# 4. Validate Data
# -------------------------
if df.empty or len(df) < 100:
    st.error("❌ Not enough data to train model.")
    st.stop()

df = df.reset_index()

st.subheader(f"{nifty_50[symbol]} - Recent Data")
st.dataframe(df[['Date', 'Close', 'High']].tail())

# -------------------------
# 5. Preprocessing
# -------------------------
data = df[['Close']].values

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

sequence_length = 60
X, y = [], []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i])

X, y = np.array(X), np.array(y)

X = X.reshape((X.shape[0], X.shape[1], 1))

# Train-Test Split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# -------------------------
# 6. Build Model (Cached)
# -------------------------
@st.cache_resource
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = build_model((X.shape[1], 1))

# Train only once
if "trained" not in st.session_state:
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    st.session_state.trained = True

# -------------------------
# 7. Predict Next Day
# -------------------------
last_60 = scaled_data[-60:]
X_input = last_60.reshape(1, 60, 1)

pred_scaled = model.predict(X_input)
pred_price = scaler.inverse_transform(pred_scaled)[0][0]

st.success(f"🔮 Predicted Next Day Price: ₹{pred_price:.2f}")

# -------------------------
# 8. Plot Results
# -------------------------
predicted = model.predict(X_test)

train_plot = scaler.inverse_transform(y_train)
actual_plot = scaler.inverse_transform(y_test)
predicted_plot = scaler.inverse_transform(predicted)

fig, ax = plt.subplots()

ax.plot(range(len(train_plot)), train_plot, label="Train")
ax.plot(range(len(train_plot), len(train_plot)+len(actual_plot)), actual_plot, label="Actual")
ax.plot(range(len(train_plot), len(train_plot)+len(predicted_plot)), predicted_plot, label="Predicted")

ax.set_title(f"{nifty_50[symbol]} Stock Forecast")
ax.set_xlabel("Time")
ax.set_ylabel("Price (₹)")
ax.legend()

st.pyplot(fig)

# -------------------------
# 9. Footer
# -------------------------
st.caption("⚠️ This is a demo prediction model. Not financial advice.")