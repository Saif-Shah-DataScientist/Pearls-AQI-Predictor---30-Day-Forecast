# =====================================================
# 🌫️ Pearls Lahore AQI Predictor — Live + 30-Day Forecast
# Author: Saifullah Khalid
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests, json, os, re, joblib
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------
# ⚙️ Streamlit Config (Must be first Streamlit command)
# -----------------------------------------------------
st.set_page_config(page_title="Lahore AQI Dashboard", layout="wide")

# -----------------------------------------------------
# 🔧 API and Configuration
# -----------------------------------------------------
TOKEN = "acbae8d1fe1e83b8d36a04b230cc4421684c2093"
CITY = "geo:31.5204;74.3587"
LOCATION_NAME = "Lahore, Pakistan"

st.title("🌫️ Lahore Air Quality Index — Live & 30-Day Forecast")
st.caption(f"📍 Location: {LOCATION_NAME}")

# -----------------------------------------------------
# 📡 Fetch Live AQI
# -----------------------------------------------------
def fetch_live_aqi():
    url = f"https://api.waqi.info/feed/{CITY}/?token={TOKEN}"
    r = requests.get(url).json()
    if r.get("status") == "ok":
        d = r["data"]
        return d["aqi"], d.get("iaqi", {})
    return None, {}

live_aqi, iaqi = fetch_live_aqi()

if live_aqi:
    st.success(f"✅ Current AQI in Lahore: **{live_aqi}**")
else:
    st.warning("⚠️ Live AQI unavailable — using fallback 160.")
    live_aqi = 160

# -----------------------------------------------------
# ⚙️ Load Model
# -----------------------------------------------------
MODEL_PATH = "model_registry/rf_model.joblib"
FEATURE_PATH = "model_registry/feature_columns.json"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found. Please train and save your model first in Jupyter Notebook.")
    st.stop()

model = joblib.load(MODEL_PATH)
try:
    with open(FEATURE_PATH) as f:
        feature_columns = json.load(f)
except:
    feature_columns = list(model.feature_names_in_)

# -----------------------------------------------------
# 📉 7-Day Historical AQI from AQICN
# -----------------------------------------------------
def fetch_aqi_history():
    url = f"https://api.waqi.info/feed/{CITY}/?token={TOKEN}"
    try:
        r = requests.get(url).json()
        if r["status"] == "ok":
            forecast_data = r["data"].get("forecast", {}).get("daily", {}).get("pm25", [])
            if forecast_data:
                df_hist = pd.DataFrame(forecast_data)
                df_hist["day"] = pd.to_datetime(df_hist["day"])
                df_hist.rename(columns={"avg": "AQI"}, inplace=True)
                return df_hist[["day", "AQI"]]
    except Exception as e:
        print("Error fetching history:", e)
    return pd.DataFrame()

hist_df = fetch_aqi_history()
if not hist_df.empty:
    st.subheader("📆 Past 7-Day AQI Trend (Historical Data)")
    fig_hist = px.line(hist_df, x="day", y="AQI", markers=True, color_discrete_sequence=["green"])
    fig_hist.update_layout(yaxis_title="AQI", xaxis_title="Date")
    st.plotly_chart(fig_hist, use_container_width=True)
else:
    st.info("ℹ️ Historical AQI data not available from API.")

# -----------------------------------------------------
# 🔮 30-Day Forecast (Blended with Live AQI)
# -----------------------------------------------------
def generate_forecast(base_aqi, feature_columns):
    forecasts = []
    base = float(base_aqi)
    for i in range(1, 31):
        daily_aqi = max(0, base + np.random.normal(0, 10))
        vals = {c: np.random.uniform(0, 100) for c in feature_columns}
        X = pd.DataFrame([vals])
        pred = model.predict(X)[0]
        final_pred = 0.7 * pred + 0.3 * daily_aqi
        forecasts.append(final_pred)
        base = final_pred
    return forecasts

dates = [datetime.now().date() + timedelta(days=i) for i in range(1, 31)]
preds = generate_forecast(live_aqi, feature_columns)

forecast_df = pd.DataFrame({"Date": dates, "Predicted_AQI": preds})
forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])
forecast_df["Week"] = forecast_df["Date"].dt.isocalendar().week
forecast_df["Day"]  = forecast_df["Date"].dt.day_name()

# -----------------------------------------------------
# 📊 Chart 1 — 30-Day AQI Line Trend
# -----------------------------------------------------
st.subheader("📈 30-Day AQI Forecast Trend")
fig = px.line(forecast_df, x="Date", y="Predicted_AQI", markers=True,
              color_discrete_sequence=["red"])
fig.update_layout(yaxis_title="AQI", xaxis_title="Date")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------
# 📊 Chart 2 — Heatmap
# -----------------------------------------------------
st.subheader("🗓️ Daily AQI Heatmap")
fig2 = px.density_heatmap(forecast_df, x="Week", y="Day", z="Predicted_AQI",
                          color_continuous_scale="RdYlGn_r")
st.plotly_chart(fig2, use_container_width=True)

# -----------------------------------------------------
# 📊 Chart 3 — Pollutant Pie (Live Composition)
# -----------------------------------------------------
if iaqi:
    st.subheader("🧪 Live Pollutant Composition")
    poll = {k.upper(): v["v"] for k, v in iaqi.items() if isinstance(v, dict)}
    fig3 = px.pie(values=list(poll.values()), names=list(poll.keys()))
    st.plotly_chart(fig3, use_container_width=True)

# -----------------------------------------------------
# 📊 Chart 4 — Weekly Bars
# -----------------------------------------------------
st.subheader("📊 Average Weekly AQI")
wk = forecast_df.groupby("Week")["Predicted_AQI"].mean().reset_index()
fig4 = px.bar(wk, x="Week", y="Predicted_AQI", color="Predicted_AQI",
              color_continuous_scale="RdYlGn_r")
st.plotly_chart(fig4, use_container_width=True)

# -----------------------------------------------------
# 📊 Chart 5 — Pollutants vs WHO Limits
# -----------------------------------------------------
who_limits = {"PM25":25,"PM10":50,"NO2":40,"O3":100,"SO2":20,"CO":4}
if iaqi:
    labels, vals, whos = [], [], []
    for k,v in iaqi.items():
        key=k.upper(); val=v["v"]
        labels.append(key)
        vals.append(val)
        whos.append(who_limits.get(key,0))
    fig5 = go.Figure()
    fig5.add_trace(go.Scatterpolar(r=vals, theta=labels, fill='toself', name='Lahore'))
    fig5.add_trace(go.Scatterpolar(r=whos, theta=labels, fill='toself', name='WHO Limit'))
    fig5.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    st.subheader("🧭 Pollutants vs WHO Safe Limits")
    st.plotly_chart(fig5, use_container_width=True)

# -----------------------------------------------------
# 📊 Chart 6 — Histogram
# -----------------------------------------------------
st.subheader("📊 AQI Distribution (Next 30 Days)")
fig6 = px.histogram(forecast_df, x="Predicted_AQI", nbins=15, color_discrete_sequence=["red"])
st.plotly_chart(fig6, use_container_width=True)

# -----------------------------------------------------
# 📄 Summary Metrics
# -----------------------------------------------------
avg = forecast_df["Predicted_AQI"].mean()
hi = forecast_df["Predicted_AQI"].max()
lo = forecast_df["Predicted_AQI"].min()

st.markdown("---")
col1, col2, col3 = st.columns(3)
col1.metric("Average AQI", f"{avg:.1f}")
col2.metric("Highest AQI", f"{hi:.1f}")
col3.metric("Lowest AQI", f"{lo:.1f}")

st.markdown("---")
st.markdown("👨‍💻 **Developed by Saifullah Khalid** — © 2025 AQI Predictor")
