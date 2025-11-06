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
# 🏷️ App Setup
# -----------------------------------------------------
st.set_page_config(page_title="Lahore AQI Dashboard", layout="wide")
st.title("🌫️ Lahore Air Quality Index — Live & 30-Day Forecast")
st.caption("📍 Location: Lahore, Pakistan")

# -----------------------------------------------------
# 🔧 API Configuration
# -----------------------------------------------------
TOKEN = "acbae8d1fe1e83b8d36a04b230cc4421684c2093"  # AQICN token
CITY = "geo:31.5204;74.3587"  # Lahore coordinates for AQICN

# -----------------------------------------------------
# 📡 Fetch Live AQI
# -----------------------------------------------------
def fetch_live_aqi():
    """Fetch live AQI and pollutant data for Lahore."""
    url = f"https://api.waqi.info/feed/{CITY}/?token={TOKEN}"
    try:
        response = requests.get(url).json()
        if response.get("status") == "ok":
            data = response["data"]
            return data["aqi"], data.get("iaqi", {})
        else:
            return None, {}
    except Exception as e:
        st.error(f"⚠️ Error fetching AQI: {e}")
        return None, {}

live_aqi, iaqi = fetch_live_aqi()

if live_aqi:
    st.success(f"✅ Current AQI in Lahore: **{live_aqi}**")
else:
    st.warning("⚠️ Live AQI unavailable — using fallback 160.")
    live_aqi = 160

# -----------------------------------------------------
# ⚙️ Load Trained Model
# -----------------------------------------------------
MODEL_PATH = "model_registry/rf_model.joblib"
FEATURE_PATH = "model_registry/feature_columns.json"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found. Please train and save the model first.")
    st.stop()

model = joblib.load(MODEL_PATH)

if os.path.exists(FEATURE_PATH):
    with open(FEATURE_PATH) as f:
        feature_columns = json.load(f)
else:
    try:
        feature_columns = list(model.feature_names_in_)
    except Exception:
        st.error("❌ Could not determine model feature names.")
        st.stop()

# -----------------------------------------------------
# 🔮 Generate 30-Day Forecast (Synthetic Inputs)
# -----------------------------------------------------
def synthesize_value(col):
    c = col.lower()
    if "pm2" in c: return np.random.uniform(40, 180)
    if "pm10" in c: return np.random.uniform(50, 250)
    if "no2" in c: return np.random.uniform(10, 120)
    if "o3" in c:  return np.random.uniform(10, 100)
    if "so2" in c: return np.random.uniform(5, 80)
    if "co" in c:  return np.random.uniform(0.1, 4)
    if "temp" in c or "t" in c: return np.random.uniform(10, 40)
    if "humid" in c or "h" in c: return np.random.uniform(20, 80)
    return np.random.uniform(0, 100)

dates, preds = [], []
for i in range(1, 31):
    vals = {c: synthesize_value(c) for c in feature_columns}
    X = pd.DataFrame([vals])
    preds.append(model.predict(X)[0])
    dates.append(datetime.now().date() + timedelta(days=i))

forecast_df = pd.DataFrame({"Date": dates, "Predicted_AQI": preds})
forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])
forecast_df["Week"] = forecast_df["Date"].dt.isocalendar().week
forecast_df["Day"] = forecast_df["Date"].dt.day_name()

# -----------------------------------------------------
# 📊 Chart 1 — Line Trend
# -----------------------------------------------------
st.subheader("📈 30-Day AQI Trend")
fig = px.line(
    forecast_df, x="Date", y="Predicted_AQI", markers=True,
    title="Predicted AQI Trend for Next 30 Days",
    color_discrete_sequence=["#e74c3c"]
)
fig.update_layout(yaxis_title="AQI", xaxis_title="Date")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------
# 📊 Chart 2 — Heatmap
# -----------------------------------------------------
st.subheader("🗓️ Daily AQI Heatmap")
fig2 = px.density_heatmap(
    forecast_df, x="Week", y="Day", z="Predicted_AQI",
    color_continuous_scale="RdYlGn_r",
    title="AQI Intensity Heatmap"
)
st.plotly_chart(fig2, use_container_width=True)

# -----------------------------------------------------
# 📊 Chart 3 — Pollutant Pie (Live)
# -----------------------------------------------------
if iaqi:
    st.subheader("🧪 Live Pollutant Composition")
    poll = {k.upper(): v["v"] for k, v in iaqi.items() if isinstance(v, dict)}
    fig3 = px.pie(
        values=list(poll.values()),
        names=list(poll.keys()),
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    st.plotly_chart(fig3, use_container_width=True)

# -----------------------------------------------------
# 📊 Chart 4 — Weekly Bars
# -----------------------------------------------------
st.subheader("📊 Average Weekly AQI")
wk = forecast_df.groupby("Week")["Predicted_AQI"].mean().reset_index()
fig4 = px.bar(
    wk, x="Week", y="Predicted_AQI",
    color="Predicted_AQI", color_continuous_scale="RdYlGn_r",
    title="Average Weekly AQI"
)
st.plotly_chart(fig4, use_container_width=True)

# -----------------------------------------------------
# 📊 Chart 5 — Radar: Pollutants vs WHO
# -----------------------------------------------------
who_limits = {"PM25": 25, "PM10": 50, "NO2": 40, "O3": 100, "SO2": 20, "CO": 4}
if iaqi:
    labels, vals, whos = [], [], []
    for k, v in iaqi.items():
        key = k.upper(); val = v["v"]
        labels.append(key); vals.append(val); whos.append(who_limits.get(key, 0))
    fig5 = go.Figure()
    fig5.add_trace(go.Scatterpolar(r=vals, theta=labels, fill='toself', name='Lahore'))
    fig5.add_trace(go.Scatterpolar(r=whos, theta=labels, fill='toself', name='WHO Limit'))
    fig5.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        title="🧭 Pollutants vs WHO Safe Limits"
    )
    st.plotly_chart(fig5, use_container_width=True)

# -----------------------------------------------------
# 🗺️ Chart 6 — Lahore Map
# -----------------------------------------------------
lat, lon = 31.582045, 74.329376
fig_map = go.Figure(go.Scattermapbox(
    lat=[lat],
    lon=[lon],
    mode="markers+text",
    text=[f"Lahore AQI: {int(live_aqi)}"],
    textposition="top center",
    marker=go.scattermapbox.Marker(
        size=20,
        color=live_aqi,
        colorscale="RdYlGn_r",
        showscale=True,
        colorbar=dict(title="AQI Level")
    )
))
fig_map.update_layout(
    mapbox=dict(
        style="carto-darkmatter",
        center=dict(lat=lat, lon=lon),
        zoom=9
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    title="📍 Lahore Air Quality Map"
)
st.plotly_chart(fig_map, use_container_width=True)

# -----------------------------------------------------
# 📊 Chart 7 — AQI Distribution Histogram
# -----------------------------------------------------
st.subheader("📊 AQI Distribution (Next 30 Days)")
fig6 = px.histogram(forecast_df, x="Predicted_AQI", nbins=15,
                    color_discrete_sequence=["#e74c3c"])
st.plotly_chart(fig6, use_container_width=True)

# -----------------------------------------------------
# 📄 Summary & Footer
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
st.markdown("👨‍💻 **Developed by Saifullah Khalid** — © 2025 Pearls Data Projects")
