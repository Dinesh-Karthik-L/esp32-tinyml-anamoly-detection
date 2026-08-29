"""
app.py - Streamlit dashboard for the vibration anomaly detector.

Run with:
    streamlit run dashboard/app.py
"""
import os
import sys
import time

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from logger import ReadingLogger  # noqa: E402
from mqtt_topics import DEFAULT_DEVICE_ID  # noqa: E402

st.set_page_config(page_title="Vibration Anomaly Detector", layout="wide", page_icon="📈")

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "vibration_data.sqlite3")

db_path_input = st.sidebar.text_input("SQLite DB path", DB_PATH)
DEVICE_ID = st.sidebar.text_input("Device ID", DEFAULT_DEVICE_ID)

st.sidebar.markdown("---")
st.sidebar.caption(
    "This dashboard reads from the SQLite log written by backend/logger.py "
    "- it works identically whether the data came from the real ESP32 or "
    "simulator/virtual_sensor.py, since both run the exact same TFLite model."
)


@st.cache_resource
def get_reader(path):
    return ReadingLogger(path)


if not os.path.exists(db_path_input):
    st.warning(
        f"No database found at `{db_path_input}` yet. Start the logger and the "
        "simulator first:\n\n`python backend/logger.py` and "
        "`python simulator/virtual_sensor.py`."
    )
    st.stop()

reader = get_reader(db_path_input)
devices = reader.devices()
if DEVICE_ID not in devices:
    st.info(f"No data yet for device '{DEVICE_ID}'. Known devices: {devices or 'none'}")
    st.stop()

st.title("📈 Vibration Anomaly Detector")
st.caption("On-device TinyML: a quantized autoencoder trained only on normal vibration "
           "flags anything that reconstructs poorly.")

# ------------------------------------------------------------------ #
status_row = reader.latest(DEVICE_ID, "status", "status")
rms_row = reader.latest(DEVICE_ID, "vibration", "rms")
score_row = reader.latest(DEVICE_ID, "vibration", "anomaly_score")
alert_row = reader.latest(DEVICE_ID, "vibration", "alert")
fault_row = reader.latest(DEVICE_ID, "debug", "fault_type")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Device status", status_row[1].upper() if status_row else "unknown")
c2.metric("Vibration RMS", rms_row[1] if rms_row else "-")
c3.metric("Anomaly score", f"{float(score_row[1]):.4f}" if score_row else "-")
c4.metric("Alert", "🚨 ANOMALY" if alert_row and alert_row[1] == "1" else "✅ normal")

if fault_row and fault_row[1] != "none":
    st.caption(f"Debug (simulator only - not available on real hardware): "
               f"ground truth fault currently injected = **{fault_row[1]}**")

st.markdown("---")

# ------------------------------------------------------------------ #
st.subheader("History")
window_minutes = st.slider("Time window (minutes)", 1, 60, 5)
since = time.time() - window_minutes * 60

score_rows = reader.history(DEVICE_ID, "vibration", "anomaly_score", since=since)
if score_rows:
    df = pd.DataFrame(score_rows, columns=["timestamp", "value"])
    df["value"] = pd.to_numeric(df["value"])
    df["time"] = pd.to_datetime(df["timestamp"], unit="s")

    threshold_path = os.path.join(os.path.dirname(__file__), "..", "model", "artifacts", "threshold_quant.json")
    threshold = None
    if os.path.exists(threshold_path):
        import json
        with open(threshold_path) as f:
            threshold = json.load(f)["threshold"]

    st.line_chart(df.set_index("time")["value"], height=250, width="stretch")
    if threshold:
        st.caption(f"Anomaly score over time (threshold = {threshold:.4f} - points above this line "
                   f"triggered an alert)")
    else:
        st.caption("Anomaly score over time")
else:
    st.caption("No anomaly score data in this window yet.")

rms_rows = reader.history(DEVICE_ID, "vibration", "rms", since=since)
if rms_rows:
    rdf = pd.DataFrame(rms_rows, columns=["timestamp", "value"])
    rdf["value"] = pd.to_numeric(rdf["value"])
    rdf["time"] = pd.to_datetime(rdf["timestamp"], unit="s")
    st.line_chart(rdf.set_index("time")["value"], height=200, width="stretch")
    st.caption("Vibration RMS amplitude over time")

# ------------------------------------------------------------------ #
st.subheader("Alert log")
alert_rows = reader.history(DEVICE_ID, "vibration", "alert", since=since)
if alert_rows:
    adf = pd.DataFrame(alert_rows, columns=["timestamp", "value"])
    adf["time"] = pd.to_datetime(adf["timestamp"], unit="s")
    adf["state"] = adf["value"].map({"1": "🚨 ANOMALY", "0": "✅ normal"})
    st.dataframe(adf[["time", "state"]].sort_values("time", ascending=False), width="stretch", height=200)
else:
    st.caption("No alert transitions in this window yet.")

st.button("🔄 Refresh")
