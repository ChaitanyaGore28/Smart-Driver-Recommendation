# =======================================================
# app.py | Streamlit Front-End for Smart Driving System
# =======================================================
import streamlit as st
from feedback_engine import analyze_driving

st.set_page_config(page_title="Smart Driving Feedback System", page_icon="🚗")
st.title("🚗 Smart Driving Feedback System (RAG + AI Edition)")

st.markdown("""
Welcome to the Smart Driving Feedback System (RAG Edition) 🚗  
This app:
- ⚙ Analyzes real-time driving sensor data  
- 🌐 Fetches live driving safety information from the web  
- 🤖 Generates short, human-like personalized feedback using an AI model  
""")

# Sidebar Inputs
st.sidebar.header("📥 Input Sensor Data")
ax = st.sidebar.number_input("AX (Acceleration X):", -10.0, 10.0, 0.0)
ay = st.sidebar.number_input("AY:", -10.0, 10.0, 0.0)
az = st.sidebar.number_input("AZ:", -10.0, 10.0, 0.0)
spd = st.sidebar.number_input("Speed (km/h):", 0, 200, 50)
brk = st.sidebar.number_input("Brake Value:", 0, 50, 0)
acc = st.sidebar.number_input("Accelerator Value:", 0, 50, 0)

if st.sidebar.button("🚦 Get Feedback"):
    with st.spinner("Analyzing driving behavior and generating personalized feedback..."):
        result = analyze_driving(ax, ay, az, spd, brk, acc)

    st.success("✅ Analysis Complete!")
    st.write("### 🚦 Driving Feedback:")
    st.info(f"Event: {result['event']}")
    st.markdown(f"Message: {result['message']}")