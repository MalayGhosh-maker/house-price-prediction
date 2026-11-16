import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0")

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 House Price Prediction")
st.write("Enter the property details below to get an estimated price.")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sq ft)", min_value=200.0, max_value=20000.0, value=1000.0, step=50.0)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
    floors = st.number_input("Floors", min_value=1, max_value=5, value=2, step=1)

with col2:
    year_built = st.number_input("Year Built", min_value=1900, max_value=2026, value=1995, step=1)
    location = st.selectbox("Location", ["Downtown", "Suburban", "Urban", "Rural"])
    condition = st.selectbox("Condition", ["Excellent", "Good", "Fair", "Poor"])
    garage = st.selectbox("Garage", ["Yes", "No"])

if st.button("Predict House Price", type="primary"):
    payload = {
        "Area": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Floors": floors,
        "YearBuilt": year_built,
        "Location": location,
        "Condition": condition,
        "Garage": garage,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        price = response.json()["predicted_price"]
        st.success(f"### Estimated House Price: ${price:,.2f}")
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to FastAPI backend: {e}")
