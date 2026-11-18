import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0")

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 House Price Prediction")
st.write("Enter the property details below to get an estimated price.")

col1, col2, col3, col4 = st.columns(4)

with col1:
    house_id = st.number_input("House ID", min_value=1, max_value=12000, value=1, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=6, value=1, step=1)
    bhk = st.number_input("BHK", min_value=1, max_value=5, value=1, step=1)
    super_area_sqft = st.number_input("Super Area (sq ft)", min_value=200, max_value=5000)
    distance_to_city_center_km = st.number_input("Distance to City Center (km)", min_value=0, max_value=50, value=1, step=1)

with col2:
    city = st.selectbox("City", ["Hyderabad", "Bangalore", "Pune", "Mumbai", "Nagpur"])
    locality_tier = st.selectbox("Locality_Tier", ["Mid", "Budget", "Premium"])
    parking = st.number_input("Parking", min_value=0, max_value=1, value=1, step=1)
    furnishing = st.selectbox("Furnishing", ["Semi-Furnished", "Unfurnished", "Fully-Furnished"])
    lift = st.number_input("Lift", min_value=0, max_value=1, value=1, step=1)

with col3:
    carpet_area_sqft = st.number_input("Carpet Area (sq ft)", min_value=200, max_value=5000)
    floor_no = st.number_input("Floor No", min_value=0, max_value=30, value=0, step=1)
    total_floors = st.number_input("Total Floors", min_value=1, max_value=40, value=1, step=1)
    property_age_years = st.number_input("Property Age (years)", min_value=0, max_value=40, value=0, step=1)
    distance_to_metro_km = st.number_input("Distance to Metro (km)", min_value=0, max_value=20, value=1, step=1)

with col4:
    gated_society = st.number_input("Gated Society", min_value=0, max_value=1, value=1, step=1)
    nearby_school_km = st.number_input("Distance to Nearby School (km)", min_value=0, max_value=20, value=1, step=1)
    nearby_hospital_km = st.number_input("Distance to Nearby Hospital (km)", min_value=0, max_value=20, value=1, step=1)
    crime_rate_index = st.number_input("Crime Rate Index", min_value=0, max_value=100, value=1, step=1)
    price_per_sqft_inr = st.number_input("Price per sq ft (INR)", min_value=500, max_value=30000)



if st.button("Predict House Price", type="primary"):
    payload = {
        "House_ID": house_id,
        "City": city,
        "Locality_Tier": locality_tier,
        "Bathrooms": bathrooms,
        "BHK": bhk,
        "Super_Area_sqft": super_area_sqft,
        "Carpet_Area_sqft": carpet_area_sqft,
        "Floor_No": floor_no,
        "Total_Floors": total_floors,
        "Property_Age_years": property_age_years,
        "Parking": parking,
        "Furnishing": furnishing,
        "Lift": lift,
        "Gated_Society": gated_society,
        "Distance_to_Metro_km": distance_to_metro_km,
        "Distance_to_CityCenter_km": distance_to_city_center_km,
        "Nearby_School_km":  nearby_school_km,
        "Nearby_Hospital_km": nearby_hospital_km,
        "Crime_Rate_Index": crime_rate_index,
        "Price_per_sqft_INR": price_per_sqft_inr
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        price = response.json()["Predicted Price"]
        st.success(f"### Estimated House Price: Rs{price:,.2f}")
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to FastAPI backend: {e}")
