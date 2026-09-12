import streamlit as st
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page settings
st.set_page_config(
    page_title="Home Worth-Smart",
    page_icon="🏠",
    layout="wide"
)

# Title
st.title("🏠 Home Worth-Smart Predictions")
st.write("Enter the property details below to estimate its price.")

# Sidebar
st.sidebar.header("🏡 Property Details")

location = st.sidebar.selectbox(
    "Location Type",
    ["Urban", "Suburban", "Rural"]
)

property_type = st.sidebar.selectbox(
    "Property Type",
    ["Residential", "Commercial", "Apartment"]
)

size = st.sidebar.number_input(
    "Size (sq ft)",
    min_value=100,
    max_value=10000,
    value=1500,
    step=100
)

bedrooms = st.sidebar.number_input(
    "Bedrooms",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)

bathrooms = st.sidebar.number_input(
    "Bathrooms",
    min_value=1,
    max_value=10,
    value=2,
    step=1
)

age = st.sidebar.number_input(
    "Age of House (years)",
    min_value=0,
    max_value=100,
    value=5,
    step=1
)

latitude = st.sidebar.number_input(
    "Latitude",
    min_value=-90.0,
    max_value=90.0,
    value=17.3850,
    format="%.4f"
)

longitude = st.sidebar.number_input(
    "Longitude",
    min_value=-180.0,
    max_value=180.0,
    value=78.4867,
    format="%.4f"
)

amenity_proximity = st.sidebar.number_input(
    "Proximity to amenities (km)",
    min_value=0.1,
    max_value=100.0,
    value=2.0,
    step=0.5
)

# Prediction button
st.subheader("🔮 Price Prediction")

if st.button("PREDICT HOUSE PRICE", use_container_width=True):

    # Feature engineering
    features = pd.DataFrame({
        "latitude": [latitude],
        "longitude": [longitude],
        "size": [size],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "age": [age],
        "amenity_proximity": [amenity_proximity],
        "size_bedrooms": [size * bedrooms],
        "size_bathrooms": [size * bathrooms],
        "inverse_age": [1 / (age + 1)],
        "inverse_proximity": [1 / (amenity_proximity + 0.1)]
    })

    # Scale features
    features_scaled = scaler.transform(features)

    # Model prediction
    base_prediction = model.predict(features_scaled)[0]

    # Location multipliers
    location_multiplier = {
        "Urban": 1.3,
        "Suburban": 1.1,
        "Rural": 0.9
    }

    # Property type multipliers
    property_multiplier = {
        "Commercial": 1.5,
        "Residential": 1.0,
        "Apartment": 0.8
    }

    # Final prediction
    final_price = (
        base_prediction
        * location_multiplier[location]
        * property_multiplier[property_type]
    )

    # Display result
    st.success("Prediction completed successfully!")

    st.metric(
        label="🏠 Estimated House Price",
        value=f"₹{final_price:,.0f}"
    )

    st.write(
        f"**Location:** {location}  |  "
        f"**Property Type:** {property_type}"
    )

    # Location map
    st.subheader("📍 Property Location")

    map_data = pd.DataFrame({
        "latitude": [latitude],
        "longitude": [longitude]
    })

    st.map(map_data, zoom=11)

# Dataset section
st.divider()

with st.expander("📊 View Sample Dataset"):
    data = pd.read_csv("house_data.csv")
    st.dataframe(data.head(10), use_container_width=True)