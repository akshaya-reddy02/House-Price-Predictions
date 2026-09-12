import os
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


# Page settings
st.set_page_config(
    page_title="Home Worth-Smart",
    page_icon="🏠",
    layout="wide"
)


# Automatically create the model if it does not exist
def load_or_train_model():

    model_file = "best_model.pkl"
    scaler_file = "scaler.pkl"

    if os.path.exists(model_file) and os.path.exists(scaler_file):
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        return model, scaler

    # Load dataset
    data = pd.read_csv("house_data.csv")

    # Feature engineering
    data["size_bedrooms"] = data["size"] * data["bedrooms"]
    data["size_bathrooms"] = data["size"] * data["bathrooms"]
    data["inverse_age"] = 1 / (data["age"] + 1)
    data["inverse_proximity"] = 1 / (data["amenity_proximity"] + 0.1)

    features = [
        "latitude",
        "longitude",
        "size",
        "bedrooms",
        "bathrooms",
        "age",
        "amenity_proximity",
        "size_bedrooms",
        "size_bathrooms",
        "inverse_age",
        "inverse_proximity"
    ]

    X = data[features]
    y = data["price"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_scaled, y)

    # Save model files
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)

    return model, scaler


# Load or train model
model, scaler = load_or_train_model()


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


# Prediction section
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

    st.dataframe(
        data.head(10),
        use_container_width=True
    )