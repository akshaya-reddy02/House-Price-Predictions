import streamlit as st
import pandas as pd
import pydeck as pdk
import joblib
import numpy as np

# Load the model and scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🏡 Home Worth-Smart predictions")

# Sidebar inputs
st.sidebar.header("Enter House Features")

# 📍 Location and Property Type
location = st.sidebar.selectbox("Select Location Type", ["Urban", "Suburban", "Rural"])
property_type = st.sidebar.selectbox("Property Type", ["Residential", "Commercial", "Apartment"])

# 🛠️ House Features (Blank Inputs)
size = st.sidebar.text_input("Size (sq ft)", "")
bedrooms = st.sidebar.text_input("Bedrooms", "")
bathrooms = st.sidebar.text_input("Bathrooms", "")
age = st.sidebar.text_input("Age of House (years)", "")
latitude = st.sidebar.text_input("Latitude", "")
longitude = st.sidebar.text_input("Longitude", "")
amenity_proximity = st.sidebar.text_input("Proximity to amenities (km)", "")

# ✅ Real-time Map Section
st.subheader("📍 Real-Time Location Map")

try:
    if latitude and longitude:
        lat = float(latitude)
        lon = float(longitude)

        # Display map with a marker
        map_layer = pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame({"lat": [lat], "lon": [lon]}),
            get_position=["lon", "lat"],
            get_color=[255, 0, 0, 160],  # Red marker
            get_radius=150,
        )

        map_view = pdk.ViewState(
            latitude=lat,
            longitude=lon,
            zoom=12,
            pitch=45,
        )

        st.pydeck_chart(pdk.Deck(
            layers=[map_layer],
            initial_view_state=map_view,
            map_style='mapbox://styles/mapbox/streets-v11'
        ))
    else:
        st.warning("Enter latitude and longitude to view the map.")

except ValueError:
    st.error("Please enter valid latitude and longitude coordinates.")

# ✅ Location Multiplier Logic
location_multiplier = {
    "Urban": 1.3,       # Higher prices
    "Suburban": 1.1,    # Moderate prices
    "Rural": 0.9        # Lower prices
}

# ✅ Property Type Multiplier Logic
property_multiplier = {
    "Commercial": 1.5,   # Highest price per sqft
    "Residential": 1.0,  # Standard price
    "Apartment": 0.8     # Lower price per sqft
}

# 🛠️ Centering the "Predict Price" button using columns
col1, col2, col3 = st.columns([1, 2, 1])

prediction = None

with col2:
    if st.button("🔮 **PREDICT PRICE**", help="Click to estimate the house price", use_container_width=True):
        try:
            # Ensure all fields are filled
            if not (size and bedrooms and bathrooms and age and latitude and longitude and amenity_proximity):
                st.error("Please fill in all fields before predicting.")
            else:
                # Convert inputs
                size = int(size)
                bedrooms = int(bedrooms)
                bathrooms = int(bathrooms)
                age = int(age)
                latitude = float(latitude)
                longitude = float(longitude)
                amenity_proximity = float(amenity_proximity)

                # ✅ Apply transformations
                features = pd.DataFrame({
                    'latitude': [latitude],
                    'longitude': [longitude],
                    'size': [size],
                    'bedrooms': [bedrooms],
                    'bathrooms': [bathrooms],
                    'age': [age],
                    'amenity_proximity': [amenity_proximity],
                    'size_bedrooms': [size * bedrooms],
                    'size_bathrooms': [size * bathrooms],
                    'inverse_age': [1 / (age + 1)],
                    'inverse_proximity': [1 / (amenity_proximity + 0.1)]
                })

                # Scale the features
                features_scaled = scaler.transform(features)

                # ML model prediction
                base_prediction = model.predict(features_scaled)[0]

                # ✅ Apply location and property multipliers
                location_factor = location_multiplier.get(location, 1.0)
                property_factor = property_multiplier.get(property_type, 1.0)

                # ✅ Final price calculation
                final_price = base_prediction * location_factor * property_factor
                prediction = final_price

        except ValueError:
            st.error("Please enter valid numeric values for all fields.")

# ✅ Display the predicted price ABOVE the sample dataset with enhanced styling
if prediction is not None:
    st.markdown(
        f"""
        <div style="text-align: center; margin-top: 20px; padding: 20px; border-radius: 10px; 
        background-color: #f0f2f6; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);">
            <h2 style="color: #4CAF50;">🏠 Estimated Price: <strong>${prediction:,.2f}</strong></h2>
            <p style="color: #000;">📍 Location: <strong>{location}</strong> | 🏢 Property Type: <strong>{property_type}</strong></p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ✅ Display the sample dataset option below the prediction
if st.checkbox("Show sample dataset"):
    data = pd.read_csv('house_data.csv')
    st.write(data.head())
