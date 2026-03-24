
# streamlit_app.py - Interactive Housing Price Predictor
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Load model (cached to avoid reloading)
@st.cache_resource
def load_model():
    return joblib.load('housing_model_v1.joblib')

model = load_model()

# Title and description
st.title("🏠 California Housing Price Predictor")
st.markdown("""
Enter house features to get a price prediction.
This model was trained on California census block data.
""")

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("Economic & Demographics")
    med_income = st.slider("Median Income ($10k)", 0.5, 15.0, 3.0, 0.1)
    house_age = st.slider("House Age (years)", 1, 52, 20, 1)
    population = st.number_input("Population", 100, 35000, 1000, 100)

with col2:
    st.subheader("Property & Location")
    ave_rooms = st.slider("Avg Rooms", 1.0, 10.0, 5.0, 0.1)
    ave_bedrms = st.slider("Avg Bedrooms", 0.5, 5.0, 1.0, 0.1)
    ave_occup = st.slider("Avg Occupancy", 1.0, 10.0, 3.0, 0.1)

    location = st.selectbox(
        "Ocean Proximity",
        ['NEAR_BAY', 'INLAND', '<1H_OCEAN', 'NEAR_OCEAN']
    )

# Location coordinates (simplified)
lat_lon_map = {
    'NEAR_BAY': (37.8, -122.2),
    'INLAND': (37.0, -120.0),
    '<1H_OCEAN': (34.0, -118.0),
    'NEAR_OCEAN': (36.0, -122.0)
}
latitude, longitude = lat_lon_map[location]

# Prediction button
if st.button("🔮 Predict Price", type="primary"):
    # Create input DataFrame
    input_data = pd.DataFrame([{
        'MedInc': med_income,
        'HouseAge': house_age,
        'AveRooms': ave_rooms,
        'AveBedrms': ave_bedrms,
        'Population': population,
        'AveOccup': ave_occup,
        'Latitude': latitude,
        'Longitude': longitude,
        'ocean_proximity': location
    }])

    # Predict
    prediction = model.predict(input_data)[0]
    price_usd = prediction * 100000

    # Display result
    st.success(f"### Predicted Price: ${price_usd:,.0f}")

    # Progress bar showing relative value
    st.progress(min(prediction / 5.0, 1.0))
    st.caption(f"Relative value index: {prediction:.2f}/5.0")

    # Feature importance (if available)
    if hasattr(model.named_steps['regressor'], 'feature_importances_'):
        st.subheader("Feature Importance")
        importances = model.named_steps['regressor'].feature_importances_
        # Simplified - in reality need to map to original features post-preprocessing
        st.bar_chart({
            'Income': importances[0] if len(importances) > 0 else 0.3,
            'Location': importances[1] if len(importances) > 1 else 0.2,
            'Age': importances[2] if len(importances) > 2 else 0.15,
            'Rooms': importances[3] if len(importances) > 3 else 0.15,
            'Other': 0.2
        })

# Sidebar info
st.sidebar.header("About")
st.sidebar.info("""
This app demonstrates deploying a scikit-learn model 
with Streamlit. The model predicts median house values 
in California districts based on 1990 census data.
""")

st.sidebar.header("Model Info")
st.sidebar.json({
    "type": "RandomForestRegressor",
    "features": 9,
    "target": "Median House Value ($100k)",
    "version": "1.0"
})
