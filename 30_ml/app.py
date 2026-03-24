
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Load model (cached for performance)
@st.cache_resource
def load_model():
    artifact = joblib.load("capstone_model_v1.joblib")
    return artifact['pipeline'], artifact['model'], artifact['metadata']

pipeline, model, metadata = load_model()

# Title
st.title("🏠 House Price Prediction App")
st.markdown("Enter house details to get an instant price estimate.")

# Sidebar for inputs
st.sidebar.header("House Features")

# Numeric inputs
overall_qual = st.sidebar.slider("Overall Quality (1-10)", 1, 10, 7)
gr_liv_area = st.sidebar.number_input("Living Area (sq ft)", min_value=300, max_value=6000, value=1500)
total_bsmt = st.sidebar.number_input("Basement Area (sq ft)", min_value=0, max_value=3000, value=800)
garage_cars = st.sidebar.slider("Garage Capacity (cars)", 0, 5, 2)
year_built = st.sidebar.slider("Year Built", 1870, 2024, 2000)
full_bath = st.sidebar.slider("Full Bathrooms", 0, 5, 2)
bedrooms = st.sidebar.slider("Bedrooms", 0, 10, 3)

# Categorical inputs
neighborhood = st.sidebar.selectbox(
    "Neighborhood",
    ["CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel", 
     "Somerst", "NWAmes", "OldTown", "BrkSide", "Sawyer"]
)
house_style = st.sidebar.selectbox(
    "House Style",
    ["1Story", "2Story", "1.5Fin", "SLvl", "SFoyer", "2.5Unf"]
)

# Prediction button
if st.sidebar.button("Predict Price", type="primary"):
    # Create input DataFrame
    input_data = pd.DataFrame([{
        "OverallQual": overall_qual,
        "GrLivArea": gr_liv_area,
        "TotalBsmtSF": total_bsmt,
        "GarageCars": garage_cars,
        "YearBuilt": year_built,
        "FullBath": full_bath,
        "BedroomAbvGr": bedrooms,
        "Neighborhood": neighborhood,
        "HouseStyle": house_style
    }])

    # Predict
    X_proc = pipeline.transform(input_data)
    pred_log = model.predict(X_proc)[0]
    pred_price = np.expm1(pred_log)

    # Display results
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Predicted Price", f"${pred_price:,.0f}")

    with col2:
        # Price per sqft
        price_per_sqft = pred_price / gr_liv_area
        st.metric("Price per Sq Ft", f"${price_per_sqft:,.0f}")

    with col3:
        st.metric("Model Confidence", f"R² = {metadata.get('metrics', {}).get('holdout_r2', 'N/A'):.2f}")

    # Visualization
    st.subheader("Price Context")

    # Show where this falls in distribution
    fig, ax = plt.subplots()
    # Simulated distribution (in production, use actual training data distribution)
    np.random.seed(42)
    sample_prices = np.random.lognormal(12, 0.4, 1000)
    ax.hist(sample_prices, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(pred_price, color='red', linestyle='--', linewidth=2, label='Your House')
    ax.set_xlabel('Price ($)')
    ax.set_ylabel('Frequency')
    ax.legend()
    st.pyplot(fig)

    # Feature importance for this prediction (simplified)
    st.subheader("Key Value Drivers")
    st.write(f"- **Quality Rating**: {overall_qual}/10 {'✅ Premium' if overall_qual >= 8 else '⚠️ Average' if overall_qual >= 5 else '❌ Below Average'}")
    st.write(f"- **Living Space**: {gr_liv_area:,} sq ft")
    st.write(f"- **Age**: {2024 - year_built} years old")
    st.write(f"- **Location**: {neighborhood}")

# Batch prediction section
st.markdown("---")
st.subheader("Batch Prediction")
uploaded_file = st.file_uploader("Upload CSV with house features", type=['csv'])

if uploaded_file:
    batch_df = pd.read_csv(uploaded_file)
    st.write(f"Uploaded {len(batch_df)} houses")
    st.dataframe(batch_df.head())

    if st.button("Run Batch Prediction"):
        X_batch = pipeline.transform(batch_df)
        preds_log = model.predict(X_batch)
        preds = np.expm1(preds_log)

        batch_df['Predicted_Price'] = preds
        st.success("Predictions complete!")
        st.dataframe(batch_df[['Predicted_Price']].describe())

        # Download link
        csv = batch_df.to_csv(index=False)
        st.download_button(
            label="Download Results",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.caption(f"Model version {metadata['version']} | Trained on {metadata['training_samples']} samples")
