# app.py

"""
Streamlit application for car price prediction.

Loads trained model from: models/best_model.pkl
"""

from pathlib import Path
import pandas as pd
import joblib
import streamlit as st


# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"


# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            "Model not found.\n\n"
            "Please run training first:\n"
            "```bash\npython src/train.py\n```"
        )
        st.stop()
    return joblib.load(MODEL_PATH)


model = load_model()


# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────

st.title("Car Price Prediction")

st.write("Enter car details:")

brand = st.text_input("Brand", "Toyota")
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Electric"])
condition = st.selectbox("Condition", ["New", "Used", "Certified"])
year = st.number_input("Year", min_value=1990, max_value=2025, value=2015)
mileage = st.number_input("Mileage", min_value=0, value=120000)
engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=6.0, value=1.6)

if st.button("Predict price"):
    input_df = pd.DataFrame(
        [
            {
                "Mileage": mileage,
                "Engine Size": engine_size,
                "Year": year,
                "Fuel Type": fuel_type,
                "Brand": brand,
                "Condition": condition,
            }
        ]
    )

    prediction = model.predict(input_df)[0]

    st.success(f"Estimated price: {prediction:,.0f}")
