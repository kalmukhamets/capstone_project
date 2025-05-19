import joblib
import pandas as pd
import streamlit as st

model = joblib.load('model_pipeline.pkl')  # pipeline includes encoder + model

st.title("Car Price Predictor")

# --- User Inputs ---
brand = st.text_input("Brand")
model_car = st.text_input("Model")
city = st.text_input("City")
year = st.number_input("Year", min_value=1990, max_value=2025)
engine = st.number_input("Engine Size (L)", min_value=0.5, max_value=6.0, step=0.1)
mileage = st.number_input("Mileage (km)", min_value=0)

body = st.text_input("Body")
transmission = st.text_input("Transmission")
wheel = st.text_input("Wheel")
color = st.text_input("Color")
drive = st.text_input("Drive")
customs = st.text_input("CustomsCleared")
fuel = st.text_input("FuelType")

# --- Feature Engineering ---
current_year = 2025
car_age = current_year - year
mileage_per_year = mileage / car_age if car_age > 0 else mileage  # prevent division by zero

# Luxury brand flag
luxury_brands = ['BMW', 'Audi', 'Lexus', 'Mercedes-Benz', 'Porsche']
is_luxury = 1 if brand in luxury_brands else 0

# --- Create Input DataFrame ---
input_df = pd.DataFrame([{
    'Brand': brand,
    'Model': model_car,
    'City': city,
    'Year': year,
    'EngineSize': engine,
    'Mileage': mileage,
    'Body': body,
    'Transmission': transmission,
    'Wheel': wheel,
    'Color': color,
    'Drive': drive,
    'CustomsCleared': customs,
    'FuelType': fuel,
    'CarAge': car_age,
    'MileagePerYear': mileage_per_year,
    'IsLuxuryBrand': is_luxury
}])


# --- Prediction ---
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Price: {int(prediction):,} Tenge")
