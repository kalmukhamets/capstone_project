import joblib
import pandas as pd
import streamlit as st

model = joblib.load('model_pipeline.pkl')

# UI
brand = st.text_input("Brand")
model_car = st.text_input("Model")
city = st.text_input("City")
year = st.number_input("Year", 1990, 2025)
engine = st.number_input("Engine Size (L)", 0.5, 6.0)
mileage = st.number_input("Mileage", 0)

# other categorical fields
body = st.text_input("Body")
transmission = st.text_input("Transmission")
wheel = st.text_input("Wheel")
color = st.text_input("Color")
drive = st.text_input("Drive")
customs = st.text_input("CustomsCleared")
fuel = st.text_input("FuelType")

if st.button("Predict Price"):
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
        'FuelType': fuel
    }])
    
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Price: {int(prediction):,} Tenge")
