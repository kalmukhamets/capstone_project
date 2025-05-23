import joblib
import pandas as pd
import streamlit as st
import psycopg2
import os

# Load model pipeline (make sure 'model_pipeline.pkl' is in the same directory)
model = joblib.load('model_pipeline.pkl')  # pipeline includes encoder + model

st.title("🚗 Car Price Predictor")

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
mileage_per_year = mileage / car_age if car_age > 0 else mileage

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
if st.button("🔮 Predict Price"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Estimated Price: {int(prediction):,} Tenge")

        # --- Feedback Form ---
        st.subheader("Was this prediction accurate?")
        feedback = st.radio("Your opinion:", ["Too high", "Too low", "Reasonable"])
        comment = st.text_area("Additional comments (optional)")

        if st.button("Submit Feedback"):
            try:
                # Connect to PostgreSQL (You can replace this with your environment variable if preferred)
                conn = psycopg2.connect(
                    "postgresql://neondb_owner:npg_Mke3v1tQcoAg@ep-morning-surf-a8bcjk1b-pooler.eastus2.azure.neon.tech/neondb?sslmode=require"
                )
                cur = conn.cursor()

                # Ensure feedback table exists
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS public.feedback (
                        id SERIAL PRIMARY KEY,
                        brand TEXT,
                        model TEXT,
                        prediction FLOAT,
                        feedback TEXT,
                        comment TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()

                # Insert feedback
                cur.execute("""
                    INSERT INTO public.feedback (brand, model, prediction, feedback, comment)
                    VALUES (%s, %s, %s, %s, %s)
                """, (brand, model_car, prediction, feedback, comment))
                conn.commit()

                st.success("✅ Feedback submitted. Thank you!")
            except Exception as e:
                st.error(f"❌ Failed to submit feedback: {e}")
            finally:
                if 'conn' in locals():
                    conn.close()

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
