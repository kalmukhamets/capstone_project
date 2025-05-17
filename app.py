import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("xgb_model.pkl")

st.title("Car Price Predictor")

# Input fields
brand = st.text_input("Бренд")
model_car = st.text_input("Модель")
year = st.number_input("Год", 1990, 2025)

# Example features dict – adjust as per your training features
if st.button("Predict Price"):
    input_data = pd.DataFrame([{
        "Brand": brand,
        "Model": model_car,
        "Year": year,
    }])

    # Make sure to preprocess the input the same way as training
    # input_data = preprocess(input_data)  # Optional

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
input_data['Бренд'] = le.fit_transform(input_data['Бренд'])
input_data['Модель'] = le.fit_transform(input_data['Модель'])

    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Price: {int(prediction):,} Tenge")
