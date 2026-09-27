import streamlit as st
import pandas as pd
import pickle


#load model

IXGB_Model = pickle.load(open("Nutriclass_model.pkl",'rb'))
label_encoder = pickle.load(open("food_label_encoder.pkl", "rb"))

st.title("NutriClass - Food Prediction")

st.write("Enter the nutrition values")

calories = st.number_input("Calories")
protein = st.number_input("Protein")
fat = st.number_input("Fat")
carbs = st.number_input("Carbs")
sugar = st.number_input("Sugar")
fiber = st.number_input("Fiber")
sodium = st.number_input("Sodium")
cholesterol = st.number_input("Cholesterol")
glycemic_index = st.number_input("Glycemic Index")
water_content = st.number_input("Water Content")
serving_size = st.number_input("Serving Size")
meal_type = st.selectbox(
    "Meal Type",
    ["breakfast", "lunch", "dinner", "snack"]
)
if st.button("Predict"):

    new_data = pd.DataFrame({
        "calories": [calories],
        "protein": [protein],
        "fat": [fat],
        "carbs": [carbs],
        "sugar": [sugar],
        "fiber": [fiber],
        "sodium": [sodium],
        "cholesterol": [cholesterol],
        "glycemic_index": [glycemic_index],
        "water_content": [water_content],
        "serving_size": [serving_size],
        "meal_type": [meal_type]
    })

    prediction = IXGB_Model.predict(new_data)

    prediction_food = label_encoder.inverse_transform(
    prediction)

    st.success(f"Predicted Food: {prediction_food[0]}")