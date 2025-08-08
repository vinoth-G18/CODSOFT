import streamlit as st
import pandas as pd
import numpy as np
import joblib
from geopy.distance import geodesic
from sklearn.preprocessing import LabelEncoder

# Load model
model = joblib.load("model.pkl")

# Load encoders
le_category = joblib.load("label_encoder_category.pkl")
le_merchant = joblib.load("label_encoder_merchant.pkl")
le_gender = joblib.load("label_encoder_gender.pkl")

# Distance function
def hav(lat1, long1, merch_lat2, merch_long2):
    return np.array([geodesic((a, b), (c, d)).km for a, b, c, d in zip(lat1, long1, merch_lat2, merch_long2)])

# Streamlit UI
st.title("💳 Credit Card Fraud Detection")

st.header("Transaction Input")

# Inputs

cc_num = st.number_input("Credit Card Number", format="%.0f")
merchant = st.selectbox("Merchant", le_merchant.classes_)
category = st.selectbox("Category", le_category.classes_)
amt = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")
gender = st.selectbox("Gender", le_gender.classes_)
lat = st.number_input("Customer Latitude", format="%.6f")
long = st.number_input("Customer Longitude", format="%.6f")
merch_lat = st.number_input("Merchant Latitude", format="%.6f")
merch_long = st.number_input("Merchant Longitude", format="%.6f")
hour = st.number_input("Hour (0–23)", min_value=0, max_value=23)
day = st.number_input("Day (1–31)", min_value=1, max_value=31)
month = st.number_input("Month (1–12)", min_value=1, max_value=12)

# When Predict button is clicked
if st.button("Predict Fraud"):

    # Encode categorical features
    category_encoded = le_category.transform([category])[0]
    merchant_encoded = le_merchant.transform([merchant])[0]
    gender_encoded = le_gender.transform([gender])[0]

    # Calculate distance
    distance = hav([lat], [long], [merch_lat], [merch_long])[0]

    # ✅ Correct feature order
    input_data = pd.DataFrame([[
        cc_num, merchant_encoded, category_encoded, amt, gender_encoded,
        lat, long, merch_lat, merch_long, day, hour, month, distance
    ]], columns=[
        'cc_num', 'merchant', 'category', 'amt', 'gender',
        'lat', 'long', 'merch_lat', 'merch_long',
        'day', 'hour', 'month', 'distance'
    ])

    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Display result
    if prediction == 1:
        st.error(f"⚠️ Fraud Detected! (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Legitimate Transaction (Probability of fraud: {probability:.2f})")
