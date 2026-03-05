import streamlit as st
import pandas as pd
import pickle

# -----------------------
# Page setup
# -----------------------

st.set_page_config(
    page_title="Booking Prediction",
    layout="wide"
)

# -----------------------
# Load model (cached)
# -----------------------

@st.cache_resource
def load_model():
    with open("booking_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("model_columns.pkl", "rb") as f:
        columns = pickle.load(f)

    return model, columns


model, model_columns = load_model()

st.title("✈️ British Airways Booking Prediction")

# -----------------------
# Layout columns
# -----------------------

col1, col2, col3 = st.columns(3)

# -----------------------
# Column 1
# -----------------------

with col1:

    st.subheader("Trip Info")

    purchase_lead = st.slider(
        "Days Before Flight",
        0, 365
    )

    flight_distance = st.number_input(
        "Flight Distance (km)",
        0
    )

# -----------------------
# Column 2
# -----------------------

with col2:

    st.subheader("Travel Details")

    length_of_stay = st.slider(
        "Length of Stay",
        1, 30
    )

    flight_hour = st.slider(
        "Flight Hour",
        0, 23
    )

# -----------------------
# Column 3
# -----------------------

with col3:

    st.subheader("Add-ons")

    baggage = st.selectbox(
        "Extra Baggage",
        [0,1]
    )

    seat = st.selectbox(
        "Preferred Seat",
        [0,1]
    )

    meal = st.selectbox(
        "In Flight Meal",
        [0,1]
    )

# -----------------------
# Feature engineering
# -----------------------

total_add_ons = baggage + seat + meal

input_data = pd.DataFrame({

    "purchase_lead":[purchase_lead],
    "flight_distance":[flight_distance],
    "length_of_stay":[length_of_stay],
    "flight_hour":[flight_hour],
    "total_add_ons":[total_add_ons]

})

# Match training columns
for col in model_columns:
    if col not in input_data:
        input_data[col] = 0

input_data = input_data[model_columns]

# -----------------------
# Prediction Section
# -----------------------

st.markdown("### Prediction")

center1, center2, center3 = st.columns([2,1,2])

with center2:

    if st.button("Predict Booking"):

        probability = model.predict_proba(input_data)[0][1]

        st.metric(
            "Booking Probability",
            f"{probability:.2f}"
        )

        if probability > 0.35:
            st.success("Likely to Book")

        elif probability > 0.20:
            st.warning("Moderate likelihood")

        else:
            st.error("Unlikely to Book")
