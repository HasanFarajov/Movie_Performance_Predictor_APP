import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load("movie_gross_predictor.pkl")

st.title("🎬 Movie Box Office Predictor")
st.write("Enter movie details to estimate expected gross revenue.")

# Input fields
budget = st.number_input("Budget ($)", min_value=0, step=1000000)
mpaa_rating = st.selectbox("MPAA Rating", ["G", "PG", "PG-13", "R", "NC-17"])
genre = st.selectbox("Genre", ["Action", "Comedy", "Drama", "Romance", "Horror", "Crime", "Adventure", "Animation", "Biography", "Fantasy", "Sci-Fi", "War"])
runtime = st.number_input("Runtime (minutes)", min_value=30, max_value=240)
rating = st.slider("IMDb Rating", 0.0, 10.0, 7.0, 0.1)
rating_count = st.number_input("Number of IMDb Ratings", min_value=0, step=1000)

# Predict button
if st.button("Predict Gross Revenue"):
    input_df = pd.DataFrame([{
        "Budget": budget,
        "MPAA Rating": mpaa_rating,
        "Genre": genre,
        "Runtime": runtime,
        "Rating": rating,
        "Rating Count": rating_count
    }])
    prediction = model.predict(input_df)
    st.success(f"🎉 Estimated Gross Revenue: ${prediction[0]:,.0f}")
