import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functools
import urllib.parse
from PIL import Image
import requests
import io
import os


# Load models
basic_model = joblib.load(os.path.join("models", "movie_gross_predictor.pkl"))
advanced_model = joblib.load(os.path.join("models", "movie_gross_predictor_v2.pkl"))
# basic_model = joblib.load("movie_gross_predictor.pkl")
# advanced_model = joblib.load("movie_gross_predictor_v2.pkl")

# Load additional datasets
actors_df = pd.read_csv(os.path.join('data', 'actors.csv'))
movies_actors_df = pd.read_csv(os.path.join('data', 'moviesactors.csv'))
# actors_df = pd.read_csv("D:/ASOIU MASTER/II kurs II sem/Fuzzy/midterm/movies/actors.csv")
# movies_actors_df = pd.read_csv("D:/ASOIU MASTER/II kurs II sem/Fuzzy/midterm/movies/moviesactors.csv")

st.set_page_config(layout="wide")
st.title("\U0001F3AC Movie Box Office Predictor")
mode = st.radio("Select prediction mode:", ["Basic", "Advanced with Actors"])

@functools.lru_cache(maxsize=128)
def get_cached_wiki_summary(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            return {}
    except:
        return {}

@functools.lru_cache(maxsize=128)
def get_cached_image(url):
    return requests.get(url)

if mode == "Basic":
    st.subheader("\U0001F3AF Basic Prediction")
    with st.expander("Movie Input Fields", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            budget = st.number_input("Budget ($)", min_value=0, step=1000000)
            mpaa_rating = st.selectbox("MPAA Rating", ["G", "PG", "PG-13", "R", "NC-17"])
            genre = st.selectbox("Genre", ["Action", "Comedy", "Drama", "Romance", "Horror", "Crime", "Adventure", "Animation", "Biography", "Fantasy", "Sci-Fi", "War"])
        with col2:
            runtime = st.number_input("Runtime (minutes)", min_value=30, max_value=240)
            rating = st.slider("IMDb Rating", 0.0, 10.0, 7.0, 0.1)
            rating_count = st.number_input("Number of IMDb Ratings", min_value=0, step=1000)

    if st.button("Predict Gross Revenue", key="basic"):
        input_df = pd.DataFrame([{
            "Budget": budget,
            "MPAA Rating": mpaa_rating,
            "Genre": genre,
            "Runtime": runtime,
            "Rating": rating,
            "Rating Count": rating_count
        }])
        prediction = basic_model.predict(input_df)
        st.success(f"\U0001F389 Estimated Gross Revenue: ${prediction[0]:,.0f}")

elif mode == "Advanced with Actors":
    st.subheader("\U0001F3AD Advanced Prediction with Actors")
    st.write("Select actors and enter movie details to improve the prediction.")

    with st.sidebar:
        st.markdown("### \U0001F3AD Actor Selection")
        gender_filter = st.selectbox("Filter by Gender", ["All", "Male", "Female"])
        country_filter = st.selectbox("Filter by Country", ["All"] + sorted(actors_df['Birth Country'].dropna().unique().tolist()))

        filtered_actors = actors_df.copy()
        if gender_filter != "All":
            filtered_actors = filtered_actors[filtered_actors['Gender'] == gender_filter]
        if country_filter != "All":
            filtered_actors = filtered_actors[filtered_actors['Birth Country'] == country_filter]

        actor_names = filtered_actors['Name'].dropna().unique()
        if "selected_actors" not in st.session_state:
            st.session_state.selected_actors = []

        to_add = st.multiselect("Select Actors", actor_names, default=st.session_state.selected_actors, key="Select Actors")
        for actor in to_add:
            if actor not in st.session_state.selected_actors:
                st.session_state.selected_actors.append(actor)

        selected_actors = st.session_state.selected_actors.copy()

        if selected_actors:
            if st.button("\U0001F9F9 Clear All Actors"):
                st.session_state.selected_actors.clear()
                st.session_state["Select Actors"] = []

            st.markdown("---")
            for i, name in enumerate(selected_actors):
                actor_row = actors_df[actors_df['Name'] == name]
                if actor_row.empty:
                    continue
                row = actor_row.iloc[0]
                colA, colB = st.columns([4, 1])
                with colA:
                    search_name = urllib.parse.quote(name)
                    wiki_api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{search_name}"
                    response = get_cached_wiki_summary(wiki_api_url)
                    birthday = row['Date of Birth'] if pd.notna(row['Date of Birth']) else response.get('birth_date', 'N/A')
                    gender = row['Gender'] if pd.notna(row['Gender']) else response.get('description', 'N/A')
                    height = round(row['Height (Inches)'] / 10, 1) if pd.notna(row['Height (Inches)']) else 'N/A'
                    birthplace = f"{row['Birth City'] if pd.notna(row['Birth City']) else ''}, {row['Birth Country'] if pd.notna(row['Birth Country']) else ''}".strip(', ')
                    if not birthplace or birthplace == ',':
                        birthplace = response.get('description', 'N/A')
                    networth = row['NetWorth'] if pd.notna(row['NetWorth']) else 'N/A'
                    bio = row['Biography'] if pd.notna(row['Biography']) else response.get('extract', 'N/A')

                    with st.expander(f"{name} - Profile Details", expanded=False):
                        st.markdown(f"## {name}")
                        if 'thumbnail' in response and 'source' in response['thumbnail']:
                            st.image(response['thumbnail']['source'], width=200)
                        st.markdown(f"**Birthday:** {birthday}")
                        st.markdown(f"**Gender:** {gender}")
                        st.markdown(f"**Height:** {height} inches")
                        st.markdown(f"**Birthplace:** {birthplace}")
                        st.markdown(f"**Net Worth:** {networth}")
                        st.markdown(f"**Biography:**\n\n{bio if isinstance(bio, str) else 'No biography available.'}")

                with colB:
                    if st.button(f"❌", key=f"remove_{i}"):
                        st.session_state.selected_actors.remove(name)
                        st.experimental_rerun()

    actors_df.to_csv("enriched_actors.csv", index=False)

    with st.expander("Movie Input Fields", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            budget = st.number_input("Budget ($)", min_value=0, step=1000000, key="adv_budget")
            mpaa_rating = st.selectbox("MPAA Rating", ["G", "PG", "PG-13", "R", "NC-17"], key="adv_rating")
            genre = st.selectbox("Genre", ["Action", "Comedy", "Drama", "Romance", "Horror", "Crime", "Adventure", "Animation", "Biography", "Fantasy", "Sci-Fi", "War"], key="adv_genre")
        with col2:
            runtime = st.number_input("Runtime (minutes)", min_value=30, max_value=240, key="adv_runtime")
            rating = st.slider("IMDb Rating", 0.0, 10.0, 7.0, 0.1, key="adv_imdb")
            rating_count = st.number_input("Number of IMDb Ratings", min_value=0, step=1000, key="adv_count")

    if st.button("Predict Gross Revenue", key="advanced"):
        actor_info = actors_df[actors_df['Name'].isin(selected_actors)]
        num_actors = len(actor_info)
        avg_height = actor_info['Height (Inches)'].mean() if not actor_info['Height (Inches)'].isna().all() else 0
        male_ratio = (actor_info['Gender'] == 'Male').sum() / num_actors if num_actors else 0

        input_df = pd.DataFrame([{
            "Budget": budget,
            "MPAA Rating": mpaa_rating,
            "Genre": genre,
            "Runtime": runtime,
            "Rating": rating,
            "Rating Count": rating_count,
            "Actor Count": num_actors,
            "Avg Height": avg_height,
            "Male Ratio": male_ratio
        }])

        prediction = advanced_model.predict(input_df)
        st.success(f"\U0001F389 Estimated Gross Revenue: ${prediction[0]:,.0f}")

        if not actor_info.empty:
            st.markdown("### \U0001F4CA Actor Group Stats")
            col1, col2 = st.columns(2)
            with col1:
                gender_counts = actor_info['Gender'].value_counts()
                fig1, ax1 = plt.subplots(figsize=(2.5, 2.5))
                ax1.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
                ax1.axis('equal')
                st.pyplot(fig1)
            with col2:
                fig2, ax2 = plt.subplots(figsize=(3.5, 3))
                actor_info['Height (Inches)'].dropna().plot(kind='hist', bins=10, ax=ax2)
                ax2.set_xlabel('Height (Inches)')
                ax2.set_ylabel('Frequency')
                st.pyplot(fig2)

        
