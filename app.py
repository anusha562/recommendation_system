import streamlit as st
from model_checkpoints.hybrid import hybrid_recommendation
from model_checkpoints.llm import get_llm_based_recommendations
import requests
import pandas as pd

# Load external CSS file
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("./style/style.css")  # Call this function to load the CSS file

# Function to fetch movie poster
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=2800d0e6c92ffc562ded93351b86ead3&language=en-US"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    # Return a placeholder image if no poster is available
    return "https://via.placeholder.com/500x750?text=No+Image+Available"

# Streamlit app setup
st.header("CineCraft - A Personalized Movie Recommendation System")
movie_data = pd.read_csv('./dataset_pkl/movies_metadata.csv')
movie_list = movie_data['title'].values

# Select a movie
selected_movie = st.selectbox("Select a movie:", movie_list)

if st.button('Show Recommendations'):
    movie_ids, recommended_titles = hybrid_recommendation(selected_movie)
    st.subheader("Recommended Movies:")
    for i in range(0, len(recommended_titles), 5):
        cols = st.columns(5)
        for j, col in enumerate(cols):
            if i + j < len(recommended_titles):
                movie_title = recommended_titles[i + j]
                movie_id = movie_ids[i + j]
                poster_url = fetch_poster(movie_id)
                with col:
                    # HTML block with CSS classes
                    st.markdown(
                        f"<div class='movie-container'><img src='{poster_url}'><br><span class='movie-title'>{movie_title}</span></div>",
                        unsafe_allow_html=True,
                    )

#QUERY SEARCH FOR LLM MODEL
query = st.text_input("Ask a question or type a search query:", "")
if st.button('Search'):
    # Get the top N recommendations based on the query
    recommendations = get_llm_based_recommendations(query)
    st.subheader("Recommended Movies:")

    for i in range(0, len(recommendations), 5):
        cols = st.columns(5)
        for j, col in enumerate(cols):
            if i + j < len(recommendations):
                movie_title = recommendations.iloc[i + j]['title']
                movie_id = recommendations.iloc[i + j]['id']
                poster_url = fetch_poster(movie_id)
                with col:
                    # HTML block with CSS classes
                    st.markdown(
                        f"<div class='movie-container'><img src='{poster_url}'><br><span class='movie-title'>{movie_title}</span><br>(ID: {movie_id})</div>",
                        unsafe_allow_html=True,
                    )

