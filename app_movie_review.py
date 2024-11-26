import streamlit as st
import requests
import pandas as pd
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from model_checkpoints.hybrid import hybrid_recommendation
from model_checkpoints.llm import get_llm_based_recommendations


# Initialize VADER SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

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

# Function to fetch reviews from TMDB API
def fetch_reviews(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key=2800d0e6c92ffc562ded93351b86ead3&language=en-US"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        reviews = []
        
        # Extract reviews from the response
        for review in data.get('results', []):
            # author = review.get('author')
            content = review.get('content')
            url = review.get('url')
            
            reviews.append({
                'content': content,
                'url': url
            })
        
        return reviews
    else:
        return []

# Sentiment analysis function using VADER
def analyze_sentiment_vader(review):
    sentiment_score = sid.polarity_scores(review)
    compound_score = sentiment_score['compound']
    
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Function to display movie reviews with sentiment analysis
def display_reviews_with_sentiment(movie_id):
    reviews = fetch_reviews(movie_id)
    
    if reviews:
        st.subheader("Reviews with Sentiment Analysis:")
        
        for review in reviews:
            # Analyze sentiment using VADER
            sentiment = analyze_sentiment_vader(review['content'])
            
            # Display review with sentiment
            st.write(f"**{review['author']}**: {sentiment} sentiment")
            st.write(f"Review: {review['content']}")
            st.markdown(f"[Read full review]({review['url']})")
            st.markdown("---")
    else:
        st.write("No reviews available for this movie.")

# Load external CSS file
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("./style/style.css")  # Call this function to load the CSS file

# Streamlit app setup
st.header("CineCraft - A Personalized Movie Recommendation System")
movie_data = pd.read_csv('./dataset/movies_metadata.csv')
movie_list = movie_data['title'].values

# Select a movie
selected_movie = st.selectbox("Select a movie:", movie_list)

# Hybrid recommendation (You should have your hybrid_recommendation function from earlier)
from model_checkpoints.hybrid import hybrid_recommendation

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
                    # HTML block with CSS classes for displaying movie poster and name
                    st.markdown(
                        f"<div class='movie-container'><img src='{poster_url}'><br><span class='movie-title'>{movie_title}</span><br>(ID: {movie_id})</div>",
                        unsafe_allow_html=True,
                    )
                
                # Call the display_reviews_with_sentiment function to show reviews for each recommended movie
                display_reviews_with_sentiment(movie_id)


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