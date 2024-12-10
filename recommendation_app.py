import streamlit as st
import pandas as pd
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from model_checkpoints.hybrid import hybrid_recommendation
from model_checkpoints.llm import get_llm_based_recommendations
import base64
from streamlit_option_menu import option_menu
from streamlit_extras.let_it_rain import rain
from model_checkpoints.s3 import load_data_from_s3

# Load custom CSS file
with open('./style/style.css') as f:
    css = f.read()

st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def set_background(image_path_or_url, target="app"):
    """
    Sets the background image of the Streamlit app or sidebar.
    Supports both image URLs and local paths (Base64 encoded).

    Parameters:
        image_path_or_url (str): URL or local path of the image.
        target (str): "app" for the main app background, "sidebar" for the sidebar background.
    """
    if image_path_or_url.startswith("http"):
        background_style = f"""
        <style>
        [data-testid="stSidebar"] {{
            background: url("{image_path_or_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }} 
        </style>
        """ if target == "sidebar" else f"""
        <style>
        .stApp {{
            background: url("{image_path_or_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """
    else:
        # Encode the local image to Base64
        with open(image_path_or_url, "rb") as img_file:
            encoded_img = base64.b64encode(img_file.read()).decode()
        background_style = f"""
        <style>
        [data-testid="stSidebar"] {{
            background: url("data:image/jpeg;base64,{encoded_img}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }} 
        </style>
        """ if target == "sidebar" else f"""
        <style>
        .stApp {{
            background: url("data:image/jpeg;base64,{encoded_img}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """
    
    st.markdown(background_style, unsafe_allow_html=True)

# S3 Bucket and File details
target_bucket = "recommendation-system-anusha"
s3_file_path = "movies_metadata.csv"
TMDB_API_KEY = "2800d0e6c92ffc562ded93351b86ead3"

# Initialize Sentiment Analyzer
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
    return "https://via.placeholder.com/500x750?text=No+Image+Available"

# Function to fetch trending movies or TV shows
def fetch_trending(media_type="movie", time_window="day"):
    url = f"https://api.themoviedb.org/3/trending/{media_type}/{time_window}?api_key={TMDB_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("results", [])
    return []

# Function to fetch trailer URL
def fetch_trailer(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={TMDB_API_KEY}&language=en-US"
    response = requests.get(url)
    if response.status_code == 200:
        videos = response.json().get("results", [])
        for video in videos:
            if video.get("type") == "Trailer" and video.get("site") == "YouTube":
                return f"https://www.youtube.com/watch?v={video['key']}"
    return None


# Function to fetch movie genres
def fetch_genres(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=2800d0e6c92ffc562ded93351b86ead3&language=en-US"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        genres = [genre['name'] for genre in data.get('genres', [])]
        return ', '.join(genres)
    return "Unknown Genre"

# Function to fetch reviews from TMDB API
def fetch_reviews(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key=2800d0e6c92ffc562ded93351b86ead3&language=en-US"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        reviews = [
            {"content": review["content"], "url": review["url"]}
            for review in data.get("results", [])
        ]
        return reviews
    return []

# Function to fetch movie details like genre, duration, and TMDB score
def fetch_movie_details(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=2800d0e6c92ffc562ded93351b86ead3&language=en-US"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        genres = [genre['name'] for genre in data.get('genres', [])]
        genre_string = ', '.join(genres) if genres else "Unknown Genre"
        duration = data.get('runtime', 'Unknown')  # Duration in minutes
        tmdb_score = data.get('vote_average', 'Not Rated')  # TMDB score (rating)
        return genre_string, duration, tmdb_score
    return "Unknown Genre", "Unknown Duration", "Not Rated"

def fetch_trending_by_language(media_type="movie", time_window="day", original_language=None):
    url = f"https://api.themoviedb.org/3/trending/{media_type}/{time_window}?api_key={TMDB_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        movies = response.json().get("results", [])
        # Filter movies by original_language if specified
        if original_language:
            return [movie for movie in movies if movie.get("original_language") == original_language]
        return movies
    return []

# Sentiment analysis function
def analyze_sentiment_vader(review):
    sentiment_score = sid.polarity_scores(review)
    compound_score = sentiment_score['compound']
    if compound_score >= 0.05:
        return 'Good'
    elif compound_score <= -0.05:
        return 'Bad'
    else:
        return 'Neutral'

# Sidebar Layout with Custom Box and Dark Background
# Custom CSS for styling
st.markdown("""
    <style>
    .css-1d391kg {
        background-color: #C70039 !important;  /* Set the sidebar background color */
    }
    .css-ffhzg2 {
        color: #ffffff !important;  /* Set text color to white for better contrast */
    }
    .css-ffhzg2 .st-bc {
        background-color: #C70039;  /* Set the background color for the button elements */
        color: #ffffff;  /* Button text color */
    }
    .css-ffhzg2 .st-bc:hover {
        background-color: #99002b;  /* Darker shade on hover */
        color: #ffffff;  /* Button text color */
    }
    .css-ffhzg2 .st-bc-active {
        background-color: #99002b;  /* Darker shade for active state */
        color: #ffffff;  /* Button text color */
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    set_background("./style/sidebar.png", target="sidebar")

    selected = option_menu(
            menu_title="MENU",
            options=["Homepage 🏠", "Get Recommendations 🎬", "LLM Query Search 💬", "Trending Movies 🔥"],  # Menu items
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {
                    "padding": "5px",
                    "background-color": "#0E1117",
                    "font-family": "Monospace"
                },
                "nav-link": {
                    "font-size": "18px",
                    "text-align": "left",
                    "margin": "0px",
                    "font-family": "Monospace"
                },
                "nav-link-selected": {
                    "background-color": "#90EE90"
                }
            }
        )
# Load Movie Data
movie_data = load_data_from_s3(target_bucket,s3_file_path)
movie_list = movie_data['title'].values

# Homepage
if selected == "Homepage 🏠":
    set_background("./style/bgi.jpg", target="app")
    st.markdown('<h1 class="homepage-header">Welcome to CineCraft!</h1>', unsafe_allow_html=True)
    
    # Open and encode the GIF file
    with open("./style/home.gif", "rb") as file_:
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")

    # Display the subheader
    st.markdown('<h2 class="homepage-subheader">Your Personalized Movie Recommendation System</h2>', unsafe_allow_html=True)
    
    # Display the list and image separately
    st.markdown(
        '<ul class="homepage-features">'
        '<li>Get recommendations using hybrid algorithms.</li>'
        '<li>Ask questions and get recommendations powered by LLM.</li>'
        '<li>View reviews and sentiments for movies.</li>'
        '</ul>',
        unsafe_allow_html=True
    )
    
    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="home gif" style="width: 250px; height: auto; float: right; margin-right: 15px;">',
        unsafe_allow_html=True
    )

    
# Hybrid Recommendation
elif selected == "Get Recommendations 🎬":
    st.markdown('<h2 class="recommendation-header">Hybrid Recommendation System 🎬</h2>', unsafe_allow_html=True)
    selected_movie = st.selectbox("Select a movie:", movie_list)

    if st.button('Show Recommendations'):
        movie_ids, recommended_titles = hybrid_recommendation(selected_movie)

        recommended_movies = []
        for movie_id, movie_title in zip(movie_ids, recommended_titles):
            poster_url = fetch_poster(movie_id)
            genre, duration, tmdb_score = fetch_movie_details(movie_id)
            reviews = fetch_reviews(movie_id)

            review_data = []
            for review in reviews[:3]:
                sentiment = analyze_sentiment_vader(review["content"])
                review_data.append({
                    "Review": review["content"][:300] + '...',
                    "Sentiment": sentiment,
                    "URL": review["url"]
                })

            recommended_movies.append({
                "Movie": movie_title,
                "Poster": poster_url,
                "Genre": genre,
                "Duration": duration,
                "TMDB Score": tmdb_score,
                "Reviews": review_data
            })
        st.markdown('<h2 class="recommendation-header">Recommended Movies</h2>', unsafe_allow_html=True)
        rain(
            emoji="🍿",
            font_size=54,
            falling_speed=2,
            animation_length= 1
        )
        
        for i in range(0, len(recommended_movies), 5):
            cols = st.columns(5)
            for idx, movie in enumerate(recommended_movies[i:i+5]):
                with cols[idx]:
                    st.image(movie["Poster"], width=150)
                    st.markdown(f'<p class="movie-title">{movie["Movie"]}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="genre-info">Genre: {movie["Genre"]}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="genre-info">Duration: {movie["Duration"]} mins</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="genre-info">TMDB Score: {movie["TMDB Score"]}</p>', unsafe_allow_html=True)

        movie_reviews = []

        for movie in recommended_movies:
            for review in movie["Reviews"]:
                movie_reviews.append({
                    "Movie": movie["Movie"],
                    "Review": review["Review"],
                    "Sentiment": review["Sentiment"]
                })

        movie_reviews_df = pd.DataFrame(movie_reviews)

        st.markdown('<div class="table-container">', unsafe_allow_html=True)
        st.dataframe(movie_reviews_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

# LLM Recommendation
elif selected == "LLM Query Search 💬":
    st.markdown('<h2 class="recommendation-header">LLM Query Search 💬</h2>', unsafe_allow_html=True)
    query = st.text_input("Ask a question or type a search query:")

    if st.button('Search'):
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
                        st.image(poster_url, use_container_width=True, caption=movie_title)
# Trending Today
elif selected == "Trending Movies 🔥":
    st.title("🎥 Welcome to Trending Movies and Shows")
    st.markdown("Millions of movies, TV shows, and people to discover. Explore now.")

    st.subheader("🔥 Trending Today")
    trending_today = fetch_trending(media_type="movie", time_window="day")

    cols = st.columns(5)
    for idx, movie in enumerate(trending_today[:5]):
        with cols[idx]:
            poster_path = movie.get("poster_path", "")
            title = movie.get("title", "Unknown")
            trailer_url = fetch_trailer(movie.get("id"))
            
            if poster_path:
                st.image(f"https://image.tmdb.org/t/p/w500{poster_path}", use_container_width=True)
            st.markdown(f"**{title}**")
            if trailer_url:
                st.markdown(f'[🎬 Watch Trailer]({trailer_url})', unsafe_allow_html=True)    