import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re
import joblib as jb
import os

metadata_path = "./dataset_pkl/movies_metadata.csv"
movies_metadata = pd.read_csv(metadata_path, low_memory=False)

movies = movies_metadata[['id', 'title', 'overview', 'genres', 'release_date', 'vote_average', 'original_language', 'poster_path', 'imdb_id']]
movies = movies.dropna(subset=['overview', 'genres', 'release_date'])
movies['genres'] = movies['genres'].apply(eval).apply(lambda x: [genre['name'] for genre in x])
movies['release_year'] = movies['release_date'].apply(lambda x: str(x).split("-")[0])
movies['text'] = (
    movies['title'] + " " +
    movies['overview'] + " " +
    movies['genres'].apply(lambda x: ' '.join(x)) + " " +
    movies['release_year']
)

# Load pre-trained embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embedding_file = './dataset_pkl/movie_embeddings.pkl'

if os.path.exists(embedding_file):
    movie_embeddings_tensor = jb.load(embedding_file)
else:
    movie_embeddings = model.encode(movies['text'].tolist(), show_progress_bar=True)
    movie_embeddings_tensor = torch.tensor(movie_embeddings)
    jb.dump(movie_embeddings_tensor, embedding_file)

def fetch_movie_details(row):
    poster_url = f"https://image.tmdb.org/t/p/w500{row['poster_path']}" if pd.notna(row['poster_path']) else "https://via.placeholder.com/500x750?text=No+Image+Available"
    return {
        "Title": row["title"],
        "Overview": row["overview"],
        "Genres": ", ".join(row["genres"]),
        "Release Year": row["release_year"],
        "IMDb Score": row["vote_average"],
        "Poster": poster_url,
        "Trailer Link": f"https://www.youtube.com/results?search_query={row['title']} trailer"
    }

def parse_query(query):
    genres = re.findall(r'(action|comedy|drama|thriller|romance|horror|animation|adventure)', query.lower())
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', query)
    year = year_match.group(0) if year_match else None
    return genres, year

def get_recommendations(query, top_n=5):
    genres, year = parse_query(query)
    
    filtered_movies = movies
    if genres:
        filtered_movies = filtered_movies[filtered_movies['genres'].apply(lambda x: any(g.lower() in map(str.lower, x) for g in genres))]
    if year:
        filtered_movies = filtered_movies[filtered_movies['release_year'] == year]

    if filtered_movies.empty:
        return f"No movies found matching the criteria: Genres={genres}, Year={year}."

    filtered_movies = filtered_movies.reset_index(drop=True)

    filtered_indices = filtered_movies.index
    
    query_embedding = model.encode([query])
    query_embedding_tensor = torch.tensor(query_embedding)
    
    similarities = cosine_similarity(query_embedding_tensor.numpy(), movie_embeddings_tensor[filtered_indices].numpy())[0]

    top_indices = np.argsort(similarities)[-top_n:][::-1]
    top_movies = filtered_movies.iloc[top_indices]    

    recommendations = [fetch_movie_details(row) for _, row in top_movies.iterrows()]
    return recommendations


