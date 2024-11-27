import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

movie_embeddings = pd.read_csv('/Users/anushaanandhan/Downloads/CineCraft/dataset_pkl/movie_embeddings.csv')

movies = pd.read_csv('/Users/anushaanandhan/Downloads/CineCraft/dataset_pkl/movies_llm.csv')

# Convert to tensor for faster computation
movie_embeddings_tensor = torch.tensor(movie_embeddings.values)

def get_llm_based_recommendations(query, top_n=5):
    # Encode the query
    query_embedding = model.encode([query])
    
    # Convert the query embedding to tensor
    query_embedding_tensor = torch.tensor(query_embedding)
    
    # Convert tensor to NumPy array for cosine similarity calculation
    query_embedding_numpy = query_embedding_tensor.numpy()

    # Calculate cosine similarity between the query embedding and all movie embeddings
    similarities = cosine_similarity(query_embedding_numpy, movie_embeddings_tensor.numpy())[0]
    
    # Get the indices of the top N similar movies
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    # Get the recommended movies from the movies DataFrame (assuming 'movies' DataFrame exists)
    recommendations = movies.iloc[top_indices]
    
    # Return a subset of relevant columns (title, overview, genres)
    return recommendations[['title', 'overview', 'genres', 'id']]
