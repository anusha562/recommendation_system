from model_checkpoints.s3 import load_data_from_s3
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

target_bucket = "recommendation-system-anusha"
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

movie_embeddings = load_data_from_s3(target_bucket,'movie_embeddings.csv')

movies = load_data_from_s3(target_bucket,'movies_llm.csv')

movie_embeddings_tensor = torch.tensor(movie_embeddings.values)

def get_llm_based_recommendations(query, top_n=5):
    query_embedding = model.encode([query])
    
    query_embedding_tensor = torch.tensor(query_embedding)
    
    query_embedding_numpy = query_embedding_tensor.numpy()

    similarities = cosine_similarity(query_embedding_numpy, movie_embeddings_tensor.numpy())[0]
    
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    recommendations = movies.iloc[top_indices]
    
    return recommendations[['title', 'overview', 'genres', 'id']]
