from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from model_checkpoints.s3 import load_data_from_s3

target_bucket = "recommendation-system-anusha"

movie_features_read = load_data_from_s3(target_bucket, "movie_features.csv",index_col=0)
movie_data =load_data_from_s3(target_bucket,'movies_metadata.csv')

def recommend_movies(movie_name, n_recommendations=5):
    """
    Recommend movies based on a given movie name using item-based collaborative filtering.

    Parameters:
    - movie_name (str): The name of the movie for which recommendations are to be made.
    - n_recommendations (int): The number of similar movies to recommend.

    Returns:
    - Two lists:
      1. A list of recommended movie titles.
      2. A list of recommended movie IDs.
    """

    if movie_name not in movie_features_read.index:
        print(f"Movie '{movie_name}' not found in the dataset.")
        return 0,""

    movie_index = movie_features_read.index.get_loc(movie_name)

    movie_features_matrix = csr_matrix(movie_features_read.values)

    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')

    model_knn.fit(movie_features_matrix)

    distances, indices = model_knn.kneighbors(movie_features_read.iloc[movie_index,:].values.reshape(1, -1), n_recommendations + 1)

    movie_titles = []
    movie_ids = []
    for i in range(1, len(distances.flatten())):
        similar_movie_title = movie_features_read.index[indices.flatten()[i]]
        similar_movie_id = movie_data[movie_data['title'] == similar_movie_title]['id'].values[0]  # Get movie ID from movie_data
        movie_titles.append(similar_movie_title)
        movie_ids.append(similar_movie_id)

    return movie_ids, movie_titles
