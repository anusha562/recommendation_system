from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import pandas as pd

# Load your movie features (Make sure the dataset includes movie titles and IDs)
movie_features_read = pd.read_csv('./dataset_pkl/movie_features.csv',index_col=0)
movie_data = pd.read_csv('./dataset_pkl/movies_metadata.csv')  # Assuming this contains 'id' and 'title'

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

    # Ensure the movie name exists in the movie_features_df
    if movie_name not in movie_features_read.index:
        print(f"Movie '{movie_name}' not found in the dataset.")
        return 0,""

    # Get the movie index for the input movie
    movie_index = movie_features_read.index.get_loc(movie_name)

    # Create a sparse matrix of movie features
    movie_features_matrix = csr_matrix(movie_features_read.values)

    # Initialize NearestNeighbors model
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')

    # Fit the model on the movie features matrix
    model_knn.fit(movie_features_matrix)

    # Find the top 'n_recommendations' most similar movies to the given movie
    distances, indices = model_knn.kneighbors(movie_features_read.iloc[movie_index,:].values.reshape(1, -1), n_recommendations + 1)

    # Prepare recommendations (skip the first movie, as it's the input movie itself)
    movie_titles = []
    movie_ids = []
    for i in range(1, len(distances.flatten())):
        similar_movie_title = movie_features_read.index[indices.flatten()[i]]
        similar_movie_id = movie_data[movie_data['title'] == similar_movie_title]['id'].values[0]  # Get movie ID from movie_data
        movie_titles.append(similar_movie_title)  # Add title to the list
        movie_ids.append(similar_movie_id)  # Add ID to the list

    return movie_ids, movie_titles  # Return two separate lists: one for titles, one for IDs
