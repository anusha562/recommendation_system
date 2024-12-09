from model_checkpoints.content_based import content_based_recommendation
from model_checkpoints.collaborative_filerting import recommend_movies

# Hybrid Recommendation System
def hybrid_recommendation(movie_title):
    # Get content-based recommendations
    content_ids, content_titles = content_based_recommendation(movie_title.lower())

    # Get collaborative-based recommendations
    collab_ids, collab_titles = recommend_movies(movie_title)

    if collab_ids:
        combined_titles = content_titles + collab_titles
        combined_ids = content_ids + collab_ids
    else:
        combined_titles = content_titles
        combined_ids = content_ids

    return combined_ids, combined_titles



