import pandas as pd
import joblib

similarity_cosine = joblib.load(open("./dataset_pkl/cosine_similarity.pkl", 'rb'))
movie_data = pd.read_csv('./dataset_pkl/preprocessed_data_content_based.csv')

def content_based_recommendation(movie_title):

    movie_title = movie_title.strip().lower()

    movie_match = movie_data[movie_data['title'].str.lower() == movie_title]

    if not movie_match.empty:
        id_of_movie = movie_match.index[0]
    else:
        raise ValueError(f"No recommendation found for Movie: '{movie_title}'")

    distances = similarity_cosine[id_of_movie]

    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:11]

    title_list=[]
    id_list=[]
    for i in movie_list:
        title_list.append(movie_data.iloc[i[0]].title)
        id_list.append(movie_data.iloc[i[0]].id)

    return(id_list, title_list)






