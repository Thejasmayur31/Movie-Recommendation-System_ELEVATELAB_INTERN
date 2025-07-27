import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.utils import clean_title 

def load_and_preprocess_data(movies_path='data/movies.csv', ratings_path='data/ratings.csv'):
    """
    Loads MovieLens data, preprocesses it, and prepares matrices for recommendation.
    """
    movies_df = pd.read_csv(movies_path)
    ratings_df = pd.read_csv(ratings_path)

    
    movies_df["clean_title"] = movies_df["title"].apply(clean_title)
    movies_df["genres"] = movies_df["genres"].apply(lambda x: x.split("|"))
    movies_df['genres_str'] = movies_df['genres'].apply(lambda x: ' '.join(x))

   
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres_str'])
    genre_cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    
    movie_ratings = pd.merge(ratings_df, movies_df[['movieId', 'title']], on='movieId')
    user_movie_matrix = movie_ratings.pivot_table(index='userId', columns='title', values='rating').fillna(0)

    
    item_user_matrix = user_movie_matrix.T
    item_similarity = cosine_similarity(item_user_matrix)

    
    movie_to_index = {movie: i for i, movie in enumerate(movies_df['title'])}
    index_to_movie = {i: movie for i, movie in enumerate(movies_df['title'])}

    cf_movie_titles = user_movie_matrix.columns
    cf_movie_to_index = {movie: i for i, movie in enumerate(cf_movie_titles)}
    cf_index_to_movie = {i: movie for i, movie in enumerate(cf_movie_titles)}

    return movies_df, user_movie_matrix, genre_cosine_sim, item_similarity, \
           movie_to_index, index_to_movie, cf_movie_to_index, cf_index_to_movie