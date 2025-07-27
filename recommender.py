import pandas as pd
from src.utils import clean_title 

def get_hybrid_recommendations(movie_title, movies_df, user_movie_matrix,
                               genre_cosine_sim, item_similarity,
                               movie_to_index, index_to_movie,
                               cf_movie_to_index, cf_index_to_movie,
                               top_n=5, genre_weight=0.5):
    """
    Generates hybrid movie recommendations (collaborative + content-based).

    Args:
        movie_title (str): The title of the movie to get recommendations for.
        movies_df (pd.DataFrame): DataFrame containing movie information.
        user_movie_matrix (pd.DataFrame): User-item matrix (for CF).
        genre_cosine_sim (np.array): Cosine similarity matrix based on genres.
        item_similarity (np.array): Item-item similarity matrix based on ratings.
        movie_to_index (dict): Maps movie titles to their index in genre_cosine_sim.
        index_to_movie (dict): Maps index to movie titles from genre_cosine_sim.
        cf_movie_to_index (dict): Maps movie titles to their index in item_similarity.
        cf_index_to_movie (dict): Maps index to movie titles from item_similarity.
        top_n (int): Number of top recommendations to return.
        genre_weight (float): Weight for genre-based recommendations (0.0 to 1.0).
                              1.0 means purely genre-based, 0.0 means purely collaborative.

    Returns:
        list: A list of recommended movie titles.
    """
    
    cleaned_input_title = clean_title(movie_title)
    matching_movies = movies_df[movies_df['clean_title'].str.contains(cleaned_input_title, case=False, na=False)]

    if matching_movies.empty:
        return [f"Sorry, '{movie_title}' not found in our database. Please try another movie."]

    
    exact_match = movies_df[movies_df['title'] == movie_title]
    if not exact_match.empty:
        actual_movie_title = movie_title
    else:
        actual_movie_title = matching_movies.iloc[0]['title']
        
    movie_idx_genre = movie_to_index.get(actual_movie_title)
    if movie_idx_genre is None:
        
        print(f"Warning: Could not find '{actual_movie_title}' in genre index map.")
        genre_recommendations = []
        genre_scores_dict = {} 
    else:
        genre_scores = genre_cosine_sim[movie_idx_genre]
        genre_recommendations_raw = sorted(list(enumerate(genre_scores)), key=lambda x: x[1], reverse=True)
        
        genre_recommendations = [rec for rec in genre_recommendations_raw if index_to_movie[rec[0]] != actual_movie_title]
        genre_scores_dict = {index_to_movie[idx]: score for idx, score in genre_recommendations}


    
    cf_recommendations = []
    cf_scores_dict = {}

    if actual_movie_title not in cf_movie_to_index:
        print(f"Warning: Movie '{actual_movie_title}' not found in the collaborative filtering matrix. Relying on content-based only.")
    else:
        movie_idx_cf = cf_movie_to_index[actual_movie_title]
        cf_scores = item_similarity[movie_idx_cf]
        cf_recommendations_raw = sorted(list(enumerate(cf_scores)), key=lambda x: x[1], reverse=True)
       
        cf_recommendations = [rec for rec in cf_recommendations_raw if cf_index_to_movie[rec[0]] != actual_movie_title]
        cf_scores_dict = {cf_index_to_movie[idx]: score for idx, score in cf_recommendations}


    
    combined_scores = {}

    
    for movie, score in genre_scores_dict.items():
        combined_scores[movie] = score * genre_weight

    
    for movie, score in cf_scores_dict.items():
        
        combined_scores[movie] = combined_scores.get(movie, 0) + (score * (1 - genre_weight))

    
    sorted_recommendations = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

    
    top_recommendations = []
    seen_movies = set()
    for movie, score in sorted_recommendations:
        if movie not in seen_movies and movie != actual_movie_title: 
            top_recommendations.append(movie)
            seen_movies.add(movie)
        if len(top_recommendations) >= top_n:
            break
    return top_recommendations