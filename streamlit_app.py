import streamlit as st
import pandas as pd
from src.data_loader import load_and_preprocess_data
from src.recommender import get_hybrid_recommendations

# --- Caching for performance ---
@st.cache_resource
def load_data_and_models():
    """Load data and models used in recommendation."""
    movies_df, user_movie_matrix, genre_cosine_sim, item_similarity, \
    movie_to_index, index_to_movie, cf_movie_to_index, cf_index_to_movie = \
        load_and_preprocess_data()
    return movies_df, user_movie_matrix, genre_cosine_sim, item_similarity, \
           movie_to_index, index_to_movie, cf_movie_to_index, cf_index_to_movie

# --- Load Data ---
movies_df, user_movie_matrix, genre_cosine_sim, item_similarity, \
movie_to_index, index_to_movie, cf_movie_to_index, cf_index_to_movie = load_data_and_models()

all_movie_titles = sorted(movies_df['title'].tolist())

# --- Page Configuration ---
st.set_page_config(
    page_title="Cinematch: Your AI Movie Recommender",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎬" # More prominent movie emoji
)

# --- Custom Blue & White Theme (via Markdown for limited control) ---
st.markdown(
    """
    <style>
    /* Main background and text */
    .stApp {
        background-color: #f0f8ff; /* Light Blue/Alice Blue */
        color: #1a1a2e; /* Dark Blue for general text */
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #003366; /* Deep Blue for headers */
    }

    /* Sidebar background */
    .st-emotion-cache-vk33gh, .st-emotion-cache-1fm0v3z { /* Targets sidebar */
        background-color: #e6f2ff; /* Slightly darker light blue for sidebar */
        border-right: 1px solid #cce0ff;
    }

    /* Selectbox, Slider, Button styling */
    .st-emotion-cache-1n76tmc, .st-emotion-cache-ue6hbm, .st-emotion-cache-sy3bdc { /* Selectbox label */
        color: #003366;
    }
    .st-emotion-cache-16idsd6 button { /* Button styling */
        background-color: #0056b3; /* Medium Blue */
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 18px;
        transition: background-color 0.3s ease;
    }
    .st-emotion-cache-16idsd6 button:hover {
        background-color: #004080; /* Darker blue on hover */
        color: white;
    }

    /* Info, Warning, Success boxes */
    .stAlert {
        border-radius: 8px;
    }
    .stAlert.info {
        background-color: #e0f2f7; /* Light cyan */
        color: #005f73; /* Darker cyan */
        border-left: 5px solid #007bb5;
    }
    .stAlert.warning {
        background-color: #fff3cd; /* Light yellow */
        color: #856404; /* Darker yellow */
        border-left: 5px solid #ffc107;
    }
    .stAlert.success {
        background-color: #d4edda; /* Light green */
        color: #155724; /* Darker green */
        border-left: 5px solid #28a745;
    }

    /* Custom movie card styling */
    .movie-card {
        background-color: #ffffff; /* White card background */
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 12px rgba(0,0,0,0.08); /* More pronounced shadow */
        border: 1px solid #e0f2f7; /* Subtle border */
        transition: transform 0.2s ease-in-out; /* Pop effect on hover */
    }
    .movie-card:hover {
        transform: translateY(-5px); /* Lift effect */
    }
    .movie-card h4 {
        margin:0; 
        color:#004080; /* Darker blue for movie titles */
        font-size: 1.25rem;
    }
    .movie-card p {
        color: #336699; /* Medium blue for genre/info */
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }

    /* Horizontal Rule */
    hr {
        border-top: 1px solid #cce0ff; /* Light blue border */
    }

    /* Slider specific styling */
    .stSlider > div > div > div:nth-child(2) { /* The track */
        background: #cce0ff;
    }
    .stSlider > div > div > div:nth-child(2) > div { /* The thumb */
        background: #0056b3;
    }
    </style>
    """, unsafe_allow_html=True
)


# --- Sidebar: User Inputs ---
with st.sidebar:
    st.title("🎬 Cinematch AI")
    st.markdown(
        "<p style='font-size: 1rem; color: #336699;'>Your intelligent movie recommendation engine.</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    selected_movie = st.selectbox(
        "✨ **Pick a movie you enjoyed:**",
        all_movie_titles,
        index=all_movie_titles.index("Toy Story (1995)") if "Toy Story (1995)" in all_movie_titles else 0,
        help="Select a movie to get personalized recommendations based on its style and what other users liked."
    )

    genre_weight = st.slider(
        "⚖️ **How much should genre influence recommendations?**",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Adjust the slider: 0.0 means recommendations are based heavily on what similar users watched. 1.0 means recommendations are primarily based on the genre of your selected movie. A value in between blends both approaches."
    )

    st.info("💡 **Tip**: Experiment with the slider! A higher genre weight (closer to 1.0) might give you movies that *feel* more similar in style, while a lower weight (closer to 0.0) will surface movies enjoyed by people with similar viewing habits to yours.")
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; font-size: 0.85rem; color: #6699cc;'>"
        "Developed with ❤️ by your Data Science Team"
        "</div>",
        unsafe_allow_html=True
    )

# --- Main Section ---
st.markdown("<h1 style='text-align: center; color: #003366; font-size: 3rem; margin-bottom: 0.5rem;'>🍿 Cinematch: Discover Your Next Favorite Film</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:1.15rem; color: #336699;'>We blend the best of both worlds—what users like you watch and the unique characteristics of movies—to bring you top-notch recommendations.</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Recommendation Button ---
st.markdown("<br>", unsafe_allow_html=True) # Add some space
if st.button("🚀 Get My Movie Recommendations!"):
    st.subheader(f"✨ Movies we think you'll love, similar to **_{selected_movie}_**:")
    st.write("") # Add a small gap

    with st.spinner("Crunching numbers and finding cinematic gems..."):
        recommendations = get_hybrid_recommendations(
            selected_movie,
            movies_df,
            user_movie_matrix,
            genre_cosine_sim,
            item_similarity,
            movie_to_index,
            index_to_movie,
            cf_movie_to_index,
            cf_index_to_movie,
            top_n=5,
            genre_weight=genre_weight
        )

    if recommendations and "Sorry" in recommendations[0]:
        st.warning(recommendations[0])
    elif recommendations:
        cols = st.columns(2) # Display in two columns for better layout
        for idx, movie_title in enumerate(recommendations):
            # Fetch genre for display
            movie_genre = movies_df[movies_df['title'] == movie_title]['genres'].iloc[0]
            with cols[idx % 2]:
                st.markdown(f"""
                    <div class='movie-card'>
                        <h4>🎬 {idx+1}. {movie_title}</h4>
                        <p>Genre: <em>{movie_genre}</em></p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("No recommendations found for this movie. Perhaps try a different movie or adjust the genre influence slider?")

st.markdown("<br><br>", unsafe_allow_html=True) # More space at the bottom

# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; font-size:1rem; color: #6699cc;'>"
    "📊 Built with <b>Python</b>, <b>Pandas</b>, <b>Scikit-learn</b>, and <b>Streamlit</b><br>"
    "© 2025 Cinematch. All rights reserved."
    "</div>",
    unsafe_allow_html=True
)