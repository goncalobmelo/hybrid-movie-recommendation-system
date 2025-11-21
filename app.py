from model import svd_model, ratings, movies_simple, cosine_sim_matrix, get_hybrid_recommendations
import streamlit as st
from surprise import Dataset, Reader, SVD

@st.cache_data
def train_svd(ratings_df):
    reader = Reader(rating_scale=(1,5))
    data = Dataset.load_from_df(ratings_df[['user_id','movie_id','rating']], reader)
    trainset = data.build_full_trainset()
    svd = SVD()
    svd.fit(trainset)
    return svd

svd_model = train_svd(ratings)

st.title("Movie Recommendation System")

max_user_id = ratings['user_id'].max()
user_id = st.number_input("Enter User ID: ", min_value=1, max_value=max_user_id)

n = st.slider("Number of recommendations:", min_value=1, max_value=20, value=5)

if user_id not in ratings['user_id'].values:
    st.warning(f"User ID {user_id} not found in dataset.")
    st.stop()

if st.button("Get Recommendations"):
    top_rec = get_hybrid_recommendations(user_id, ratings, movies_simple, svd_model, cosine_sim_matrix, n)
    for i, (m_id, title, rating) in enumerate(top_rec, 1):
        st.write(f"{i}. {title} (predicted rating: {rating:.2f})")