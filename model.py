import pandas as pd
from surprise import Dataset, Reader
from pandas import DataFrame
from surprise import SVD
from surprise.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.base import defaultdict
from surprise import accuracy

ratings = pd.read_csv("data/u.data", sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
ratings = ratings.drop('timestamp', axis=1)

movies = pd.read_csv("data/u.item", sep='|', encoding='latin-1', names=['movie_id','title'] + [f'col{i}' for i in range(22)])
movie_genres = movies.iloc[:, 5:]
movies_simple = movies[['movie_id','title']]

movie_index = {movie_id: idx for idx, movie_id in enumerate(movies['movie_id'])}

cosine_sim_matrix = cosine_similarity(movie_genres, movie_genres)

def recommend_similar_movies(movie_id, movies_df, cosine_sim_matrix, top_n):
   
    idx = movie_index[movie_id]
    
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1: top_n + 1]
    
    movie_indexes = [i[0] for i in sim_scores]

    return movies_df.iloc[movie_indexes][['movie_id', 'title']]

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)

train, test = train_test_split(data, test_size=0.2, random_state=24)

svd_model = SVD(random_state=24)
svd_model.fit(train)

predictions = svd_model.test(test)

rmse_svd = accuracy.rmse(predictions, verbose=False)
acc_cf = 1 / rmse_svd

def get_user_rating_based_recommendations(user_id, ratings_df, movies_df, model, n):

    rated_movies = ratings_df[ratings_df['user_id'] == user_id]['movie_id'].tolist()
    all_movies = movies_df['movie_id'].tolist()
    unrated_movies = [m for m in all_movies if m not in rated_movies]
    
    predictions = []
    for movie_id in unrated_movies:
        pred = model.predict(uid=user_id, iid=movie_id)
        predictions.append((movie_id, pred.est))
    
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    top_n = predictions[:n]
    top_n_with_titles = [(m_id, movies_df[movies_df['movie_id'] == m_id]['title'].values[0], rating) for m_id, rating in top_n]
    
    return top_n_with_titles

def content_accuracy(rated_movies, cosine_sim_matrix):
    ids = rated_movies['movie_id'].to_list()
    
    if len(rated_movies) < 2:
        return 0.0

    sims = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            idx_i = movie_index[ids[i]]
            idx_j = movie_index[ids[j]]
            sims.append(cosine_sim_matrix[idx_i][idx_j])

    return sum(sims) / len(sims)

def get_hybrid_recommendations(user_id, ratings_df, movies_df, model, cosine_sim_matrix, top_n):
    all_movies = movies_df['movie_id'].tolist()
    rated_movies = ratings_df[ratings_df['user_id'] == user_id]
    rated_movies_ids = rated_movies['movie_id'].tolist()    
    unrated_movies = [m for m in all_movies if m not in rated_movies_ids]

    content_scores = defaultdict(float)

    for _, row in rated_movies.iterrows():
        m_id = row['movie_id']
        rating = row['rating']
        idx = movie_index[m_id]

        sim_vector = cosine_sim_matrix[idx]

        for movie_id, score in zip(all_movies, sim_vector):
            content_scores[movie_id] += score * rating 

    if content_scores:
        max_content_score = max(content_scores.values())
    else:
        max_content_score = 1

    for m in content_scores:
        content_scores[m] /= max_content_score

    cf_scores = {m: model.predict(user_id, m).est for m in unrated_movies}

    acc_cb = content_accuracy(rated_movies, cosine_sim_matrix)

    alpha = acc_cf/(acc_cf + acc_cb)

    hybrid_scores = {}
    for m in unrated_movies:
        cb = content_scores.get(m, 0)
        cf = cf_scores[m]
        hybrid_scores[m] = alpha * cf + (1 - alpha) * cb

    sorted_movies = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    top = sorted_movies[:top_n]

    movie_titles = dict(zip(movies_df['movie_id'], movies_df['title']))

    results = []
    for m_id, score in top:
        if m_id not in rated_movies_ids:
            results.append((m_id, movie_titles[m_id], round(score, 2)))

    return results

print(get_hybrid_recommendations(24, ratings, movies, svd_model, cosine_sim_matrix, 5))
#print(recommend_similar_movies("Monty Python and the Holy Grail (1974)", movies_simple, cosine_sim_matrix, top_n=5))
#print(get_user_rating_based_recommendations(24, ratings, movies_simple, svd_model, 5))