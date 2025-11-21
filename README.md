# Hybrid Movie Recommendation System

An intelligent movie recommendation system combining collaborative filtering and content-based approaches with adaptive weighting for personalized suggestions.

## Overview

**Type:** Personal project  
**Duration:** November 2024 (1 week)  
**Motivation:** Develop a practical machine learning application demonstrating recommendation system techniques

This project implements a hybrid recommendation engine that intelligently balances collaborative filtering (SVD) and content-based filtering (cosine similarity) to provide personalized movie recommendations. The system dynamically adjusts the weight between both approaches based on their respective accuracy metrics, resulting in more robust predictions.

Built with a Streamlit web interface for easy interaction and testing.

## Technologies Used

- **Language:** Python 3.10.12
- **ML Libraries:** Scikit-surprise (SVD), Scikit-learn (cosine similarity)
- **Data Processing:** Pandas, NumPy
- **Web Interface:** Streamlit
- **Dataset:** MovieLens 100K (100,000+ ratings, 1,682 movies)

## Key Features

### 1. Hybrid Recommendation Approach
- **Collaborative Filtering (SVD):** Learns from user rating patterns
- **Content-Based Filtering:** Uses movie genre similarity (cosine similarity)
- **Adaptive Weighting:** Dynamically adjusts α based on accuracy of each method

### 2. Intelligent Scoring System
```
Hybrid Score = α × CF_score + (1 - α) × CB_score
where α = acc_CF / (acc_CF + acc_CB)
```

### 3. Interactive Web Interface
- User-friendly Streamlit UI
- Adjustable number of recommendations (1-20)
- Real-time prediction display with estimated ratings

### 4. Robust Data Processing
- Handles 100,000+ movie ratings
- Filters unrated movies for recommendations
- Normalized content scores for fair comparison

## How It Works

### Collaborative Filtering (SVD)
Uses Singular Value Decomposition to find latent factors in user-movie rating patterns:
- Trained on 80% of ratings (80,000 reviews)
- Predicts ratings for unseen user-movie pairs
- RMSE-based accuracy measurement

### Content-Based Filtering
Recommends movies similar to those the user has rated highly:
- Genre-based movie similarity using cosine similarity matrix
- Weighted by user's actual ratings
- Calculates pairwise similarity between rated movies for accuracy

### Hybrid Integration
The system combines both approaches with dynamic weighting:
- If collaborative filtering performs better → higher α (more weight on CF)
- If content-based performs better → lower α (more weight on CB)
- Provides balanced recommendations leveraging both methods' strengths

## Dataset

**MovieLens 100K Dataset:** [[Source](https://grouplens.org/datasets/movielens/100k/)]
- **100,000 ratings** from **943 users** on **1682 movies**
- **Rating scale:** 1-5 stars (integer value)
- **Minimum ratings per user:** 20 movies
- **Timestamp:** Unix seconds since 1/1/1970 UTC

**Data files used:**

**`u.data`: - User Ratings:**
```
Format: user_id | movie_id | rating | timestamp
- Tab-separated values
- Randomly ordered
- Contains all 100,000 ratings
```

**`u.item` - Movie metadata:**
```
Format: movie_id | title | release_date | video_release_date | IMDb_URL | 
        unknown | Action | Adventure | Animation | Children's | Comedy | 
        Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | 
        Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western
- Tab-separated values
- Last 19 fields are binary genre indicators (1 = movie is of that genre, 0 = not)
- Movies can belong to multiple genres simultaneously
- Genre features used for content-based filtering via cosine similarity
```

**Citation:**
> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems (TiiS)* 5, 4, Article 19 (December 2015), 19 pages. https://doi.org/10.1145/2827872

## Project Structure

```
movie-recommender/
├── app.py              # Streamlit web interface
├── model.py            # Hybrid recommendation model
├── requirements.txt    # Python dependencies
├── data/
│   ├── u.data         # Movie ratings
│   └── u.item         # Movie metadata
└── README.md
```

## Setup & Usage

### **Installation**

```bash
# Clone repository
git clone https://github.com/goncalobmelo/hybrid-movie-recommendation-system.git
cd hybrid-movie-recommendation-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **Running the Application**

```bash
# Launch Streamlit app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### **Using the Recommender**

1. Enter a User ID (1 to max user ID in dataset)
2. Select number of recommendations (1-20)
3. Click "Get Recommendations"
4. View personalized movie suggestions with predicted ratings

## Results

- **SVD Model:** RMSE-based accuracy on test set
- **Hybrid Approach:** Balanced predictions combining user preferences and content similarity
- **Personalization:** Recommendations adapt based on individual user rating history

## Technical Highlights

### Adaptive Hybrid Weighting
Rather than using a fixed α value, the system dynamically calculates the optimal balance between collaborative and content-based filtering based on their performance metrics. This ensures the hybrid model leverages the stronger approach for each user.

### Efficient Similarity Computation
Pre-computed cosine similarity matrix for all movie pairs enables fast content-based recommendations without real-time calculation overhead.

### User-Centric Filtering
Only recommends movies the user hasn't rated, preventing redundant suggestions while maximizing discovery of new content.

## Skills Demonstrated

- **Machine Learning:** Collaborative filtering (SVD), content-based filtering, hybrid systems
- **Data Science:** Feature engineering, similarity metrics, model evaluation (RMSE)
- **Python Libraries:** Pandas, NumPy, Scikit-learn, Surprise
- **Web Development:** Streamlit for ML application deployment
- **Algorithm Design:** Hybrid scoring with adaptive weighting
- **Data Processing:** Efficient handling of 100K ratings

## Future Improvements

- [ ] Add more recommendation algorithms (e.g., neural collaborative filtering)
- [ ] Implement cold start handling for new users/movies
- [ ] Include movie posters and additional metadata
- [ ] Add explanation features (why this movie was recommended)
- [ ] Deploy to Streamlit Cloud for public access
- [ ] Implement user feedback loop to improve recommendations
- [ ] Add filtering options (genre, year, rating threshold)

## Contact

**Gonçalo Melo**  
goncalo.b.melo@gmail.com | [github.com/goncalobmelo](https://github.com/goncalobmelo)

---

*Built with Python, Surprise, and Streamlit. Dataset: MovieLens 100K.*
