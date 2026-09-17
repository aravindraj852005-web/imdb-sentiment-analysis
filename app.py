
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from database import create_database, add_review, get_reviews

# ---------------- CONFIG ----------------

st.set_page_config(
    page_title="Padam Epdi Irukku Bro?",
    page_icon="🎬",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
create_database()

# ---------------- CSS ----------------

st.markdown("""
<style>
.stApp {
    background: #0f1117;
    color: #ffffff;
}

[data-testid="stSidebar"] {
    background: #171923;
}

[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

.hero {
    padding: 35px;
    border-radius: 20px;
    background: linear-gradient(135deg, #241442, #171923);
    border: 1px solid #3d315b;
    margin-bottom: 30px;
}

.hero h1 {
    color: white;
    font-size: 42px;
    font-weight: 800;
}

.hero p {
    color: #c4bfd5;
    font-size: 17px;
}

.movie-card {
    background: #1b1e29;
    padding: 18px;
    border-radius: 18px;
    border: 1px solid #303442;
    margin-top: 10px;
    margin-bottom: 25px;
}

.movie-title {
    font-size: 23px;
    font-weight: 700;
    color: white;
}

.movie-info {
    color: #d0cede;
    font-size: 15px;
    margin-top: 8px;
}

.rating {
    color: #ffc107;
    font-size: 18px;
    font-weight: bold;
}

.section-title {
    font-size: 30px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 20px;
}

.review-card {
    background: #1b1e29;
    border-left: 4px solid #8b5cf6;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 14px;
}

.ai-positive {
    background: #123d2a;
    color: #4ade80;
    padding: 20px;
    border-radius: 12px;
    font-size: 22px;
    font-weight: bold;
}

.ai-negative {
    background: #451b25;
    color: #fb7185;
    padding: 20px;
    border-radius: 12px;
    font-size: 22px;
    font-weight: bold;
}

.footer {
    text-align: center;
    color: #888888;
    padding: 30px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------

movies_path = BASE_DIR / "movies.csv"
reviews_path = BASE_DIR / "reviews.csv"
models_path = BASE_DIR / "models"

movies_df = pd.read_csv(movies_path)
sample_reviews = pd.read_csv(reviews_path)

vectorizer = joblib.load(
    models_path / "tfidf_vectorizer.pkl"
)

model = joblib.load(
    models_path / "logistic_regression.pkl"
)

# ---------------- SIDEBAR ----------------

st.sidebar.markdown("## 🎬 Movie Menu")
st.sidebar.markdown("---")

movie_names = movies_df["title"].tolist()

selected_movie = st.sidebar.selectbox(
    "Choose a Movie",
    movie_names
)

selected_movie_data = movies_df[
    movies_df["title"] == selected_movie
].iloc[0]

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "⭐ Movie Reviews",
        "🤖 AI Sentiment",
        "📊 Model Performance"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("Powered by NLP & Machine Learning")

# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">
<h1>🎬 Padam Epdi Irukku Bro?</h1>
<p>Discover movies. Share reviews. Let AI understand your emotions.</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠 Home":
    
    # ---------------- DASHBOARD ----------------

    total_movies = len(movies_df)
    total_reviews = len(sample_reviews)

    dashboard_col1, dashboard_col2, dashboard_col3 = st.columns(3)

    with dashboard_col1:
        st.metric(
            "🎬 Total Movies",
            total_movies
        )

    with dashboard_col2:
        st.metric(
            "⭐ Total Reviews",
            total_reviews
        )

    with dashboard_col3:
        st.metric(
            "🤖 Model Accuracy",
            "88.85%"
        )

    st.markdown("---")

    st.markdown(
        '<div class="section-title">🔥 Explore Movies</div>',
        unsafe_allow_html=True
    )

    # Search bar

    search_text = st.text_input(
        "🔍 Search Movie",
        placeholder="Search Leo, Master, Vikram..."
    )

    # Genre filter

    genre_options = ["All"] + sorted(
        movies_df["genre"].unique().tolist()
    )

    selected_genre = st.selectbox(
        "🎭 Filter by Genre",
        genre_options
    )

    # Filter movies

    filtered_movies = movies_df.copy()

    if search_text.strip():

        filtered_movies = filtered_movies[
            filtered_movies["title"].str.contains(
                search_text.strip(),
                case=False,
                na=False
            )
        ]

    if selected_genre != "All":

        filtered_movies = filtered_movies[
            filtered_movies["genre"] == selected_genre
        ]

    st.markdown("---")

    st.write(
        f"🎬 **{len(filtered_movies)} movies found**"
    )

    if len(filtered_movies) == 0:

        st.warning("No movies found. Try another search.")

    else:

        cols = st.columns(3)

        for index, movie in filtered_movies.iterrows():

            with cols[index % 3]:

                movie_id = int(movie["movie_id"])

                db_reviews = get_reviews(movie_id)

                csv_reviews = sample_reviews[
                    sample_reviews["movie_id"] == movie_id
                ]

                ratings = list(csv_reviews["rating"])

                if db_reviews:

                    ratings.extend(
                        [review[3] for review in db_reviews]
                    )

                average_rating = (
                    sum(ratings) / len(ratings)
                    if ratings else 0
                )

                poster_url = movie.get(
                    "poster_url",
                    ""
                )

                if pd.notna(poster_url) and poster_url:

                    st.image(
                        poster_url,
                        use_container_width=True
                    )

                movie_html = f"""
<div class="movie-card">
<div class="movie-title">{movie["title"]}</div>
<div class="movie-info">📅 Year: {movie["year"]}</div>
<div class="movie-info">🎭 Genre: {movie["genre"]}</div>
<br>
<div class="rating">⭐ {average_rating:.1f}/5</div>
<div class="movie-info">💬 {len(ratings)} Reviews</div>
</div>
"""

                st.markdown(
                    movie_html,
                    unsafe_allow_html=True
                )

    st.markdown("---")

    st.markdown("### 💡 How It Works")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("### 🎬")
        st.write("**Choose a Movie**")
        st.caption("Explore movies and ratings.")

    with col2:

        st.markdown("### ⭐")
        st.write("**Share Your Review**")
        st.caption("Give ratings and write comments.")

    with col3:

        st.markdown("### 🤖")
        st.write("**AI Sentiment Analysis**")
        st.caption("Analyze reviews using NLP.")

# =========================================================
# MOVIE REVIEWS
# =========================================================

elif page == "⭐ Movie Reviews":

    st.markdown(
        f'<div class="section-title">⭐ Reviews for {selected_movie}</div>',
        unsafe_allow_html=True
    )

    movie_id = int(
        selected_movie_data["movie_id"]
    )

    csv_reviews = sample_reviews[
        sample_reviews["movie_id"] == movie_id
    ]

    db_reviews = get_reviews(movie_id)

    if len(csv_reviews) == 0 and len(db_reviews) == 0:

        st.info(
            "No reviews available yet. Be the first to review!"
        )

    else:

        for _, review in csv_reviews.iterrows():

            review_html = f"""
<div class="review-card">
<b>👤 {review["user_name"]}</b>
<span style="float:right;">{review["emoji"]}</span>
<br>
<span class="rating">{"⭐" * int(review["rating"])}</span>
<br><br>
{review["comment"]}
</div>
"""

            st.markdown(
                review_html,
                unsafe_allow_html=True
            )

        for review in db_reviews:

            review_html = f"""
<div class="review-card">
<b>👤 {review[2]}</b>
<span style="float:right;">{review[4]}</span>
<br>
<span class="rating">{"⭐" * int(review[3])}</span>
<br><br>
{review[5]}
</div>
"""

            st.markdown(
                review_html,
                unsafe_allow_html=True
            )

    st.markdown("---")

    st.markdown("### ✍️ Write Your Review")

    with st.form("review_form"):

        user_name = st.text_input(
            "Your Name"
        )

        rating = st.slider(
            "Give Rating",
            min_value=1,
            max_value=5,
            value=5
        )

        emoji = st.selectbox(
            "Choose Emoji",
            ["😍", "😊", "🔥", "❤️", "😎", "😢", "😡"]
        )

        comment = st.text_area(
            "Your Movie Comment",
            placeholder="Write your opinion about this movie..."
        )

        submitted = st.form_submit_button(
            "🚀 Submit Review"
        )

        if submitted:

            if (
                user_name.strip() == ""
                or comment.strip() == ""
            ):

                st.warning(
                    "Please enter your name and comment."
                )

            else:

                add_review(
                    movie_id,
                    user_name,
                    rating,
                    emoji,
                    comment
                )

                st.success(
                    "Your review has been added successfully! 🎉"
                )

                st.rerun()

# =========================================================
# AI SENTIMENT
# =========================================================

elif page == "🤖 AI Sentiment":

    st.markdown(
        '<div class="section-title">🤖 AI Sentiment Analysis</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter a movie comment and our Machine Learning "
        "model will predict its sentiment."
    )

    comment = st.text_area(
        "Enter Your Movie Comment",
        placeholder="Example: This movie was amazing!"
    )

    if st.button("🔍 Analyze Sentiment"):

        if comment.strip() == "":

            st.warning(
                "Please enter a comment."
            )

        else:

            transformed_text = vectorizer.transform(
                [comment]
            )

            prediction = model.predict(
                transformed_text
            )[0]

            probabilities = model.predict_proba(
                transformed_text
            )[0]

            confidence = max(probabilities) * 100

            if prediction == 1:

                st.markdown(
                    '<div class="ai-positive">'
                    '😊 Positive Sentiment'
                    '</div>',
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    '<div class="ai-negative">'
                    '😞 Negative Sentiment'
                    '</div>',
                    unsafe_allow_html=True
                )

            st.markdown("### 🎯 AI Confidence")

            st.progress(
                min(int(confidence), 100)
            )

            st.success(
                f"Model Confidence: {confidence:.2f}%"
            )

# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "📊 Model Performance":

    st.markdown(
        '<div class="section-title">📊 Model Performance</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Performance comparison of the Machine Learning "
        "models used in this project."
    )

    results_path = models_path / "model_results.csv"

    if results_path.exists():

        results_df = pd.read_csv(results_path)

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Model results file was not found."
        )

    st.markdown("---")

    st.markdown("### 🧠 Project Technologies")

    tech_col1, tech_col2, tech_col3 = st.columns(3)

    with tech_col1:

        st.info(
            "🐍 Python\n\nProgramming Language"
        )

    with tech_col2:

        st.info(
            "🧹 TF-IDF\n\nText Feature Extraction"
        )

    with tech_col3:

        st.info(
            "🤖 Logistic Regression\n\nSentiment Prediction"
        )

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
🎬 Padam Epdi Irukku Bro?
<br>
Powered by NLP & Machine Learning ❤️
</div>
""", unsafe_allow_html=True)