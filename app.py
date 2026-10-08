import re
import difflib
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import tensorflow as tf
from src.model_loader import load_recommender


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Book Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #a7adba;
        margin-bottom: 30px;
    }

    .stat-card {
        background: linear-gradient(145deg, #171b24, #11141b);
        border: 1px solid #292f3a;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        min-height: 120px;
    }

    .stat-value {
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .stat-label {
        color: #9ca3af;
        font-size: 14px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 750;
        margin-top: 35px;
        margin-bottom: 18px;
    }

    .book-card {
        background: linear-gradient(145deg, #171b24, #11141b);
        border: 1px solid #292f3a;
        border-radius: 16px;
        padding: 24px;
        margin-top: 10px;
        margin-bottom: 8px;
    }

    .book-title {
        font-size: 22px;
        font-weight: 750;
        margin-bottom: 8px;
    }

    .explanation {
        background: #151922;
        border-left: 4px solid #7c3aed;
        padding: 14px 18px;
        border-radius: 8px;
        margin-top: 12px;
        margin-bottom: 25px;
        color: #d1d5db;
    }

    .search-info {
        background: #151922;
        border: 1px solid #292f3a;
        border-radius: 10px;
        padding: 12px 16px;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    section[data-testid="stSidebar"] {
        background-color: #171a21;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data

def load_data():
    books = pd.read_csv(
        "data/processed/books_clean.csv",
        encoding="latin-1",
    )

    ratings = pd.read_csv(
        "data/processed/ratings_clean.csv",
        encoding="latin-1",
    )

    books["ISBN"] = books["ISBN"].astype(str)
    ratings["ISBN"] = ratings["ISBN"].astype(str)
    ratings["User-ID"] = pd.to_numeric(
        ratings["User-ID"], errors="coerce"
    ).astype("Int64")
    ratings["Book-Rating"] = pd.to_numeric(
        ratings["Book-Rating"], errors="coerce"
    )

    return books, ratings


@st.cache_resource

def load_content_model():
    vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
    content_model = joblib.load("models/content_model.joblib")
    tfidf_matrix = joblib.load("models/tfidf_matrix.joblib")
    return vectorizer, content_model, tfidf_matrix


@st.cache_resource

def load_neural_network():
    model = load_recommender("models/neural_recommender.keras")

    raw_user_to_index = joblib.load("models/user_to_index.joblib")
    raw_book_to_index = joblib.load("models/book_to_index.joblib")

    # Normalize mapping keys so the app works whether the mappings were
    # saved with integer or string IDs.
    user_to_index = {}
    for key, value in raw_user_to_index.items():
        try:
            user_to_index[int(key)] = int(value)
        except (ValueError, TypeError):
            continue

    book_to_index = {}
    for key, value in raw_book_to_index.items():
        book_to_index[str(key)] = int(value)

    return model, user_to_index, book_to_index


books, ratings = load_data()

vectorizer, content_model, tfidf_matrix = load_content_model()
neural_model, neural_user_to_index, neural_book_to_index = load_neural_network()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_title(title):
    """Normalize titles so different editions are easier to detect."""
    title = str(title).strip().lower()

    title = re.sub(r"\([^)]*\)", "", title)
    title = re.sub(r"\[[^\]]*\]", "", title)

    title = re.sub(
        r"\b(paperback|hardcover|hardback|large print|mass market|"
        r"audio|audiobook|edition|book club|special edition|deluxe edition)\b",
        "",
        title,
    )

    title = re.sub(r"[^a-z0-9\s]", "", title)
    return " ".join(title.split())


def is_non_book_material(title, author="", publisher=""):
    """Filter study guides and other supplementary material."""
    text = (
        f"{str(title).lower()} "
        f"{str(author).lower()} "
        f"{str(publisher).lower()}"
    )

    unwanted_keywords = [
        "cliffs notes",
        "cliffsnotes",
        "sparknotes",
        "york notes",
        "pearson york notes",
        "yorknotes",
        "study guide",
        "study guides",
        "student guide",
        "student guides",
        "teacher guide",
        "teacher guides",
        "teacher's guide",
        "teachers guide",
        "summary",
        "summaries",
        "book summary",
        "literary summary",
        "analysis guide",
        "analysis guides",
        "reading guide",
        "reading guides",
        "revision guide",
        "revision guides",
        "exam guide",
        "exam guides",
        "revision notes",
        "study notes",
        "workbook",
        "work book",
        "answer key",
        "answer keys",
        "test bank",
        "test banks",
        "lesson plan",
        "lesson plans",
        "literary criticism",
        "literary analysis",
        "companion guide",
        "study edition",
        "notes edition",
        "critical edition",
        "teacher edition",
        "teacher editions",
        "student edition",
        "student editions",
        "teacher's edition",
        "student's edition",
        "teachers edition",
        "students edition",
        "student companion",
        "student companions",
        "teacher companion",
        "teacher companions",
    ]

    return any(keyword in text for keyword in unwanted_keywords)


# ============================================================
# CONTENT-BASED MODEL - USED FOR NEW USERS / COLD START
# ============================================================

def content_recommendations(isbn, top_n=50):
    matching_indices = books.index[
        books["ISBN"].astype(str) == str(isbn)
    ].tolist()

    if not matching_indices:
        return []

    book_index = matching_indices[0]

    n_neighbors = min(top_n + 1, len(books))

    distances, indices = content_model.kneighbors(
        tfidf_matrix[book_index],
        n_neighbors=n_neighbors,
    )

    recommendations = []

    for distance, index in zip(distances[0][1:], indices[0][1:]):
        recommendations.append(
            {
                "ISBN": str(books.iloc[index]["ISBN"]),
                "Content Score": round(float(1 - distance), 3),
            }
        )

    return recommendations


# ============================================================
# NEURAL NETWORK RECOMMENDATION MODEL
# ============================================================

def neural_network_recommendations(user_id, top_n=5):
    """
    Generate Top-N recommendations using the trained neural network.

    The neural model learns user and book embeddings from historical
    user-book ratings and predicts a rating from 1 to 10.
    """

    try:
        user_id = int(user_id)
    except (ValueError, TypeError):
        return []

    if user_id not in neural_user_to_index:
        return []

    user_index = neural_user_to_index[user_id]

    # Books already rated by the selected user are not recommended again.
    rated_isbns = set(
        ratings.loc[
            ratings["User-ID"] == user_id,
            "ISBN",
        ].astype(str)
    )

    candidate_isbns = [
        str(isbn)
        for isbn in neural_book_to_index.keys()
        if str(isbn) not in rated_isbns
    ]

    if not candidate_isbns:
        return []

    candidate_book_indices = np.array(
        [neural_book_to_index[isbn] for isbn in candidate_isbns],
        dtype=np.int32,
    )

    user_indices = np.full(
        len(candidate_book_indices),
        user_index,
        dtype=np.int32,
    )

    predicted_ratings = neural_model.predict(
        [user_indices, candidate_book_indices],
        batch_size=1024,
        verbose=0,
    ).reshape(-1)

    predicted_ratings = np.clip(predicted_ratings, 1, 10)

    predictions = pd.DataFrame(
        {
            "ISBN": candidate_isbns,
            "Predicted Rating": predicted_ratings,
        }
    )

    predictions = predictions.merge(
        books[
            [
                "ISBN",
                "Book-Title",
                "Book-Author",
                "Publisher",
            ]
        ],
        on="ISBN",
        how="left",
    )

    predictions = predictions.dropna(subset=["Book-Title"])

    predictions = predictions[
        ~predictions.apply(
            lambda row: is_non_book_material(
                row["Book-Title"],
                row["Book-Author"],
                row["Publisher"],
            ),
            axis=1,
        )
    ]

    predictions["Normalized Title"] = predictions["Book-Title"].apply(
        normalize_title
    )

    predictions = predictions[
        predictions["Normalized Title"].str.len() > 0
    ]

    predictions = predictions.sort_values(
        "Predicted Rating",
        ascending=False,
    )

    # Keep one recommendation per normalized title.
    predictions = predictions.drop_duplicates(
        subset=["Normalized Title"],
        keep="first",
    )

    return predictions.head(top_n).to_dict("records")


# ============================================================
# NEW USER CONTENT-BASED RECOMMENDATIONS
# ============================================================

def new_user_recommendations(selected_books, top_n=5):
    """Create recommendations for a user with no training history."""

    content_scores = {}

    selected_isbns = {str(isbn) for isbn, _ in selected_books}

    for isbn, rating in selected_books:
        weight = max(float(rating), 1.0) / 10.0

        for recommendation in content_recommendations(
            isbn,
            top_n=50,
        ):
            rec_isbn = str(recommendation["ISBN"])

            if rec_isbn in selected_isbns:
                continue

            score = float(recommendation["Content Score"])
            content_scores[rec_isbn] = (
                content_scores.get(rec_isbn, 0.0)
                + score * weight
            )

    if not content_scores:
        return []

    result = pd.DataFrame(
        [
            {
                "ISBN": isbn,
                "Content Score": score,
            }
            for isbn, score in content_scores.items()
        ]
    )

    result = result.merge(
        books[
            [
                "ISBN",
                "Book-Title",
                "Book-Author",
                "Publisher",
            ]
        ],
        on="ISBN",
        how="left",
    )

    result = result.dropna(subset=["Book-Title"])

    result = result[
        ~result.apply(
            lambda row: is_non_book_material(
                row["Book-Title"],
                row["Book-Author"],
                row["Publisher"],
            ),
            axis=1,
        )
    ]

    result["Normalized Title"] = result["Book-Title"].apply(
        normalize_title
    )

    result = result.sort_values(
        "Content Score",
        ascending=False,
    )

    result = result.drop_duplicates(
        subset=["Normalized Title"],
        keep="first",
    )

    return result.head(top_n).to_dict("records")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🧠 AI Recommendation System")
    st.markdown("---")

    st.markdown("### 🤖 AI Techniques")
    st.markdown(
        """
        **1. Neural Network Recommendation**

        Learns user and book embeddings from historical ratings.

        **2. Content-Based Filtering**

        Uses TF-IDF and cosine similarity for new-user cold start.

        **3. Baseline / Experimental Models**

        Earlier content-based, collaborative and hybrid approaches
        are retained for comparison and experimentation.
        """
    )

    st.markdown("---")

    st.markdown("### 🧠 Recommendation Logic")
    st.markdown(
        """
        **Existing User**

        User ID → Neural Network → Predicted Ratings → Top-N

        **New User**

        Selected Books + Ratings → Content-Based → Top-N
        """
    )

    st.markdown("---")

    st.markdown("### 🛡️ Recommendation Quality")
    st.markdown(
        """
        The system filters:

        • Duplicate editions  
        • York Notes  
        • Cliffs Notes  
        • SparkNotes  
        • Study guides  
        • Workbooks  
        • Summaries  
        • Teacher/student guides  
        • Answer keys
        """
    )

    st.markdown("---")
    st.caption("AI-Based Intelligent Book Recommendation System")


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        📚 AI-Based Intelligent Book Recommendation System
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Personalized book recommendations using a trained Neural Network,
        with Content-Based Filtering for new-user cold-start recommendations.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATASET STATISTICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-value">{len(books):,}</div>
            <div class="stat-label">Books</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-value">{ratings["User-ID"].nunique():,}</div>
            <div class="stat-label">Rated Users</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-value">{len(ratings):,}</div>
            <div class="stat-label">Ratings</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        """
        <div class="stat-card">
            <div class="stat-value">Neural</div>
            <div class="stat-label">Primary AI Model</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# RECOMMENDATION TABS
# ============================================================

st.markdown(
    '<div class="section-title">👤 Choose Recommendation Mode</div>',
    unsafe_allow_html=True,
)

existing_tab, new_tab = st.tabs(
    ["👤 Existing User", "🆕 New User"]
)


# ============================================================
# EXISTING USER
# ============================================================

with existing_tab:
    st.info(
        "Existing users receive personalized recommendations using "
        "the **trained Neural Network model**. The model learns user and "
        "book representations from historical rating data and predicts "
        "ratings for books the user has not rated."
    )

    available_users = sorted(
        int(user_id) for user_id in neural_user_to_index.keys()
    )

    if not available_users:
        st.error("No users are available in the neural-network mapping.")
    else:
        default_index = (
            available_users.index(8)
            if 8 in available_users
            else 0
        )

        selected_user = st.selectbox(
            "Select User ID",
            available_users,
            index=default_index,
            key="existing_user",
        )

        top_n_existing = st.slider(
            "Number of Recommendations",
            min_value=5,
            max_value=10,
            value=5,
            key="existing_top_n",
        )

        st.markdown(
            '<div class="section-title">📖 User Reading History</div>',
            unsafe_allow_html=True,
        )

        user_history = (
            ratings[ratings["User-ID"] == selected_user]
            [["ISBN", "Book-Rating"]]
            .merge(
                books[["ISBN", "Book-Title", "Book-Author"]],
                on="ISBN",
                how="left",
            )
        )

        # Some ratings in the original dataset do not have matching book
        # metadata. Do not display rows containing None as if they were books.
        user_history = user_history.dropna(subset=["Book-Title"])

        if not user_history.empty:
            history_display = (
                user_history[
                    [
                        "Book-Title",
                        "Book-Author",
                        "Book-Rating",
                    ]
                ]
                .sort_values(
                    "Book-Rating",
                    ascending=False,
                )
                .head(10)
            )

            st.dataframe(
                history_display,
                width="stretch",
                hide_index=True,
            )
        else:
            st.info(
                "This user has rating records, but the rated books do not "
                "have matching metadata in the cleaned book dataset."
            )

        generate_existing = st.button(
            "✨ Generate Neural Network Recommendations",
            type="primary",
            width="stretch",
            key="generate_existing",
        )

        if generate_existing:
            with st.spinner(
                "🤖 Neural network is generating personalized recommendations..."
            ):
                recommendations = neural_network_recommendations(
                    selected_user,
                    top_n_existing,
                )

            st.markdown(
                '<div class="section-title">🎯 Top Recommended Books</div>',
                unsafe_allow_html=True,
            )

            if recommendations:
                recommendation_df = pd.DataFrame(recommendations)

                for rank, (_, row) in enumerate(
                    recommendation_df.iterrows(),
                    start=1,
                ):
                    title = str(row["Book-Title"])
                    author = str(row["Book-Author"])
                    publisher = str(row["Publisher"])
                    predicted_rating = float(row["Predicted Rating"])

                    st.markdown(
                        f"""
                        <div class="book-card">
                            <div class="book-title">
                                #{rank} 📚 {title}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.caption(f"{author} • {publisher}")

                    st.metric(
                        "Predicted Rating",
                        f"{predicted_rating:.2f} / 10",
                    )

                    st.markdown(
                        """
                        <div class="explanation">
                            🧠 <b>Why was this recommended?</b><br>
                            The trained neural network learned the relationship
                            between users and books from historical rating data.
                            This book received a high predicted rating for the
                            selected user, so it was placed among the Top-N
                            recommendations.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    '<div class="section-title">📊 Predicted Rating Analysis</div>',
                    unsafe_allow_html=True,
                )

                chart_df = recommendation_df[
                    ["Book-Title", "Predicted Rating"]
                ].set_index("Book-Title")

                st.bar_chart(
                    chart_df,
                    width="stretch",
                )
            else:
                st.warning(
                    "No recommendations could be generated for this user."
                )


# ============================================================
# NEW USER
# ============================================================

with new_tab:
    st.info(
        "🆕 New users can select books they like and rate them. "
        "Because a new user has no learned neural-network user embedding, "
        "the system uses **Content-Based Filtering** to handle the initial "
        "cold-start problem."
    )

    st.markdown(
        '<div class="section-title">👋 Tell Us About Yourself</div>',
        unsafe_allow_html=True,
    )

    user_name = st.text_input(
        "Your Name",
        placeholder="Enter your name",
        key="new_user_name",
    )

    st.markdown(
        '<div class="section-title">📚 Select Books You Like</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Search for books you like and select up to 5 books "
        "to create your preference profile."
    )

    book_options = (
        books[
            [
                "ISBN",
                "Book-Title",
                "Book-Author",
            ]
        ]
        .drop_duplicates(subset=["Book-Title"])
        .dropna(subset=["Book-Title"])
        .copy()
    )

    book_options["ISBN"] = book_options["ISBN"].astype(str)
    book_options["Book-Title"] = book_options["Book-Title"].astype(str)
    book_options["Book-Author"] = (
        book_options["Book-Author"]
        .fillna("Unknown Author")
        .astype(str)
    )

    # Build the searchable index only once. This avoids scanning and
    # normalizing all 271k+ books every time the user types a character.
    @st.cache_resource
    def prepare_book_search_index():
        indexed = book_options[[
            "ISBN",
            "Book-Title",
            "Book-Author",
        ]].copy()

        indexed["_title_lower"] = indexed["Book-Title"].str.lower()
        indexed["_author_lower"] = indexed["Book-Author"].str.lower()
        indexed["_title_compact"] = indexed["_title_lower"].str.replace(
            r"[^a-z0-9]", "", regex=True
        )
        indexed["_author_compact"] = indexed["_author_lower"].str.replace(
            r"[^a-z0-9]", "", regex=True
        )
        return indexed

    search_index = prepare_book_search_index()

    search_text = st.text_input(
        "🔎 Search for a book",
        placeholder="Example: Harry Potter, Hobbit, Pride and Prejudice...",
        key="book_search",
    )

    if search_text.strip():
        query = search_text.strip()
        query_lower = query.lower()

        # ----------------------------------------------------
        # 1. FAST NORMAL SEARCH
        # ----------------------------------------------------
        # Search the pre-built lowercase index instead of repeatedly
        # converting the entire dataset.
        search_mask = (
            search_index["_title_lower"].str.contains(
                query_lower, case=False, na=False, regex=False
            )
            | search_index["_author_lower"].str.contains(
                query_lower, case=False, na=False, regex=False
            )
        )

        search_results = search_index.loc[
            search_mask, ["ISBN", "Book-Title", "Book-Author"]
        ].copy()

        # ----------------------------------------------------
        # 2. FAST FUZZY FALLBACK
        # ----------------------------------------------------
        # Example: "madolduwa" -> "Madol Doova"
        # Instead of comparing against every book, first create a
        # small candidate set using the first 3-4 normalized letters.
        if search_results.empty and len(query_lower) >= 3:
            compact_query = re.sub(r"[^a-z0-9]", "", query_lower)

            if len(compact_query) >= 4:
                seed = compact_query[:4]
            else:
                seed = compact_query[:3]

            candidate_mask = (
                search_index["_title_compact"].str.contains(
                    seed, case=False, na=False, regex=False
                )
                | search_index["_author_compact"].str.contains(
                    seed, case=False, na=False, regex=False
                )
            )

            candidates = search_index.loc[candidate_mask].copy()

            # If the 4-character seed is too strict, fall back to 3 chars.
            if candidates.empty and len(seed) >= 4:
                seed = compact_query[:3]
                candidate_mask = (
                    search_index["_title_compact"].str.contains(
                        seed, case=False, na=False, regex=False
                    )
                    | search_index["_author_compact"].str.contains(
                        seed, case=False, na=False, regex=False
                    )
                )
                candidates = search_index.loc[candidate_mask].copy()

            # Only run the relatively expensive difflib calculation on
            # this small candidate set.
            if not candidates.empty:
                fuzzy_scores = []
                query_words = set(re.findall(r"[a-z0-9]+", query_lower))

                for idx, row in candidates.iterrows():
                    title_compact = row["_title_compact"]
                    author_compact = row["_author_compact"]

                    title_score = difflib.SequenceMatcher(
                        None, compact_query, title_compact
                    ).ratio()
                    author_score = difflib.SequenceMatcher(
                        None, compact_query, author_compact
                    ).ratio()

                    title_words = set(
                        re.findall(r"[a-z0-9]+", str(row["Book-Title"]).lower())
                    )
                    word_bonus = (
                        0.15
                        if query_words and query_words.issubset(title_words)
                        else 0.0
                    )

                    score = max(title_score, author_score * 0.9) + word_bonus

                    if score >= 0.45:
                        fuzzy_scores.append((idx, score))

                if fuzzy_scores:
                    fuzzy_scores.sort(key=lambda item: item[1], reverse=True)
                    best_indices = [idx for idx, _ in fuzzy_scores[:50]]
                    search_results = search_index.loc[
                        best_indices, ["ISBN", "Book-Title", "Book-Author"]
                    ].copy()

    else:
        search_results = pd.DataFrame(columns=book_options.columns)

    if not search_results.empty:
        st.markdown(
            f"""
            <div class="search-info">
                🔎 Found <b>{len(search_results)}</b> matching books.
            </div>
            """,
            unsafe_allow_html=True,
        )

        search_options = search_results["ISBN"].astype(str).tolist()

        search_lookup = {}
        for _, row in search_results.iterrows():
            isbn = str(row["ISBN"])
            title = str(row["Book-Title"])
            author = str(row["Book-Author"])
            search_lookup[isbn] = f"{title} — {author}"

        selected_from_search = st.multiselect(
            "Select books you like",
            options=search_options,
            format_func=lambda isbn: search_lookup.get(isbn, isbn),
            max_selections=5,
            key="searched_books",
        )
    else:
        selected_from_search = []

        if search_text.strip():
            st.warning(
                "No books found. Try a different search term."
            )
        else:
            st.info("Enter a book title above to search.")

    selected_books_with_ratings = []

    if selected_from_search:
        st.markdown("### ⭐ Rate Your Selected Books")

        for isbn in selected_from_search:
            book_info = book_options[
                book_options["ISBN"].astype(str) == str(isbn)
            ]

            if not book_info.empty:
                title = str(book_info.iloc[0]["Book-Title"])
                author = str(book_info.iloc[0]["Book-Author"])

                rating = st.slider(
                    f"{title} — {author}",
                    min_value=1,
                    max_value=10,
                    value=8,
                    key=f"rating_{isbn}",
                )

                selected_books_with_ratings.append(
                    (str(isbn), rating)
                )

    generate_new = st.button(
        "🚀 Get My Recommendations",
        type="primary",
        width="stretch",
        key="generate_new",
    )

    if generate_new:
        if not selected_books_with_ratings:
            st.warning(
                "Please search for and select at least one book "
                "before generating recommendations."
            )
        else:
            with st.spinner(
                "🤖 AI is learning your preferences..."
            ):
                new_recommendations = new_user_recommendations(
                    selected_books_with_ratings,
                    top_n=5,
                )

            if user_name.strip():
                st.success(
                    f"Great, {user_name}! Here are your personalized recommendations."
                )
            else:
                st.success(
                    "Here are your personalized recommendations."
                )

            st.markdown(
                '<div class="section-title">🎯 Recommended For You</div>',
                unsafe_allow_html=True,
            )

            if new_recommendations:
                st.markdown(
                    """
                    <div class="explanation">
                        🧠 <b>How were these recommendations generated?</b><br>
                        These books were recommended based on their content
                        similarity to the books you selected and rated highly.
                        This content-based approach is used for new users who
                        do not yet have a learned neural-network user profile.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                recommendation_df = pd.DataFrame(new_recommendations)

                for rank, (_, row) in enumerate(
                    recommendation_df.iterrows(),
                    start=1,
                ):
                    title = str(row["Book-Title"])
                    author = str(row["Book-Author"])
                    publisher = str(row["Publisher"])
                    score = float(row["Content Score"])

                    st.markdown(
                        f"""
                        <div class="book-card">
                            <div class="book-title">
                                #{rank} 📚 {title}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.caption(f"{author} • {publisher}")
                    st.metric(
                        "Content Similarity Score",
                        f"{score:.3f}",
                    )

            else:
                st.warning(
                    "We could not generate recommendations from the selected books."
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "AI-Based Intelligent Book Recommendation System "
    "| Neural Network + Content-Based Cold Start"
)
