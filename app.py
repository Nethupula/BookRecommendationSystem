import streamlit as st
import pandas as pd
import joblib
import re
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Book Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
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
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    books = pd.read_csv(
        "data/processed/books_clean.csv",
        encoding="latin-1"
    )

    ratings = pd.read_csv(
        "data/processed/ratings_clean.csv",
        encoding="latin-1"
    )

    books["ISBN"] = books["ISBN"].astype(str)
    ratings["ISBN"] = ratings["ISBN"].astype(str)

    return books, ratings


@st.cache_resource
def load_content_model():

    vectorizer = joblib.load(
        "models/tfidf_vectorizer.joblib"
    )

    content_model = joblib.load(
        "models/content_model.joblib"
    )

    tfidf_matrix = joblib.load(
        "models/tfidf_matrix.joblib"
    )

    return (
        vectorizer,
        content_model,
        tfidf_matrix
    )


books, ratings = load_data()

(
    vectorizer,
    content_model,
    tfidf_matrix
) = load_content_model()


# ============================================================
# TITLE NORMALIZATION
# ============================================================

def normalize_title(title):
    """
    Normalize book titles so different editions,
    formats and series labels can be recognized
    as the same underlying book.
    """

    title = str(title).strip().lower()

    # Remove text inside parentheses
    title = re.sub(
        r"\([^)]*\)",
        "",
        title
    )

    # Remove text inside square brackets
    title = re.sub(
        r"\[[^\]]*\]",
        "",
        title
    )

    # Remove common edition / format terms
    title = re.sub(
        r"\b("
        r"paperback|"
        r"hardcover|"
        r"hardback|"
        r"large print|"
        r"mass market|"
        r"audio|"
        r"audiobook|"
        r"edition|"
        r"book club|"
        r"special edition|"
        r"deluxe edition"
        r")\b",
        "",
        title
    )

    # Remove punctuation
    title = re.sub(
        r"[^a-z0-9\s]",
        "",
        title
    )

    # Remove extra spaces
    title = " ".join(
        title.split()
    )

    return title


# ============================================================
# FILTER UNWANTED / SUPPLEMENTARY MATERIAL
# ============================================================

def is_non_book_material(
    title,
    author="",
    publisher=""
):
    """
    Detect study guides, notes, summaries,
    workbooks and other supplementary material.

    These materials are excluded from recommendations
    because the goal is to recommend actual books.
    """

    title_text = str(title).lower()
    author_text = str(author).lower()
    publisher_text = str(publisher).lower()

    text = (
        title_text
        + " "
        + author_text
        + " "
        + publisher_text
    )

    unwanted_keywords = [

        # ----------------------------------------------------
        # Cliffs / Spark notes
        # ----------------------------------------------------

        "cliffs notes",
        "cliffsnotes",
        "sparknotes",

        # ----------------------------------------------------
        # York Notes
        # ----------------------------------------------------

        "york notes",
        "pearson york notes",
        "yorknotes",

        # ----------------------------------------------------
        # Study guides
        # ----------------------------------------------------

        "study guide",
        "study guides",
        "student guide",
        "student guides",
        "teacher guide",
        "teacher guides",
        "teacher's guide",
        "teachers guide",

        # ----------------------------------------------------
        # Summaries
        # ----------------------------------------------------

        "summary",
        "summaries",
        "book summary",
        "literary summary",

        # ----------------------------------------------------
        # Reading / analysis guides
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Workbooks / educational material
        # ----------------------------------------------------

        "workbook",
        "work book",
        "answer key",
        "answer keys",
        "test bank",
        "test banks",
        "lesson plan",
        "lesson plans",

        # ----------------------------------------------------
        # Literary study material
        # ----------------------------------------------------

        "literary criticism",
        "literary analysis",
        "companion guide",
        "study edition",
        "notes edition",
        "critical edition",

        # ----------------------------------------------------
        # Teacher / student editions
        # ----------------------------------------------------

        "teacher edition",
        "teacher editions",
        "student edition",
        "student editions",
        "teacher's edition",
        "student's edition",
        "teachers edition",
        "students edition",

        # ----------------------------------------------------
        # Companions
        # ----------------------------------------------------

        "student companion",
        "student companions",
        "teacher companion",
        "teacher companions"
    ]

    # Check all keywords
    for keyword in unwanted_keywords:

        if keyword in text:
            return True

    # Explicit York Notes check
    if "york notes" in title_text:
        return True

    if "york notes" in publisher_text:
        return True

    return False


# ============================================================
# PREPARE COLLABORATIVE MODEL
# ============================================================

@st.cache_resource
def prepare_collaborative_model(ratings):

    filtered = ratings.copy()

    # --------------------------------------------------------
    # Active users
    # --------------------------------------------------------

    user_counts = filtered[
        "User-ID"
    ].value_counts()

    active_users = user_counts[
        user_counts >= 5
    ].index

    filtered = filtered[
        filtered["User-ID"].isin(
            active_users
        )
    ]

    # --------------------------------------------------------
    # Frequently rated books
    # --------------------------------------------------------

    book_counts = filtered[
        "ISBN"
    ].value_counts()

    active_books = book_counts[
        book_counts >= 5
    ].index

    filtered = filtered[
        filtered["ISBN"].isin(
            active_books
        )
    ]

    # --------------------------------------------------------
    # User-book matrix
    # --------------------------------------------------------

    user_book_matrix = filtered.pivot_table(
        index="User-ID",
        columns="ISBN",
        values="Book-Rating",
        fill_value=0
    )

    sparse_matrix = csr_matrix(
        user_book_matrix.values
    )

    # --------------------------------------------------------
    # Nearest-neighbor model
    # --------------------------------------------------------

    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=6,
        n_jobs=-1
    )

    model.fit(
        sparse_matrix
    )

    return (
        filtered,
        user_book_matrix,
        sparse_matrix,
        model
    )


(
    filtered_ratings,
    user_book_matrix,
    sparse_matrix,
    collaborative_model
) = prepare_collaborative_model(
    ratings
)


# ============================================================
# CONTENT-BASED RECOMMENDATIONS
# ============================================================

def content_recommendations(
    isbn,
    top_n=50
):

    matching_indices = books.index[
        books["ISBN"].astype(str)
        == str(isbn)
    ].tolist()

    if not matching_indices:
        return []

    book_index = matching_indices[0]

    distances, indices = (
        content_model.kneighbors(
            tfidf_matrix[book_index],
            n_neighbors=top_n + 1
        )
    )

    recommendations = []

    for distance, index in zip(
        distances[0][1:],
        indices[0][1:]
    ):

        recommendations.append(
            {
                "ISBN": str(
                    books.iloc[index]["ISBN"]
                ),
                "Content Score": round(
                    1 - distance,
                    3
                )
            }
        )

    return recommendations


# ============================================================
# COLLABORATIVE RECOMMENDATIONS
# ============================================================

def collaborative_recommendations(
    user_id,
    top_n=50
):

    if user_id not in user_book_matrix.index:
        return []

    user_index = (
        user_book_matrix.index.get_loc(
            user_id
        )
    )

    distances, indices = (
        collaborative_model.kneighbors(
            sparse_matrix[user_index],
            n_neighbors=6
        )
    )

    recommendations = {}

    already_rated = set(
        ratings[
            ratings["User-ID"] == user_id
        ]["ISBN"].astype(str)
    )

    for distance, similar_index in zip(
        distances[0][1:],
        indices[0][1:]
    ):

        similarity = 1 - distance

        similar_user = (
            user_book_matrix.index[
                similar_index
            ]
        )

        similar_ratings = ratings[
            ratings["User-ID"] == similar_user
        ]

        for _, row in (
            similar_ratings.iterrows()
        ):

            isbn = str(
                row["ISBN"]
            )

            rating = row[
                "Book-Rating"
            ]

            # Only use books liked by similar users
            if (
                isbn not in already_rated
                and rating >= 7
            ):

                score = (
                    similarity * rating
                )

                recommendations[isbn] = (
                    recommendations.get(
                        isbn,
                        0
                    ) + score
                )

    sorted_recommendations = sorted(
        recommendations.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        {
            "ISBN": isbn,
            "Collaborative Score": score
        }
        for isbn, score
        in sorted_recommendations[:top_n]
    ]


# ============================================================
# HYBRID RECOMMENDATIONS
# ============================================================

def hybrid_recommendations(
    user_id,
    top_n=10
):

    user_ratings = ratings[
        ratings["User-ID"] == user_id
    ]

    liked_books = user_ratings[
        user_ratings["Book-Rating"] >= 5
    ]

    # --------------------------------------------------------
    # CONTENT SCORES
    # --------------------------------------------------------

    content_scores = {}

    for _, row in (
        liked_books.iterrows()
    ):

        isbn = str(
            row["ISBN"]
        )

        rating = row[
            "Book-Rating"
        ]

        recommendations = (
            content_recommendations(
                isbn,
                top_n=30
            )
        )

        for recommendation in recommendations:

            rec_isbn = str(
                recommendation["ISBN"]
            )

            if rec_isbn == isbn:
                continue

            rec_rows = books[
                books["ISBN"].astype(str)
                == rec_isbn
            ]

            if rec_rows.empty:
                continue

            rec_row = rec_rows.iloc[0]

            # ------------------------------------------------
            # Filter unwanted material
            # ------------------------------------------------

            if is_non_book_material(
                rec_row["Book-Title"],
                rec_row["Book-Author"],
                rec_row["Publisher"]
            ):
                continue

            # ------------------------------------------------
            # Remove duplicate edition
            # ------------------------------------------------

            original_rows = books[
                books["ISBN"].astype(str)
                == isbn
            ]

            if not original_rows.empty:

                original_title = (
                    original_rows.iloc[0]["Book-Title"]
                )

                if (
                    normalize_title(
                        rec_row["Book-Title"]
                    )
                    ==
                    normalize_title(
                        original_title
                    )
                ):
                    continue

            # ------------------------------------------------
            # Weighted content score
            # ------------------------------------------------

            score = (
                recommendation[
                    "Content Score"
                ]
                * (rating / 10)
            )

            content_scores[rec_isbn] = max(
                content_scores.get(
                    rec_isbn,
                    0
                ),
                score
            )

    # --------------------------------------------------------
    # COLLABORATIVE SCORES
    # --------------------------------------------------------

    collaborative = (
        collaborative_recommendations(
            user_id,
            top_n=50
        )
    )

    collaborative_scores = {}

    for item in collaborative:

        isbn = str(
            item["ISBN"]
        )

        rec_rows = books[
            books["ISBN"].astype(str)
            == isbn
        ]

        if rec_rows.empty:
            continue

        rec_row = rec_rows.iloc[0]

        if is_non_book_material(
            rec_row["Book-Title"],
            rec_row["Book-Author"],
            rec_row["Publisher"]
        ):
            continue

        collaborative_scores[isbn] = (
            item["Collaborative Score"]
        )

    # --------------------------------------------------------
    # NORMALIZE CONTENT
    # --------------------------------------------------------

    if content_scores:

        max_content = max(
            content_scores.values()
        )

        if max_content > 0:

            content_scores = {
                isbn: score / max_content
                for isbn, score
                in content_scores.items()
            }

    # --------------------------------------------------------
    # NORMALIZE COLLABORATIVE
    # --------------------------------------------------------

    if collaborative_scores:

        max_collaborative = max(
            collaborative_scores.values()
        )

        if max_collaborative > 0:

            collaborative_scores = {
                isbn: score / max_collaborative
                for isbn, score
                in collaborative_scores.items()
            }

    # --------------------------------------------------------
    # COMBINE CANDIDATES
    # --------------------------------------------------------

    candidate_isbns = (
        set(content_scores.keys())
        |
        set(collaborative_scores.keys())
    )

    already_rated = set(
        user_ratings[
            "ISBN"
        ].astype(str)
    )

    results = []

    for isbn in candidate_isbns:

        if isbn in already_rated:
            continue

        rec_rows = books[
            books["ISBN"].astype(str)
            == isbn
        ]

        if rec_rows.empty:
            continue

        rec_row = rec_rows.iloc[0]

        # ------------------------------------------------
        # Remove unwanted material
        # ------------------------------------------------

        if is_non_book_material(
            rec_row["Book-Title"],
            rec_row["Book-Author"],
            rec_row["Publisher"]
        ):
            continue

        # ------------------------------------------------
        # Scores
        # ------------------------------------------------

        content_score = (
            content_scores.get(
                isbn,
                0
            )
        )

        collaborative_score = (
            collaborative_scores.get(
                isbn,
                0
            )
        )

        hybrid_score = (
            0.5 * content_score
            +
            0.5 * collaborative_score
        )

        results.append(
            {
                "ISBN": isbn,
                "Content Score": content_score,
                "Collaborative Score":
                    collaborative_score,
                "Hybrid Score":
                    hybrid_score
            }
        )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    results.sort(
        key=lambda x: x["Hybrid Score"],
        reverse=True
    )

    # --------------------------------------------------------
    # Remove duplicate book titles
    # --------------------------------------------------------

    final_results = []

    seen_titles = set()

    for item in results:

        book_rows = books[
            books["ISBN"].astype(str)
            == str(item["ISBN"])
        ]

        if book_rows.empty:
            continue

        title = str(
            book_rows.iloc[0]["Book-Title"]
        )

        normalized = normalize_title(
            title
        )

        if normalized in seen_titles:
            continue

        seen_titles.add(
            normalized
        )

        final_results.append(item)

        if len(final_results) >= top_n:
            break

    return final_results


# ============================================================
# NEW USER RECOMMENDATIONS
# ============================================================

def new_user_recommendations(
    selected_books,
    top_n=5
):

    content_scores = {}

    # --------------------------------------------------------
    # SELECTED ISBNs
    # --------------------------------------------------------

    selected_isbns = {
        str(isbn)
        for isbn, rating
        in selected_books
    }

    # --------------------------------------------------------
    # SELECTED TITLES
    # --------------------------------------------------------

    selected_titles = set()

    for isbn in selected_isbns:

        selected_rows = books[
            books["ISBN"].astype(str)
            == isbn
        ]

        for _, row in (
            selected_rows.iterrows()
        ):

            selected_titles.add(
                normalize_title(
                    row["Book-Title"]
                )
            )

    # --------------------------------------------------------
    # CONTENT RECOMMENDATIONS
    # --------------------------------------------------------

    for isbn, rating in selected_books:

        recommendations = (
            content_recommendations(
                isbn,
                top_n=50
            )
        )

        for recommendation in recommendations:

            rec_isbn = str(
                recommendation["ISBN"]
            )

            # Don't recommend selected ISBN
            if rec_isbn in selected_isbns:
                continue

            # ------------------------------------------------
            # Get book information
            # ------------------------------------------------

            rec_rows = books[
                books["ISBN"].astype(str)
                == rec_isbn
            ]

            if rec_rows.empty:
                continue

            rec_row = rec_rows.iloc[0]

            rec_title = str(
                rec_row["Book-Title"]
            )

            rec_author = str(
                rec_row["Book-Author"]
            )

            rec_publisher = str(
                rec_row["Publisher"]
            )

            # ------------------------------------------------
            # FILTER STUDY MATERIAL
            # ------------------------------------------------

            if is_non_book_material(
                rec_title,
                rec_author,
                rec_publisher
            ):
                continue

            # ------------------------------------------------
            # REMOVE DUPLICATE EDITIONS
            # ------------------------------------------------

            normalized_rec_title = (
                normalize_title(
                    rec_title
                )
            )

            if (
                normalized_rec_title
                in selected_titles
            ):
                continue

            # ------------------------------------------------
            # SCORE
            # ------------------------------------------------

            similarity = float(
                recommendation[
                    "Content Score"
                ]
            )

            weighted_score = (
                similarity
                * (rating / 10)
            )

            content_scores[rec_isbn] = max(
                content_scores.get(
                    rec_isbn,
                    0
                ),
                weighted_score
            )

    # --------------------------------------------------------
    # NO RESULTS
    # --------------------------------------------------------

    if not content_scores:
        return []

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    max_score = max(
        content_scores.values()
    )

    if max_score > 0:

        content_scores = {
            isbn: score / max_score
            for isbn, score
            in content_scores.items()
        }

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    sorted_scores = sorted(
        content_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # --------------------------------------------------------
    # FINAL FILTER
    # --------------------------------------------------------

    final_results = []

    already_added_titles = set()

    for isbn, score in sorted_scores:

        book_rows = books[
            books["ISBN"].astype(str)
            == str(isbn)
        ]

        if book_rows.empty:
            continue

        row = book_rows.iloc[0]

        title = str(
            row["Book-Title"]
        )

        author = str(
            row["Book-Author"]
        )

        publisher = str(
            row["Publisher"]
        )

        # ------------------------------------------------
        # Remove study material
        # ------------------------------------------------

        if is_non_book_material(
            title,
            author,
            publisher
        ):
            continue

        # ------------------------------------------------
        # Remove duplicate editions
        # ------------------------------------------------

        normalized_title = normalize_title(
            title
        )

        if normalized_title in already_added_titles:
            continue

        already_added_titles.add(
            normalized_title
        )

        final_results.append(
            {
                "ISBN": str(isbn),
                "Content Score": score
            }
        )

        if len(final_results) >= top_n:
            break

    return final_results


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## ⚙️ Recommendation Settings"
    )

    st.markdown("---")

    st.markdown(
        "### 🤖 AI Techniques"
    )

    st.markdown(
        """
        **1. Content-Based Filtering**

        Uses TF-IDF and cosine similarity.

        **2. Collaborative Filtering**

        Uses user similarity and rating patterns.

        **3. Hybrid Recommendation**

        Combines both recommendation signals.
        """
    )

    st.markdown("---")

    st.markdown(
        "### 🧠 Recommendation Logic"
    )

    st.markdown(
        """
        **Existing User**

        Content + Collaborative → Hybrid

        **New User**

        User Preferences → Content-Based
        """
    )

    st.markdown("---")

    st.markdown(
        "### 🛡️ Recommendation Quality"
    )

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

    st.caption(
        "AI-Based Intelligent Book Recommendation System"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        📚 AI-Based Intelligent Book Recommendation System
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Personalized book recommendations using
        Content-Based Filtering, Collaborative Filtering,
        and Hybrid Recommendation.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATASET STATISTICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-value">
                {len(books):,}
            </div>
            <div class="stat-label">
                Books
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-value">
                {ratings["User-ID"].nunique():,}
            </div>
            <div class="stat-label">
                Rated Users
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-value">
                {len(ratings):,}
            </div>
            <div class="stat-label">
                Ratings
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:

    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-value">
                {len(user_book_matrix):,}
            </div>
            <div class="stat-label">
                Active Users
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# RECOMMENDATION MODE
# ============================================================

st.markdown(
    '<div class="section-title">'
    '👤 Choose Recommendation Mode'
    '</div>',
    unsafe_allow_html=True
)

existing_tab, new_tab = st.tabs(
    [
        "👤 Existing User",
        "🆕 New User"
    ]
)


# ============================================================
# EXISTING USER
# ============================================================

with existing_tab:

    st.info(
        "Existing users receive recommendations using "
        "**Content-Based + Collaborative + Hybrid Filtering**."
    )

    # --------------------------------------------------------
    # USER SELECTION
    # --------------------------------------------------------

    available_users = sorted(
        user_book_matrix.index.tolist()
    )

    default_index = (
        available_users.index(8)
        if 8 in available_users
        else 0
    )

    selected_user = st.selectbox(
        "Select User ID",
        available_users,
        index=default_index,
        key="existing_user"
    )

    top_n_existing = st.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=10,
        value=5,
        key="existing_top_n"
    )

    # --------------------------------------------------------
    # USER HISTORY
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '📖 User Reading History'
        '</div>',
        unsafe_allow_html=True
    )

    user_history = ratings[
        ratings["User-ID"]
        == selected_user
    ].merge(
        books,
        on="ISBN",
        how="left"
    )

    if not user_history.empty:

        history_display = (
            user_history[
                [
                    "Book-Title",
                    "Book-Author",
                    "Book-Rating"
                ]
            ]
            .sort_values(
                "Book-Rating",
                ascending=False
            )
            .head(10)
        )

        st.dataframe(
            history_display,
            width="stretch",
            hide_index=True
        )

    else:

        st.warning(
            "No rating history available."
        )

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    generate_existing = st.button(
        "✨ Generate Personalized Recommendations",
        type="primary",
        width="stretch",
        key="generate_existing"
    )

    if generate_existing:

        with st.spinner(
            "🤖 AI is generating personalized recommendations..."
        ):

            recommendations = (
                hybrid_recommendations(
                    selected_user,
                    top_n_existing
                )
            )

        st.markdown(
            '<div class="section-title">'
            '🎯 Top Recommended Books'
            '</div>',
            unsafe_allow_html=True
        )

        if recommendations:

            recommendation_df = (
                pd.DataFrame(
                    recommendations
                )
            )

            recommendation_df = (
                recommendation_df.merge(
                    books[
                        [
                            "ISBN",
                            "Book-Title",
                            "Book-Author",
                            "Publisher"
                        ]
                    ],
                    on="ISBN",
                    how="left"
                )
            )

            for rank, (_, row) in enumerate(
                recommendation_df.iterrows(),
                start=1
            ):

                title = str(
                    row["Book-Title"]
                )

                author = str(
                    row["Book-Author"]
                )

                publisher = str(
                    row["Publisher"]
                )

                content_score = float(
                    row["Content Score"]
                )

                collaborative_score = float(
                    row["Collaborative Score"]
                )

                hybrid_score = float(
                    row["Hybrid Score"]
                )

                if (
                    content_score > 0
                    and collaborative_score > 0
                ):

                    explanation = (
                        "Recommended using both book-content "
                        "similarity and preferences of similar users."
                    )

                elif content_score > 0:

                    explanation = (
                        "Recommended primarily because its "
                        "content is similar to books you liked."
                    )

                else:

                    explanation = (
                        "Recommended primarily from the "
                        "reading patterns of similar users."
                    )

                st.markdown(
                    f"""
                    <div class="book-card">
                        <div class="book-title">
                            #{rank} 📚 {title}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.caption(
                    f"{author} • {publisher}"
                )

                score1, score2, score3 = st.columns(3)

                with score1:

                    st.metric(
                        "Content Score",
                        f"{content_score:.3f}"
                    )

                with score2:

                    st.metric(
                        "Collaborative Score",
                        f"{collaborative_score:.3f}"
                    )

                with score3:

                    st.metric(
                        "Hybrid Score",
                        f"{hybrid_score:.3f}"
                    )

                st.markdown(
                    f"""
                    <div class="explanation">
                        🧠 <b>Why was this recommended?</b><br>
                        {explanation}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # ------------------------------------------------
            # CHART
            # ------------------------------------------------

            st.markdown(
                '<div class="section-title">'
                '📊 Recommendation Score Analysis'
                '</div>',
                unsafe_allow_html=True
            )

            chart_df = (
                recommendation_df[
                    [
                        "Book-Title",
                        "Content Score",
                        "Collaborative Score",
                        "Hybrid Score"
                    ]
                ]
                .set_index(
                    "Book-Title"
                )
            )

            st.bar_chart(
                chart_df,
                width="stretch"
            )

        else:

            st.warning(
                "No recommendations could be generated."
            )


# ============================================================
# NEW USER
# ============================================================

with new_tab:

    st.info(
        "🆕 New users can select books they like and rate them. "
        "Because a new user has no previous interaction history, "
        "the system uses **Content-Based Filtering** to handle "
        "the initial cold-start problem."
    )

    # --------------------------------------------------------
    # NAME
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '👋 Tell Us About Yourself'
        '</div>',
        unsafe_allow_html=True
    )

    user_name = st.text_input(
        "Your Name",
        placeholder="Enter your name",
        key="new_user_name"
    )

    # --------------------------------------------------------
    # BOOK SELECTION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '📚 Select Books You Like'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Search for books you like and select up to "
        "5 books to create your preference profile."
    )

    book_options = (
        books[
            [
                "ISBN",
                "Book-Title",
                "Book-Author"
            ]
        ]
        .drop_duplicates(
            subset=["Book-Title"]
        )
        .dropna(
            subset=["Book-Title"]
        )
        .copy()
    )

    book_options["ISBN"] = (
        book_options["ISBN"]
        .astype(str)
    )

    book_options["Book-Title"] = (
        book_options["Book-Title"]
        .astype(str)
    )

    book_options["Book-Author"] = (
        book_options["Book-Author"]
        .fillna("Unknown Author")
        .astype(str)
    )

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    search_text = st.text_input(
        "🔎 Search for a book",
        placeholder=(
            "Example: Harry Potter, Hobbit, "
            "Pride and Prejudice..."
        ),
        key="book_search"
    )

    if search_text.strip():

        search_results = (
            book_options[
                book_options[
                    "Book-Title"
                ].str.contains(
                    search_text.strip(),
                    case=False,
                    na=False
                )
            ]
            .head(50)
        )

    else:

        search_results = pd.DataFrame(
            columns=book_options.columns
        )

    # --------------------------------------------------------
    # SEARCH RESULTS
    # --------------------------------------------------------

    if not search_results.empty:

        st.markdown(
            f"""
            <div class="search-info">
                🔎 Found <b>{len(search_results)}</b>
                matching books.
            </div>
            """,
            unsafe_allow_html=True
        )

        search_options = (
            search_results[
                "ISBN"
            ]
            .astype(str)
            .tolist()
        )

        search_lookup = {}

        for _, row in (
            search_results.iterrows()
        ):

            isbn = str(
                row["ISBN"]
            )

            title = str(
                row["Book-Title"]
            )

            author = str(
                row["Book-Author"]
            )

            search_lookup[isbn] = (
                f"{title} — {author}"
            )

        selected_from_search = (
            st.multiselect(
                "Select books you like",
                options=search_options,
                format_func=lambda isbn:
                    search_lookup.get(
                        isbn,
                        isbn
                    ),
                max_selections=5,
                key="searched_books"
            )
        )

    else:

        selected_from_search = []

        if search_text.strip():

            st.warning(
                "No books found. "
                "Try a different search term."
            )

        else:

            st.info(
                "Enter a book title above to search."
            )

    # --------------------------------------------------------
    # RATING
    # --------------------------------------------------------

    if selected_from_search:

        st.markdown(
            "### ⭐ Rate Your Selected Books"
        )

        selected_books_with_ratings = []

        for isbn in selected_from_search:

            book_info = book_options[
                book_options[
                    "ISBN"
                ].astype(str)
                == str(isbn)
            ]

            if not book_info.empty:

                title = str(
                    book_info.iloc[0][
                        "Book-Title"
                    ]
                )

                author = str(
                    book_info.iloc[0][
                        "Book-Author"
                    ]
                )

                rating = st.slider(
                    f"{title} — {author}",
                    min_value=1,
                    max_value=10,
                    value=8,
                    key=f"rating_{isbn}"
                )

                selected_books_with_ratings.append(
                    (
                        str(isbn),
                        rating
                    )
                )

    else:

        selected_books_with_ratings = []

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    generate_new = st.button(
        "🚀 Get My Recommendations",
        type="primary",
        width="stretch",
        key="generate_new"
    )

    if generate_new:

        if not selected_books_with_ratings:

            st.warning(
                "Please search for and select at least "
                "one book before generating recommendations."
            )

        else:

            with st.spinner(
                "🤖 AI is learning your preferences..."
            ):

                new_recommendations = (
                    new_user_recommendations(
                        selected_books_with_ratings,
                        top_n=5
                    )
                )

            # ------------------------------------------------
            # SUCCESS MESSAGE
            # ------------------------------------------------

            if user_name.strip():

                st.success(
                    f"Great, {user_name}! "
                    "Here are your personalized recommendations."
                )

            else:

                st.success(
                    "Here are your personalized recommendations."
                )

            st.markdown(
                '<div class="section-title">'
                '🎯 Recommended For You'
                '</div>',
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # RESULTS
            # ------------------------------------------------

            if new_recommendations:

                recommendation_df = (
                    pd.DataFrame(
                        new_recommendations
                    )
                )

                recommendation_df = (
                    recommendation_df.merge(
                        books[
                            [
                                "ISBN",
                                "Book-Title",
                                "Book-Author",
                                "Publisher"
                            ]
                        ],
                        on="ISBN",
                        how="left"
                    )
                )

                for rank, (_, row) in enumerate(
                    recommendation_df.iterrows(),
                    start=1
                ):

                    title = str(
                        row["Book-Title"]
                    )

                    author = str(
                        row["Book-Author"]
                    )

                    publisher = str(
                        row["Publisher"]
                    )

                    score = float(
                        row["Content Score"]
                    )

                    st.markdown(
                        f"""
                        <div class="book-card">
                            <div class="book-title">
                                #{rank} 📚 {title}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.caption(
                        f"{author} • {publisher}"
                    )

                    st.metric(
                        "Content Similarity Score",
                        f"{score:.3f}"
                    )

                    st.markdown(
                        """
                        <div class="explanation">
                            🧠 <b>Why was this recommended?</b><br>
                            This book was recommended because its
                            content is similar to the books you
                            selected and rated highly.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            else:

                st.warning(
                    "We could not generate recommendations "
                    "from the selected books."
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "AI-Based Intelligent Book Recommendation System "
    "| Content-Based + Collaborative + Hybrid Recommendation"
)