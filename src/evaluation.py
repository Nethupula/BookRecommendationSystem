import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import joblib


# ============================================================
# SETTINGS
# ============================================================

TOP_K = 5
MIN_RATING = 8
MIN_USER_RATINGS = 10
MIN_BOOK_RATINGS = 5
NUM_USERS = 100


# ============================================================
# LOAD RATINGS
# ============================================================

print("Loading ratings...")

ratings = pd.read_csv(
    "data/processed/ratings_clean.csv"
)

print(f"Total ratings: {len(ratings)}")


# ============================================================
# FILTER DATA
# ============================================================

print("\nFiltering users...")

user_counts = ratings["User-ID"].value_counts()

active_users = user_counts[
    user_counts >= MIN_USER_RATINGS
].index

ratings = ratings[
    ratings["User-ID"].isin(active_users)
].copy()

print(
    f"Ratings after user filtering: "
    f"{len(ratings)}"
)


print("\nFiltering books...")

book_counts = ratings["ISBN"].value_counts()

frequent_books = book_counts[
    book_counts >= MIN_BOOK_RATINGS
].index

ratings = ratings[
    ratings["ISBN"].isin(frequent_books)
].copy()

print(
    f"Ratings after book filtering: "
    f"{len(ratings)}"
)


# ============================================================
# FIND USERS WITH ENOUGH RATINGS
# ============================================================

user_rating_counts = (
    ratings.groupby("User-ID")
    .size()
)

eligible_users = user_rating_counts[
    user_rating_counts >= MIN_USER_RATINGS
].index


# Only users with at least 2 highly rated books
liked_counts = (
    ratings[
        ratings["Book-Rating"] >= MIN_RATING
    ]
    .groupby("User-ID")
    .size()
)

eligible_users = eligible_users.intersection(
    liked_counts[
        liked_counts >= 2
    ].index
)


# Select users
selected_users = eligible_users[:NUM_USERS]

ratings = ratings[
    ratings["User-ID"].isin(
        selected_users
    )
].copy()

print(
    f"Users selected for evaluation: "
    f"{len(selected_users)}"
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

train_parts = []
test_items = {}


for user_id in selected_users:

    user_data = ratings[
        ratings["User-ID"] == user_id
    ].copy()

    liked_books = user_data[
        user_data["Book-Rating"] >= MIN_RATING
    ].copy()

    # Use 20% of liked books as test data
    test_count = max(
        1,
        int(len(liked_books) * 0.2)
    )

    # Keep at least one liked book for training
    if len(liked_books) - test_count < 1:
        test_count = 1

    test_books = liked_books.tail(
        test_count
    )

    test_items[user_id] = set(
        test_books["ISBN"]
    )

    test_isbns = set(
        test_books["ISBN"]
    )

    train_user_data = user_data[
        ~user_data["ISBN"].isin(
            test_isbns
        )
    ]

    train_parts.append(
        train_user_data
    )


train_ratings = pd.concat(
    train_parts,
    ignore_index=True
)


print(
    f"Training ratings: "
    f"{len(train_ratings)}"
)

print(
    f"Test users: "
    f"{len(test_items)}"
)


# ============================================================
# CREATE USER-BOOK MATRIX
# ============================================================

user_ids = train_ratings[
    "User-ID"
].unique()

book_ids = train_ratings[
    "ISBN"
].unique()


user_to_index = {
    user_id: i
    for i, user_id in enumerate(
        user_ids
    )
}

book_to_index = {
    isbn: i
    for i, isbn in enumerate(
        book_ids
    )
}


rows = train_ratings[
    "User-ID"
].map(
    user_to_index
)

cols = train_ratings[
    "ISBN"
].map(
    book_to_index
)

values = train_ratings[
    "Book-Rating"
].values


user_book_matrix = csr_matrix(
    (
        values,
        (rows, cols)
    ),
    shape=(
        len(user_ids),
        len(book_ids)
    )
)


print(
    f"Evaluation matrix: "
    f"{user_book_matrix.shape}"
)


# ============================================================
# COLLABORATIVE MODEL
# ============================================================

print(
    "\nTraining evaluation "
    "collaborative model..."
)

collab_model = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=6,
    n_jobs=-1
)

collab_model.fit(
    user_book_matrix
)

print(
    "Collaborative evaluation "
    "model ready."
)


# ============================================================
# LOAD CONTENT MODEL
# ============================================================

print(
    "\nLoading content model..."
)

content_model = joblib.load(
    "models/content_model.joblib"
)

tfidf_matrix = joblib.load(
    "models/tfidf_matrix.joblib"
)

content_books = joblib.load(
    "models/content_books.joblib"
)

print(
    "Content model loaded."
)


# ============================================================
# NORMALIZE
# ============================================================

def normalize_scores(scores):

    if not scores:
        return {}

    values = np.array(
        list(scores.values())
    )

    minimum = values.min()
    maximum = values.max()

    if maximum == minimum:
        return {
            key: 1.0
            for key in scores
        }

    return {
        key:
        (value - minimum)
        / (maximum - minimum)

        for key, value
        in scores.items()
    }


# ============================================================
# CONTENT RECOMMENDATIONS
# ============================================================

def get_content_recommendations(
    user_id
):

    user_data = train_ratings[
        train_ratings["User-ID"]
        == user_id
    ]

    liked_books = user_data[
        user_data["Book-Rating"]
        >= MIN_RATING
    ]

    scores = {}

    for _, row in liked_books.iterrows():

        isbn = row["ISBN"]

        matches = content_books[
            content_books["ISBN"]
            == isbn
        ]

        if matches.empty:
            continue

        book_index = matches.index[0]

        distances, indices = (
            content_model.kneighbors(
                tfidf_matrix[book_index],
                n_neighbors=6
            )
        )

        for distance, index in zip(
            distances[0],
            indices[0]
        ):

            recommended_isbn = (
                content_books.iloc[
                    index
                ]["ISBN"]
            )

            if recommended_isbn == isbn:
                continue

            similarity = 1 - distance

            score = (
                similarity
                *
                (row["Book-Rating"] / 10)
            )

            scores[
                recommended_isbn
            ] = max(
                scores.get(
                    recommended_isbn,
                    0
                ),
                score
            )

    # Remove books already used for training
    known_books = set(
        train_ratings[
            train_ratings["User-ID"]
            == user_id
        ]["ISBN"]
    )

    scores = {
        isbn: score
        for isbn, score
        in scores.items()
        if isbn not in known_books
    }

    return normalize_scores(
        scores
    )


# ============================================================
# COLLABORATIVE RECOMMENDATIONS
# ============================================================

def get_collaborative_recommendations(
    user_id
):

    if user_id not in user_to_index:
        return {}

    user_index = user_to_index[
        user_id
    ]

    distances, indices = (
        collab_model.kneighbors(
            user_book_matrix[
                user_index
            ],
            n_neighbors=6
        )
    )

    scores = {}

    for distance, similar_index in zip(
        distances[0][1:],
        indices[0][1:]
    ):

        similarity = 1 - distance

        similar_user = user_ids[
            similar_index
        ]

        similar_ratings = (
            train_ratings[
                train_ratings["User-ID"]
                == similar_user
            ]
        )

        for _, row in (
            similar_ratings.iterrows()
        ):

            if row["Book-Rating"] < MIN_RATING:
                continue

            isbn = row["ISBN"]

            score = (
                similarity
                * row["Book-Rating"]
            )

            scores[isbn] = max(
                scores.get(
                    isbn,
                    0
                ),
                score
            )

    # Remove books already known
    known_books = set(
        train_ratings[
            train_ratings["User-ID"]
            == user_id
        ]["ISBN"]
    )

    scores = {
        isbn: score
        for isbn, score
        in scores.items()
        if isbn not in known_books
    }

    return normalize_scores(
        scores
    )


# ============================================================
# HYBRID RECOMMENDATIONS
# ============================================================

def get_hybrid_recommendations(
    user_id
):

    content_scores = (
        get_content_recommendations(
            user_id
        )
    )

    collaborative_scores = (
        get_collaborative_recommendations(
            user_id
        )
    )

    candidates = (
        set(content_scores)
        |
        set(collaborative_scores)
    )

    scores = {}

    for isbn in candidates:

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

        scores[isbn] = (
            0.5 * content_score
            +
            0.5 * collaborative_score
        )

    return scores


# ============================================================
# EVALUATION
# ============================================================

content_precision_sum = 0
content_recall_sum = 0

collab_precision_sum = 0
collab_recall_sum = 0

hybrid_precision_sum = 0
hybrid_recall_sum = 0

evaluated_users = 0


print(
    "\n=========================================="
)

print(
    "MODEL EVALUATION"
)

print(
    "=========================================="
)


for user_id, actual_books in (
    test_items.items()
):

    if user_id not in user_to_index:
        continue

    content_scores = (
        get_content_recommendations(
            user_id
        )
    )

    collaborative_scores = (
        get_collaborative_recommendations(
            user_id
        )
    )

    hybrid_scores = (
        get_hybrid_recommendations(
            user_id
        )
    )


    content_top = sorted(
        content_scores,
        key=content_scores.get,
        reverse=True
    )[:TOP_K]


    collaborative_top = sorted(
        collaborative_scores,
        key=collaborative_scores.get,
        reverse=True
    )[:TOP_K]


    hybrid_top = sorted(
        hybrid_scores,
        key=hybrid_scores.get,
        reverse=True
    )[:TOP_K]


    content_hits = len(
        set(content_top)
        &
        actual_books
    )

    collaborative_hits = len(
        set(collaborative_top)
        &
        actual_books
    )

    hybrid_hits = len(
        set(hybrid_top)
        &
        actual_books
    )


    # Precision
    content_precision_sum += (
        content_hits / TOP_K
    )

    collab_precision_sum += (
        collaborative_hits / TOP_K
    )

    hybrid_precision_sum += (
        hybrid_hits / TOP_K
    )


    # Recall
    content_recall_sum += (
        content_hits
        / len(actual_books)
    )

    collab_recall_sum += (
        collaborative_hits
        / len(actual_books)
    )

    hybrid_recall_sum += (
        hybrid_hits
        / len(actual_books)
    )


    evaluated_users += 1


# ============================================================
# FINAL METRICS
# ============================================================

if evaluated_users == 0:

    print(
        "No users could be evaluated."
    )

    exit()


content_precision = (
    content_precision_sum
    / evaluated_users
)

content_recall = (
    content_recall_sum
    / evaluated_users
)


collab_precision = (
    collab_precision_sum
    / evaluated_users
)

collab_recall = (
    collab_recall_sum
    / evaluated_users
)


hybrid_precision = (
    hybrid_precision_sum
    / evaluated_users
)

hybrid_recall = (
    hybrid_recall_sum
    / evaluated_users
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "Model": [
        "Content-Based",
        "Collaborative Filtering",
        "Hybrid"
    ],
    "Precision@5": [
        content_precision,
        collab_precision,
        hybrid_precision
    ],
    "Recall@5": [
        content_recall,
        collab_recall,
        hybrid_recall
    ]
})


results.to_csv(
    "data/evaluation_results.csv",
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print(
    "\n=========================================="
)

print(
    "EVALUATION RESULTS"
)

print(
    "=========================================="
)

print(
    f"Users evaluated: "
    f"{evaluated_users}"
)

print(
    f"Top-K: {TOP_K}"
)


print("\n")

print(
    results.to_string(
        index=False
    )
)


print(
    "\nEvaluation results saved to:"
)

print(
    "data/evaluation_results.csv"
)


print(
    "\n=========================================="
)

print(
    "Evaluation completed."
)

print(
    "=========================================="
)