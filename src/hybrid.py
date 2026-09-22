import pandas as pd
import numpy as np
import joblib
from scipy.sparse import csr_matrix


# ==========================================
# 1. LOAD CONTENT-BASED MODEL
# ==========================================

print("Loading content-based model...")

content_books = joblib.load(
    "models/content_books.joblib"
)

content_model = joblib.load(
    "models/content_model.joblib"
)

tfidf_matrix = joblib.load(
    "models/tfidf_matrix.joblib"
)

print("Content-based model loaded.")


# ==========================================
# 2. LOAD COLLABORATIVE MODEL
# ==========================================

print("\nLoading collaborative model...")

collaborative_model = joblib.load(
    "models/collaborative_model.joblib"
)

print("Collaborative model loaded.")


# ==========================================
# 3. LOAD RATINGS
# ==========================================

print("\nLoading ratings...")

ratings = pd.read_csv(
    "data/processed/ratings_clean.csv"
)

print("Ratings loaded.")
print("Ratings:", len(ratings))


# ==========================================
# 4. PREPARE COLLABORATIVE DATA
# ==========================================

print("\nPreparing collaborative data...")

user_counts = ratings["User-ID"].value_counts()

active_users = user_counts[
    user_counts >= 5
].index

ratings_filtered = ratings[
    ratings["User-ID"].isin(active_users)
]

book_counts = ratings_filtered["ISBN"].value_counts()

popular_books = book_counts[
    book_counts >= 5
].index

ratings_filtered = ratings_filtered[
    ratings_filtered["ISBN"].isin(popular_books)
]

user_book_matrix = ratings_filtered.pivot_table(
    index="User-ID",
    columns="ISBN",
    values="Book-Rating"
)

sparse_matrix = csr_matrix(
    user_book_matrix.fillna(0).values
)

print("Collaborative data prepared.")
print("Matrix shape:", sparse_matrix.shape)


# ==========================================
# 5. CONTENT RECOMMENDATIONS
# ==========================================

def get_content_recommendations_for_book(
    isbn,
    number_of_candidates=20
):

    matches = content_books[
        content_books["ISBN"] == isbn
    ]

    if matches.empty:
        return {}

    book_index = matches.index[0]

    book_vector = tfidf_matrix[book_index]

    distances, indices = content_model.kneighbors(
        book_vector,
        n_neighbors=number_of_candidates + 1
    )

    recommendations = {}

    for distance, index in zip(
        distances[0],
        indices[0]
    ):

        recommended_isbn = content_books.iloc[
            index
        ]["ISBN"]

        if recommended_isbn == isbn:
            continue

        similarity = 1 - distance

        recommendations[
            recommended_isbn
        ] = max(
            recommendations.get(
                recommended_isbn,
                0
            ),
            similarity
        )

    return recommendations


# ==========================================
# 6. PERSONALIZED CONTENT SCORE
# ==========================================

def get_personalized_content_scores(
    user_id
):

    user_ratings = ratings[
        ratings["User-ID"] == user_id
    ]

    # Use books rated 5 or higher
    liked_books = user_ratings[
        user_ratings["Book-Rating"] >= 5
    ]

    print(
        "\nBooks liked by user:",
        len(liked_books)
    )

    content_scores = {}

    for _, row in liked_books.iterrows():

        isbn = row["ISBN"]
        rating = row["Book-Rating"]

        recommendations = (
            get_content_recommendations_for_book(
                isbn
            )
        )

        # Stronger ratings contribute more
        rating_weight = rating / 10

        for recommended_isbn, similarity in (
            recommendations.items()
        ):

            score = (
                similarity *
                rating_weight
            )

            if recommended_isbn not in content_scores:
                content_scores[
                    recommended_isbn
                ] = 0

            content_scores[
                recommended_isbn
            ] += score

    # Normalize scores to 0-1
    if content_scores:

        max_score = max(
            content_scores.values()
        )

        if max_score > 0:

            for isbn in content_scores:

                content_scores[isbn] /= (
                    max_score
                )

    return content_scores


# ==========================================
# 7. COLLABORATIVE SCORE
# ==========================================

def get_collaborative_recommendations(
    user_id,
    number_of_similar_users=5
):

    if user_id not in user_book_matrix.index:

        print(
            "\nUser not found in collaborative data."
        )

        return {}

    user_index = (
        user_book_matrix.index.get_loc(
            user_id
        )
    )

    user_vector = sparse_matrix[
        user_index
    ]

    distances, indices = (
        collaborative_model.kneighbors(
            user_vector,
            n_neighbors=number_of_similar_users + 1
        )
    )

    user_rated_books = set(
        user_book_matrix.loc[user_id]
        .dropna()
        .index
    )

    recommendation_scores = {}

    for i in range(
        1,
        len(indices[0])
    ):

        similar_user_index = (
            indices[0][i]
        )

        similarity = (
            1 - distances[0][i]
        )

        similar_user = (
            user_book_matrix.index[
                similar_user_index
            ]
        )

        similar_user_ratings = (
            user_book_matrix
            .loc[similar_user]
            .dropna()
        )

        for isbn, rating in (
            similar_user_ratings.items()
        ):

            if isbn in user_rated_books:
                continue

            if rating >= 7:

                score = (
                    similarity *
                    rating
                )

                if isbn not in recommendation_scores:

                    recommendation_scores[
                        isbn
                    ] = 0

                recommendation_scores[
                    isbn
                ] += score

    # Normalize
    if recommendation_scores:

        max_score = max(
            recommendation_scores.values()
        )

        if max_score > 0:

            for isbn in recommendation_scores:

                recommendation_scores[
                    isbn
                ] /= max_score

    return recommendation_scores


# ==========================================
# 8. HYBRID RECOMMENDATION
# ==========================================

def hybrid_recommend(
    user_id,
    number_of_recommendations=5
):

    print("\n==========================================")
    print("HYBRID RECOMMENDATION SYSTEM")
    print("==========================================")

    print("\nUser:", user_id)

    # -------------------------------
    # CONTENT SCORE
    # -------------------------------

    content_scores = (
        get_personalized_content_scores(
            user_id
        )
    )

    print(
        "Content candidates:",
        len(content_scores)
    )

    # -------------------------------
    # COLLABORATIVE SCORE
    # -------------------------------

    collaborative_scores = (
        get_collaborative_recommendations(
            user_id
        )
    )

    print(
        "Collaborative candidates:",
        len(collaborative_scores)
    )

    # -------------------------------
    # COMBINE CANDIDATES
    # -------------------------------

    candidate_books = set(
        content_scores.keys()
    ).union(
        collaborative_scores.keys()
    )

    hybrid_scores = {}

    for isbn in candidate_books:

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

        # 50% Content + 50% Collaborative

        hybrid_score = (
            0.5 * content_score
            +
            0.5 * collaborative_score
        )

        hybrid_scores[
            isbn
        ] = hybrid_score

    # -------------------------------
    # SORT
    # -------------------------------

    ranked_books = sorted(
        hybrid_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # -------------------------------
    # DISPLAY
    # -------------------------------

    print("\n==========================================")
    print("FINAL HYBRID RECOMMENDATIONS")
    print("==========================================")

    count = 0

    for isbn, hybrid_score in ranked_books:

        book_info = content_books[
            content_books["ISBN"] == isbn
        ]

        if book_info.empty:
            continue

        title = str(
            book_info.iloc[0]["Book-Title"]
        )

        author = str(
            book_info.iloc[0]["Book-Author"]
        )

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

        print(
            f"\n{count + 1}. "
            f"{title} "
            f"by {author}"
        )

        print(
            f"   Content Score: "
            f"{content_score:.3f}"
        )

        print(
            f"   Collaborative Score: "
            f"{collaborative_score:.3f}"
        )

        print(
            f"   Hybrid Score: "
            f"{hybrid_score:.3f}"
        )

        count += 1

        if count == number_of_recommendations:
            break


# ==========================================
# 9. TEST
# ==========================================

test_user = 8

hybrid_recommend(
    test_user,
    5
)