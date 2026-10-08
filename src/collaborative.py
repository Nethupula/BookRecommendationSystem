import pandas as pd
import numpy as np
import joblib
import os
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
try:
    from .pipeline_support import ROOT, require_files, sparse_user_books, known_ratings, similar_neighbors
except ImportError:
    from pipeline_support import ROOT, require_files, sparse_user_books, known_ratings, similar_neighbors


def main():
    require_files([ROOT / "data/processed/books_clean.csv", ROOT / "data/processed/ratings_clean.csv"])

    books = pd.read_csv(
        str(ROOT / "data/processed/books_clean.csv"), dtype={"ISBN": str}
    )

    print("Loading cleaned ratings...")

    ratings = pd.read_csv(
        str(ROOT / "data/processed/ratings_clean.csv"), dtype={"ISBN": str}
    )

    print("Ratings loaded successfully.")
    print("Number of ratings:", len(ratings))

    # ------------------------------------------
    # STEP 1: Filter active users
    # ------------------------------------------

    print("\nFiltering active users...")

    user_counts = ratings["User-ID"].value_counts()

    active_users = user_counts[user_counts >= 5].index

    ratings_filtered = ratings[
        ratings["User-ID"].isin(active_users)
    ]

    print("Ratings after user filtering:", len(ratings_filtered))

    # ------------------------------------------
    # STEP 2: Filter frequently rated books
    # ------------------------------------------

    print("\nFiltering frequently rated books...")

    book_counts = ratings_filtered["ISBN"].value_counts()

    popular_books = book_counts[book_counts >= 5].index

    ratings_filtered = ratings_filtered[
        ratings_filtered["ISBN"].isin(popular_books)
    ]

    print("Ratings after book filtering:", len(ratings_filtered))

    # ------------------------------------------
    # STEP 3: Create User-Book Matrix
    # ------------------------------------------

    print("\nCreating User-Book matrix...")

    user_book_matrix, sparse_matrix = sparse_user_books(ratings_filtered)

    print("Matrix shape:", user_book_matrix.shape)

    print("Sparse matrix created.")
    print("Sparse matrix shape:", sparse_matrix.shape)

    # ------------------------------------------
    # STEP 4: Train Collaborative Model
    # ------------------------------------------

    print("\nTraining collaborative filtering model...")

    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=min(6, len(user_book_matrix)),
        n_jobs=-1
    )

    model.fit(sparse_matrix)

    print("Collaborative filtering model trained successfully.")
    # ------------------------------------------
    # SAVE COLLABORATIVE MODEL
    # ------------------------------------------

    os.makedirs(ROOT / "models", exist_ok=True)

    joblib.dump(
        model,
        str(ROOT / "models/collaborative_model.joblib")
    )

    joblib.dump(
        user_book_matrix,
        str(ROOT / "models/user_book_matrix.joblib")
    )

    joblib.dump(
        sparse_matrix,
        str(ROOT / "models/sparse_user_book_matrix.joblib")
    )

    joblib.dump(
        books,
        str(ROOT / "models/collaborative_books.joblib")
    )

    print("\nCollaborative model saved successfully.")

    # ------------------------------------------
    # STEP 5: Recommendation Function
    # ------------------------------------------

    def recommend_books(user_id, number_of_recommendations=5):

        if user_id not in user_book_matrix.index:
            print("\nUser not found.")
            return

        user_index = user_book_matrix.index.get_loc(user_id)

        user_vector = sparse_matrix[user_index]

        similar_users = [(user_book_matrix.index[index], similarity)
                         for index, similarity in similar_neighbors(model, sparse_matrix, user_index)]

        # Books already rated by the selected user
        user_rated_books = set(known_ratings(user_book_matrix, user_id).index)

        # Store recommendation scores
        recommendation_scores = {}

        # Get books from similar users
        for similar_user, similarity in similar_users:

            similar_user_ratings = known_ratings(user_book_matrix, similar_user)

            for isbn, rating in similar_user_ratings.items():

                # Skip books already rated by the target user
                if isbn in user_rated_books:
                    continue

                # Only consider highly rated books
                if rating >= 7:

                    score = similarity * rating

                    if isbn not in recommendation_scores:
                        recommendation_scores[isbn] = 0

                    recommendation_scores[isbn] += score

        # Sort recommendations by score
        ranked_books = sorted(
            recommendation_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        print("\n==========================================")
        print("COLLABORATIVE BOOK RECOMMENDATIONS")
        print("==========================================")

        print("\nSelected User:")
        print(user_id)

        print("\nRecommended Books:")

        count = 0

        for isbn, score in ranked_books:

            book_info = books[
                books["ISBN"] == isbn
            ]

            if book_info.empty:
                continue

            title = book_info.iloc[0]["Book-Title"]
            author = book_info.iloc[0]["Book-Author"]

            print(
                f"{count + 1}. {title} "
                f"by {author} "
                f"(Score: {score:.3f})"
            )

            count += 1

            if count == number_of_recommendations:
                break

        if count == 0:
            print("No recommendations found.")

    # ------------------------------------------
    # TEST
    # ------------------------------------------

    test_user = user_book_matrix.index[0]

    recommend_books(
        test_user,
        5
    )


if __name__ == "__main__":
    main()
