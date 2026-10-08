import pandas as pd
import joblib
import os
import hashlib
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
try:
    from .pipeline_support import ROOT, require_files, load_content_books
except ImportError:
    from pipeline_support import ROOT, require_files, load_content_books


def main():
    require_files([ROOT / "data/processed/books_clean.csv"])

    # ==========================================
    # 1. LOAD CLEANED BOOK DATA
    # ==========================================

    print("Loading cleaned book data...")

    books = pd.read_csv(
        str(ROOT / "data/processed/books_clean.csv"), dtype={"ISBN": str}
    )

    print("Books loaded successfully.")
    print("Number of books:", len(books))


    # ==========================================
    # 2. PREPARE TEXT FEATURES
    # ==========================================

    books["Book-Title"] = books["Book-Title"].fillna("")
    books["Book-Author"] = books["Book-Author"].fillna("")
    books["Publisher"] = books["Publisher"].fillna("")

    books["text_features"] = (
        books["Book-Title"].astype(str) + " "
        + books["Book-Author"].astype(str) + " "
        + books["Publisher"].astype(str)
    )

    print("Text features prepared.")


    # ==========================================
    # 3. CREATE TF-IDF MODEL
    # ==========================================

    print("\nCreating TF-IDF matrix...")

    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=50000
    )

    tfidf_matrix = tfidf.fit_transform(
        books["text_features"]
    )

    print("TF-IDF matrix created.")
    print("Matrix shape:", tfidf_matrix.shape)


    # ==========================================
    # 4. CREATE NEAREST NEIGHBORS MODEL
    # ==========================================

    print("\nTraining nearest-neighbor model...")

    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=6,
        n_jobs=-1
    )

    model.fit(tfidf_matrix)

    print("Nearest-neighbor model trained successfully.")
    # ------------------------------------------
    # SAVE CONTENT-BASED MODEL
    # ------------------------------------------

    os.makedirs(ROOT / "models", exist_ok=True)

    joblib.dump(
        tfidf,
        str(ROOT / "models/tfidf_vectorizer.joblib")
    )

    joblib.dump(
        model,
        str(ROOT / "models/content_model.joblib")
    )

    joblib.dump(
        books,
        str(ROOT / "models/content_books.joblib")
    )

    joblib.dump(
        tfidf_matrix,
        str(ROOT / "models/tfidf_matrix.joblib")
    )

    (ROOT / "models/content_manifest.json").write_text(json.dumps({
        "catalog_sha256": hashlib.sha256((ROOT / "data/processed/books_clean.csv").read_bytes()).hexdigest(),
        "matrix_shape": list(tfidf_matrix.shape),
    }, indent=2))

    print("\nContent-based model saved successfully.")


    # ==========================================
    # 5. RECOMMEND SIMILAR BOOKS
    # ==========================================

    def recommend_books(book_title, number_of_recommendations=5):

        # Find the selected book
        matches = books[
            books["Book-Title"].str.contains(
                book_title,
                case=False,
                na=False, regex=False
            )
        ]

        if matches.empty:
            print("\nBook not found.")
            return

        # Use the first matching book
        book_index = matches.index[0]

        # Get TF-IDF vector
        book_vector = tfidf_matrix[book_index]

        # Get more candidates
        distances, indices = model.kneighbors(
            book_vector,
            n_neighbors=min(50, len(books))
        )

        print("\n==========================================")
        print("RECOMMENDATIONS")
        print("==========================================")

        print("\nSelected book:")
        print(books.loc[book_index, "Book-Title"])

        print("\nSimilar books:")

        recommended_titles = set()
        count = 0

        for distance, index in zip(distances[0], indices[0]):

            # Skip selected book
            if index == book_index:
                continue

            title = str(books.iloc[index]["Book-Title"])

            # Normalize title for duplicate detection
            normalized_title = (
                title.lower()
                .replace("(paperback)", "")
                .replace("(hardcover)", "")
                .replace("(book 1)", "")
                .replace("(book 2)", "")
                .replace("(book 3)", "")
                .replace("(book 4)", "")
                .replace("(book 5)", "")
                .replace("  ", " ")
                .strip()
            )

            # Skip books with the same normalized title
            if normalized_title in recommended_titles:
                continue

            similarity = 1 - distance

            print(
                f"{count + 1}. "
                f"{title} "
                f"(Similarity: {similarity:.3f})"
            )

            recommended_titles.add(normalized_title)

            count += 1

            if count == number_of_recommendations:
                break

        if count == 0:
            print("No unique recommendations found.")
    # ==========================================
    # 6. TEST THE RECOMMENDATION SYSTEM
    # ==========================================

    recommend_books(
        "Harry Potter",
        5
    )


if __name__ == "__main__":
    main()
