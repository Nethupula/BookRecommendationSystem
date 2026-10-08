import pandas as pd
import os
try:
    from .pipeline_support import ROOT, require_files, read_raw
except ImportError:
    from pipeline_support import ROOT, require_files, read_raw


def main():
    require_files([ROOT / "data/Books.csv", ROOT / "data/Ratings.csv", ROOT / "data/Users.csv"])

    # ==========================================
    # 1. LOAD DATASETS
    # ==========================================

    print("Loading datasets...")

    books = read_raw(ROOT / "data/Books.csv")
    ratings = read_raw(ROOT / "data/Ratings.csv")
    users = read_raw(ROOT / "data/Users.csv")

    print("Datasets loaded successfully.")


    # ==========================================
    # 2. CLEAN BOOKS DATA
    # ==========================================

    print("\nCleaning Books dataset...")

    # Remove unnecessary image URL columns
    books = books.drop(
        columns=["Image-URL-S", "Image-URL-M", "Image-URL-L"], errors="ignore"
    )

    # Fill missing author and publisher values
    books["Book-Author"] = books["Book-Author"].fillna("Unknown")
    books["Publisher"] = books["Publisher"].fillna("Unknown")

    # Convert year to numeric
    books["Year-Of-Publication"] = pd.to_numeric(
        books["Year-Of-Publication"],
        errors="coerce"
    )

    # Replace invalid/missing years with 0
    books["Year-Of-Publication"] = books["Year-Of-Publication"].fillna(0)

    # Remove duplicate ISBN records
    books = books.drop_duplicates(subset="ISBN")

    print("Clean Books shape:", books.shape)


    # ==========================================
    # 3. CLEAN RATINGS DATA
    # ==========================================

    print("\nCleaning Ratings dataset...")

    ratings["User-ID"] = pd.to_numeric(ratings["User-ID"], errors="coerce")
    ratings["Book-Rating"] = pd.to_numeric(ratings["Book-Rating"], errors="coerce")
    ratings = ratings.dropna(subset=["User-ID", "ISBN", "Book-Rating"])
    ratings = ratings[ratings["User-ID"] % 1 == 0].copy()
    ratings["User-ID"] = ratings["User-ID"].astype(int)
    # Remove duplicate rating records
    ratings = ratings.drop_duplicates(subset=["User-ID", "ISBN"], keep="last")

    # Keep only ratings from 1 to 10
    # Rating 0 represents an implicit/non-explicit interaction
    ratings = ratings[
        ratings["Book-Rating"].between(1, 10)
    ]

    print("Ratings with explicit ratings:", ratings.shape)


    # ==========================================
    # 4. CLEAN USERS DATA
    # ==========================================

    print("\nCleaning Users dataset...")

    # Age is not required for the core recommendation model.
    # Keep User-ID and Location for possible future use.
    users["Age"] = pd.to_numeric(
        users["Age"],
        errors="coerce"
    )

    print("Users shape:", users.shape)


    # ==========================================
    # 5. MATCH RATINGS WITH BOOKS
    # ==========================================

    print("\nMatching ratings with book information...")

    ratings_with_books = ratings.merge(
        books[["ISBN", "Book-Title", "Book-Author", "Publisher",
               "Year-Of-Publication"]],
        on="ISBN",
        how="inner"
    )

    print("Matched ratings:", ratings_with_books.shape)

    # Calculate how many ratings could not be matched
    unmatched_ratings = len(ratings) - len(ratings_with_books)

    print("Unmatched ratings:", unmatched_ratings)


    # ==========================================
    # 6. CREATE DATA FOLDER FOR CLEANED DATA
    # ==========================================

    os.makedirs(str(ROOT / "data/processed"), exist_ok=True)


    # ==========================================
    # 7. SAVE CLEANED DATA
    # ==========================================

    books.to_csv(
        str(ROOT / "data/processed/books_clean.csv"),
        index=False
    )

    ratings.to_csv(
        str(ROOT / "data/processed/ratings_clean.csv"),
        index=False
    )

    users.to_csv(
        str(ROOT / "data/processed/users_clean.csv"),
        index=False
    )

    ratings_with_books.to_csv(
        str(ROOT / "data/processed/ratings_with_books.csv"),
        index=False
    )


    # ==========================================
    # 8. FINAL SUMMARY
    # ==========================================

    print("\n==========================================")
    print("PREPROCESSING COMPLETED")
    print("==========================================")

    print("\nClean Books:", books.shape)
    print("Clean Ratings:", ratings.shape)
    print("Clean Users:", users.shape)
    print("Ratings with Book Information:", ratings_with_books.shape)

    print("\nProcessed files saved to:")
    print(str(ROOT / "data/processed/"))


if __name__ == "__main__":
    main()
