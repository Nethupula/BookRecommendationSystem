import pandas as pd
import os

# ==========================================
# 1. LOAD DATASETS
# ==========================================

print("Loading datasets...")

books = pd.read_csv(
    "data/Books.csv",
    encoding="latin-1",
    low_memory=False
)

ratings = pd.read_csv(
    "data/Ratings.csv",
    encoding="latin-1"
)

users = pd.read_csv(
    "data/Users.csv",
    encoding="latin-1"
)

print("Datasets loaded successfully.")


# ==========================================
# 2. CLEAN BOOKS DATA
# ==========================================

print("\nCleaning Books dataset...")

# Remove unnecessary image URL columns
books = books.drop(
    columns=["Image-URL-S", "Image-URL-M", "Image-URL-L"]
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

# Remove duplicate rating records
ratings = ratings.drop_duplicates()

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

os.makedirs("data/processed", exist_ok=True)


# ==========================================
# 7. SAVE CLEANED DATA
# ==========================================

books.to_csv(
    "data/processed/books_clean.csv",
    index=False
)

ratings.to_csv(
    "data/processed/ratings_clean.csv",
    index=False
)

users.to_csv(
    "data/processed/users_clean.csv",
    index=False
)

ratings_with_books.to_csv(
    "data/processed/ratings_with_books.csv",
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
print("data/processed/")