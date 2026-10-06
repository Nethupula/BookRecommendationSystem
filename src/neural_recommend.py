import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# ============================================================
# 1. FILE PATHS
# ============================================================

MODEL_PATH = "models/neural_recommender.keras"
RATINGS_PATH = "data/processed/ratings_clean.csv"
BOOKS_PATH = "data/processed/books_clean.csv"

USER_MAPPING_PATH = "models/user_to_index.joblib"
BOOK_MAPPING_PATH = "models/book_to_index.joblib"


# ============================================================
# 2. LOAD MODEL AND DATA
# ============================================================

print("Loading neural network model...")

# safe_mode=False is required because the trained model
# contains a Keras Lambda layer.
model = tf.keras.models.load_model(
    MODEL_PATH,
    safe_mode=False
)

ratings = pd.read_csv(
    RATINGS_PATH,
    encoding="latin-1"
)

books = pd.read_csv(
    BOOKS_PATH,
    encoding="latin-1"
)

user_to_index = joblib.load(
    USER_MAPPING_PATH
)

book_to_index = joblib.load(
    BOOK_MAPPING_PATH
)

print("Model and data loaded successfully.")


# ============================================================
# 3. SELECT USER
# ============================================================

target_user = "8"

if target_user not in user_to_index:

    print(
        f"\nUser {target_user} was not found."
    )

    print(
        "Please select a user that exists in the dataset."
    )

    exit()

user_index = user_to_index[target_user]


# ============================================================
# 4. FIND BOOKS ALREADY RATED BY USER
# ============================================================

user_rated_books = set(
    ratings.loc[
        ratings["User-ID"].astype(str) == target_user,
        "ISBN"
    ].astype(str)
)

print("\nUser:", target_user)

print(
    "Books already rated:",
    len(user_rated_books)
)


# ============================================================
# 5. CREATE CANDIDATE BOOKS
# ============================================================

candidate_books = []

for isbn in book_to_index.keys():

    isbn = str(isbn)

    if isbn not in user_rated_books:

        candidate_books.append(isbn)


print(
    "Candidate books:",
    len(candidate_books)
)


# ============================================================
# 6. PREPARE NEURAL NETWORK INPUT
# ============================================================

candidate_indices = np.array(
    [
        book_to_index[isbn]
        for isbn in candidate_books
    ],
    dtype=np.int32
)

user_indices = np.full(
    len(candidate_indices),
    user_index,
    dtype=np.int32
)


# ============================================================
# 7. PREDICT RATINGS
# ============================================================

print("\nGenerating predictions...")

predicted_ratings = model.predict(
    [
        user_indices,
        candidate_indices
    ],
    batch_size=1024,
    verbose=0
).flatten()


# ============================================================
# 8. ENSURE VALID RATING RANGE
# ============================================================

predicted_ratings = np.clip(
    predicted_ratings,
    1.0,
    10.0
)


# ============================================================
# 9. CREATE RECOMMENDATION DATAFRAME
# ============================================================

recommendations = pd.DataFrame({

    "ISBN": candidate_books,

    "Predicted_Rating": predicted_ratings

})


# ============================================================
# 10. FILTER SUPPLEMENTARY / NON-BOOK MATERIAL
# ============================================================

def is_non_book_material(title):

    if pd.isna(title):
        return True

    title = str(title).lower()

    keywords = [

        "study guide",
        "study guides",
        "student guide",
        "teacher guide",

        "cliffs notes",
        "cliffsnotes",

        "sparknotes",

        "reading guide",
        "revision guide",
        "exam guide",

        "workbook",
        "activity book",
        "coloring book",
        "colouring book",

        "puzzle book",
        "sticker book",
        "maze book",
        "quiz book",

        "exercise book",
        "practice book",

        "answer key",
        "solution manual",
        "test bank",

        "lesson plan",

        "literary criticism",
        "literary analysis",

        "companion guide",
        "critical notes",

        "teacher edition",
        "student edition"
    ]

    return any(
        keyword in title
        for keyword in keywords
    )


# ============================================================
# 11. ADD BOOK INFORMATION
# ============================================================

recommendations = recommendations.merge(

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


# ============================================================
# 12. REMOVE NON-BOOK MATERIAL
# ============================================================

recommendations = recommendations[
    ~recommendations[
        "Book-Title"
    ].apply(is_non_book_material)
]


# ============================================================
# 13. REMOVE DUPLICATE TITLES
# ============================================================

recommendations = recommendations.drop_duplicates(
    subset=["Book-Title"]
)


# ============================================================
# 14. REMOVE EMPTY TITLES
# ============================================================

recommendations = recommendations[
    recommendations["Book-Title"].notna()
]

recommendations = recommendations[
    recommendations["Book-Title"].astype(str).str.strip() != ""
]


# ============================================================
# 15. SORT BY PREDICTED RATING
# ============================================================

recommendations = recommendations.sort_values(
    "Predicted_Rating",
    ascending=False
)


# ============================================================
# 16. GET TOP 10
# ============================================================

top_recommendations = recommendations.head(10)


# ============================================================
# 17. DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 75)

print(
    "NEURAL NETWORK BOOK RECOMMENDATIONS"
)

print("=" * 75)

print(
    f"\nRecommendations for User {target_user}\n"
)


for rank, (_, row) in enumerate(
    top_recommendations.iterrows(),
    start=1
):

    title = row["Book-Title"]

    author = row["Book-Author"]

    predicted_rating = row[
        "Predicted_Rating"
    ]

    print(
        f"{rank}. {title}"
    )

    print(
        f"   Author: {author}"
    )

    print(
        f"   Author: {author}"
    )

    print(
        f"   Predicted Rating: "
        f"{predicted_rating:.2f}/10"
    )

    print()


# ============================================================
# 18. FINAL MESSAGE
# ============================================================

print("=" * 75)

print(
    "Recommendation generation completed."
)

print(
    f"Top {len(top_recommendations)} recommendations "
    f"generated for User {target_user}."
)