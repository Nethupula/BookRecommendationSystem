import pandas as pd

# ==============================
# LOAD DATASETS
# ==============================

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


# ==============================
# BASIC DATASET INFORMATION
# ==============================

print("\n========== DATASET SIZES ==========")
print("Books:", books.shape)
print("Ratings:", ratings.shape)
print("Users:", users.shape)


# ==============================
# MISSING VALUES
# ==============================

print("\n========== MISSING VALUES ==========")

print("\nBooks:")
print(books.isnull().sum())

print("\nRatings:")
print(ratings.isnull().sum())

print("\nUsers:")
print(users.isnull().sum())


# ==============================
# DUPLICATE RECORDS
# ==============================

print("\n========== DUPLICATES ==========")

print("Duplicate books:", books.duplicated().sum())
print("Duplicate ratings:", ratings.duplicated().sum())
print("Duplicate users:", users.duplicated().sum())


# ==============================
# UNIQUE VALUES
# ==============================

print("\n========== UNIQUE VALUES ==========")

print("Unique books (ISBN):", books["ISBN"].nunique())
print("Unique users in ratings:", ratings["User-ID"].nunique())
print("Unique rated books:", ratings["ISBN"].nunique())


# ==============================
# RATING DISTRIBUTION
# ==============================

print("\n========== RATING DISTRIBUTION ==========")
print(ratings["Book-Rating"].value_counts().sort_index())


# ==============================
# BOOK INFORMATION
# ==============================

print("\n========== BOOK DATA SAMPLE ==========")
print(books.head())


# ==============================
# RATING INFORMATION
# ==============================

print("\n========== RATING DATA SAMPLE ==========")
print(ratings.head())