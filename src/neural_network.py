import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Flatten,
    Concatenate,
    Dense,
    Dropout,
    Lambda
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

RATINGS_PATH = "data/processed/ratings_clean.csv"

MODEL_PATH = "models/neural_recommender.keras"
USER_MAPPING_PATH = "models/user_to_index.joblib"
BOOK_MAPPING_PATH = "models/book_to_index.joblib"
HISTORY_PATH = "models/neural_training_history.joblib"

RANDOM_STATE = 42

# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("NEURAL NETWORK BOOK RECOMMENDATION SYSTEM")
print("=" * 70)

ratings = pd.read_csv(RATINGS_PATH)

print("\nOriginal dataset shape:", ratings.shape)

# Keep required columns
ratings = ratings[
    ["User-ID", "ISBN", "Book-Rating"]
].copy()

# ============================================================
# 3. CLEAN DATA
# ============================================================

ratings = ratings.dropna()

ratings["User-ID"] = ratings["User-ID"].astype(str)
ratings["ISBN"] = ratings["ISBN"].astype(str)

ratings["Book-Rating"] = pd.to_numeric(
    ratings["Book-Rating"],
    errors="coerce"
)

ratings = ratings.dropna()

# Keep ratings within the valid 1–10 range
ratings = ratings[
    (ratings["Book-Rating"] >= 1) &
    (ratings["Book-Rating"] <= 10)
].copy()

print("Clean dataset shape:", ratings.shape)

# ============================================================
# 4. ENCODE USERS
# ============================================================

unique_users = ratings["User-ID"].unique()

user_to_index = {
    user_id: index
    for index, user_id in enumerate(unique_users)
}

ratings["user_index"] = ratings["User-ID"].map(
    user_to_index
)

# ============================================================
# 5. ENCODE BOOKS
# ============================================================

unique_books = ratings["ISBN"].unique()

book_to_index = {
    isbn: index
    for index, isbn in enumerate(unique_books)
}

ratings["book_index"] = ratings["ISBN"].map(
    book_to_index
)

num_users = len(unique_users)
num_books = len(unique_books)

print("\nDataset Information")
print("-" * 70)
print("Number of users:", num_users)
print("Number of books:", num_books)
print("Number of ratings:", len(ratings))

# ============================================================
# 6. PREPARE INPUTS AND TARGET
# ============================================================

X = ratings[
    ["user_index", "book_index"]
].values

y = ratings[
    "Book-Rating"
].values.astype(np.float32)

# ============================================================
# 7. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE
)

print("\nDataset Split")
print("-" * 70)
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))

# ============================================================
# 8. BUILD NEURAL NETWORK
# ============================================================

print("\nBuilding neural network...")

embedding_size = 32

# User input
user_input = Input(
    shape=(1,),
    name="user_input"
)

# Book input
book_input = Input(
    shape=(1,),
    name="book_input"
)

# User embedding
user_embedding = Embedding(
    input_dim=num_users,
    output_dim=embedding_size,
    name="user_embedding"
)(user_input)

# Book embedding
book_embedding = Embedding(
    input_dim=num_books,
    output_dim=embedding_size,
    name="book_embedding"
)(book_input)

# Flatten embeddings
user_vector = Flatten(
    name="user_vector"
)(user_embedding)

book_vector = Flatten(
    name="book_vector"
)(book_embedding)

# Combine user and book representations
combined = Concatenate(
    name="user_book_combination"
)([
    user_vector,
    book_vector
])

# Dense layer 1
x = Dense(
    128,
    activation="relu",
    name="dense_128"
)(combined)

# Dropout
x = Dropout(
    0.2,
    name="dropout"
)(x)

# Dense layer 2
x = Dense(
    64,
    activation="relu",
    name="dense_64"
)(x)

# Sigmoid gives 0–1
x = Dense(
    1,
    activation="sigmoid",
    name="sigmoid_output"
)(x)

# Convert 0–1 to 1–10
output = Lambda(
    lambda value: value * 9.0 + 1.0,
    name="rating_scale"
)(x)

# Create model
model = Model(
    inputs=[
        user_input,
        book_input
    ],
    outputs=output
)

# ============================================================
# 9. COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse",
    metrics=["mae"]
)

# ============================================================
# 10. DISPLAY ARCHITECTURE
# ============================================================

print("\nNeural Network Architecture")
print("=" * 70)

model.summary()

# ============================================================
# 11. TRAIN MODEL
# ============================================================

print("\nStarting neural network training...")
print("=" * 70)

history = model.fit(
    [
        X_train[:, 0],
        X_train[:, 1]
    ],
    y_train,

    validation_data=(
        [
            X_val[:, 0],
            X_val[:, 1]
        ],
        y_val
    ),

    epochs=10,
    batch_size=512,

    verbose=1
)

print("\nTraining completed successfully!")

# ============================================================
# 12. EVALUATE MODEL
# ============================================================

print("\nEvaluating model on test data...")
print("=" * 70)

test_loss, test_mae = model.evaluate(
    [
        X_test[:, 0],
        X_test[:, 1]
    ],
    y_test,
    verbose=1
)

test_rmse = np.sqrt(test_loss)

print("\nFinal Test Results")
print("=" * 70)
print(f"Test MSE : {test_loss:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test MAE : {test_mae:.4f}")

# ============================================================
# 13. CHECK PREDICTION RANGE
# ============================================================

sample_predictions = model.predict(
    [
        X_test[:1000, 0],
        X_test[:1000, 1]
    ],
    verbose=0
).flatten()

print("\nPrediction Range Check")
print("-" * 70)
print(
    f"Minimum prediction: "
    f"{sample_predictions.min():.2f}"
)

print(
    f"Maximum prediction: "
    f"{sample_predictions.max():.2f}"
)

# ============================================================
# 14. SAVE MODEL
# ============================================================

os.makedirs("models", exist_ok=True)

model.save(MODEL_PATH)

# ============================================================
# 15. SAVE MAPPINGS
# ============================================================

joblib.dump(
    user_to_index,
    USER_MAPPING_PATH
)

joblib.dump(
    book_to_index,
    BOOK_MAPPING_PATH
)

# ============================================================
# 16. SAVE TRAINING HISTORY
# ============================================================

joblib.dump(
    history.history,
    HISTORY_PATH
)

# ============================================================
# 17. FINAL INFORMATION
# ============================================================

print("\nSaved Files")
print("=" * 70)

print(MODEL_PATH)
print(USER_MAPPING_PATH)
print(BOOK_MAPPING_PATH)
print(HISTORY_PATH)

print("\nNeural network training pipeline completed successfully!")