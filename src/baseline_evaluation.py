import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. LOAD DATA
# ============================================================

RATINGS_PATH = "data/processed/ratings_clean.csv"
MODEL_PATH = "models/neural_recommender.keras"

ratings = pd.read_csv(
    RATINGS_PATH,
    encoding="latin-1"
)

ratings = ratings[
    ["User-ID", "ISBN", "Book-Rating"]
].dropna()

ratings["User-ID"] = ratings["User-ID"].astype(str)
ratings["ISBN"] = ratings["ISBN"].astype(str)
ratings["Book-Rating"] = pd.to_numeric(
    ratings["Book-Rating"],
    errors="coerce"
)

ratings = ratings.dropna()

ratings = ratings[
    (ratings["Book-Rating"] >= 1) &
    (ratings["Book-Rating"] <= 10)
].copy()


# ============================================================
# 2. CREATE SAME USER/BOOK ENCODING
# ============================================================

unique_users = ratings["User-ID"].unique()
unique_books = ratings["ISBN"].unique()

user_to_index = {
    user_id: index
    for index, user_id in enumerate(unique_users)
}

book_to_index = {
    isbn: index
    for index, isbn in enumerate(unique_books)
}

ratings["user_index"] = ratings["User-ID"].map(
    user_to_index
)

ratings["book_index"] = ratings["ISBN"].map(
    book_to_index
)


# ============================================================
# 3. PREPARE DATA
# ============================================================

X = ratings[
    ["user_index", "book_index"]
].values

y = ratings[
    "Book-Rating"
].values.astype(np.float32)


# ============================================================
# 4. SAME TRAIN / VALIDATION / TEST SPLIT
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42
)


# ============================================================
# 5. BASELINE MODEL
# ============================================================

baseline_prediction = np.mean(y_train)

baseline_predictions = np.full(
    len(y_test),
    baseline_prediction
)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_predictions
    )
)


# ============================================================
# 6. LOAD TRAINED NEURAL NETWORK
# ============================================================

print("Loading trained neural network...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    safe_mode=False
)


# ============================================================
# 7. NEURAL NETWORK PREDICTIONS
# ============================================================

neural_predictions = model.predict(
    [
        X_test[:, 0],
        X_test[:, 1]
    ],
    batch_size=1024,
    verbose=0
).flatten()

neural_predictions = np.clip(
    neural_predictions,
    1.0,
    10.0
)


# ============================================================
# 8. NEURAL NETWORK METRICS
# ============================================================

neural_mae = mean_absolute_error(
    y_test,
    neural_predictions
)

neural_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        neural_predictions
    )
)


# ============================================================
# 9. CALCULATE IMPROVEMENT
# ============================================================

mae_improvement = (
    (baseline_mae - neural_mae)
    / baseline_mae
) * 100

rmse_improvement = (
    (baseline_rmse - neural_rmse)
    / baseline_rmse
) * 100


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("BASELINE VS NEURAL NETWORK")
print("=" * 70)

print(
    f"\nAverage-rating baseline prediction: "
    f"{baseline_prediction:.4f}"
)

print("\nPerformance Results")
print("-" * 70)

print(
    f"Baseline MAE : {baseline_mae:.4f}"
)

print(
    f"Baseline RMSE: {baseline_rmse:.4f}"
)

print(
    f"Neural Network MAE : {neural_mae:.4f}"
)

print(
    f"Neural Network RMSE: {neural_rmse:.4f}"
)

print("\nImprovement")
print("-" * 70)

print(
    f"MAE improvement : {mae_improvement:.2f}%"
)

print(
    f"RMSE improvement: {rmse_improvement:.2f}%"
)

print("\n")
print("=" * 70)
print("Evaluation completed successfully.")
print("=" * 70)


# ============================================================
# 11. SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "Model": [
        "Average Rating Baseline",
        "Neural Network"
    ],

    "MAE": [
        baseline_mae,
        neural_mae
    ],

    "RMSE": [
        baseline_rmse,
        neural_rmse
    ]
})

results.to_csv(
    "data/neural_network_evaluation.csv",
    index=False
)

print(
    "\nResults saved to:"
)

print(
    "data/neural_network_evaluation.csv"
)