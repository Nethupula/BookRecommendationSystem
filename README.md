# 📚 AI Book Recommendation System

An intelligent book recommendation web application built with **Streamlit**, featuring **Content-Based Filtering (TF-IDF)** and **Deep Neural Network Collaborative Filtering**.

---

## 🚀 Features

- **Personalized Recommendations**: Content-based filtering using book descriptions and metadata.
- **Deep Learning Recommender**: Neural network collaborative filtering trained on user interaction patterns.
- **Interactive UI**: Modern Streamlit interface with search, filters, and dynamic book previews.

---

## 🛠️ Getting Started

### 1. Clone the Repository
```bash
git clone <repository-url>
cd BookRecommendationSystem
```

### 2. Set Up a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
BookRecommendationSystem/
├── app.py                # Main Streamlit web application
├── requirements.txt      # Python dependencies
├── data/
│   └── processed/        # Cleaned dataset catalogs (books_clean.csv, ratings_clean.csv)
├── models/               # Pre-trained models (TF-IDF vectorizer, Neural recommender, mappings)
├── src/                  # Model training, preprocessing & evaluation pipelines
└── notebooks/            # Jupyter notebooks for experiments
```
