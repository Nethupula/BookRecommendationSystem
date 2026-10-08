# Bookwise — Book Recommendation System

A local Streamlit app for finding your next read. Search by title, author or ISBN, add up to five favorites, rate them, and save recommendations to a downloadable reading list. A separate reader demo uses the supplied neural model and anonymous dataset IDs.

Discovery compares **title, author and publisher words** using the stored TF-IDF matrix. The catalog has no plot descriptions, genre labels or current availability information. Similarity is not a probability of enjoyment, and neural scores are estimated ratings rather than reviews.

## Run locally

The verified environment is Python 3.9.6 on macOS ARM64. The pinned packages support Python 3.9–3.12; other interpreters and platforms have not been tested in this QA run. Prefer Python 3.11 or 3.12 for a new environment; Apple's system Python can emit a nonfatal LibreSSL warning from urllib3.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Windows PowerShell activation: `.\.venv\Scripts\Activate.ps1`.

Open [http://127.0.0.1:8501](http://127.0.0.1:8501). The server binds to your local machine. The first catalog load prepares its search index; subsequent actions reuse it. Favorites, results and saved books last for the Streamlit session. Download your reading list to retain it after the session ends.

TensorFlow is pinned to 2.19.0 because 2.20.0 hung during import on this Mac. `src/model_loader.py` loads the supplied neural weights and replaces its Python-version-specific rating lambda with equivalent `Rescaling(9, 1)`. New training uses this portable layer directly. Discovery does not load TensorFlow or deserialize the old scikit-learn estimators.

## Required local files

Datasets and trained artifacts are excluded from Git. A fresh clone cannot run until these files are restored or created:

```text
data/processed/books_clean.csv
data/processed/ratings_clean.csv
models/tfidf_matrix.joblib
```

For the reader demo, also provide:

```text
models/neural_recommender.keras
models/user_to_index.joblib
models/book_to_index.joblib
```

Processed CSVs must be UTF-8 and ISBNs must be preserved as strings. Required book columns: `ISBN`, `Book-Title`, `Book-Author`, `Publisher`. Rating columns: `User-ID`, `ISBN`, `Book-Rating`. Every TF-IDF row must correspond to the same catalog row. New content training records a catalog fingerprint to detect subsequent mismatches.

If only the neural files are unavailable, discovery remains usable. Missing catalog files produce an actionable setup message instead of a traceback.

## QA

```bash
python -m pip install -r requirements-training.txt
python -m unittest discover -s tests -v
```

Tests cover ranking invariants, malformed data, Unicode search, edition exclusion, rating penalties, model weight preservation, Streamlit flows, missing-file recovery, import safety, and pipelines using temporary synthetic datasets. App integration and original-model tests skip when their local artifacts are absent; inspect the skipped count before treating a run as complete.

On a machine where Matplotlib cannot write its default cache, set `MPLCONFIGDIR` to a writable temporary directory before running tests.

See [the QA report](docs/QA_REPORT.md), [measured data checks](docs/qa-data.json), and [UI screenshots](docs/screenshots/).

## Offline training and evaluation

Install `requirements-training.txt` first. The commands below are explicit offline workflows and **replace trained artifacts when executed**. QA uses separate temporary directories; it does not retrain over the supplied production-sized files.

1. Supply `data/Books.csv`, `data/Ratings.csv` and `data/Users.csv`. Raw CSV loading tries UTF-8, then Latin-1. The source and its license must be verified separately.
2. Clean the data: `python -m src.preprocess`.
3. Train metadata similarity: `python -m src.content_based`.
4. Train the optional collaborative baseline: `python -m src.collaborative`.
5. Train the neural model: `python -m src.neural_network`.
6. Preview neural results: `python -m src.neural_recommend --user 8 --count 5`.
7. Compare rating errors: `python -m src.baseline_evaluation`.
8. Compare experimental rankings: `python -m src.evaluation`.
9. Preview the experimental hybrid: `python -m src.hybrid`.
10. Plot training history: `python -m src.plot_training`.

Importing these modules does not run their pipelines. Paths resolve from the project directory, regardless of the working directory. Collaborative matrices are built directly as sparse data. New neural training saves a ratings fingerprint and exact split indices; evaluation uses those when available. The supplied model lacks original split provenance, so reconstructing the seeded split cannot independently certify its held-out quality.

## Structure

- `app.py`: Streamlit interface and session behavior.
- `assets/styles.css`, `.streamlit/config.toml`: responsive styles and local runtime settings.
- `src/recommender.py`: catalog validation, literal Unicode search and shared ranking logic.
- `src/model_loader.py`: neural model compatibility.
- `src/pipeline_support.py`: offline paths, prerequisites and sparse matrix helpers.
- Other `src/` modules: explicit offline pipelines.
- `tests/`: unit, UI integration, model and temporary pipeline checks.
- `docs/`: QA findings, measurements and browser evidence.
