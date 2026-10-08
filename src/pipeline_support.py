"""Shared paths and prerequisite checks for the offline research scripts."""
from pathlib import Path
import pandas as pd
import joblib
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]


def require_files(paths):
    missing = [str(Path(p).relative_to(ROOT)) if Path(p).is_relative_to(ROOT) else str(p)
               for p in paths if not Path(p).is_file()]
    if missing:
        raise SystemExit('Missing prerequisites: ' + ', '.join(missing) + '. See README.md for the training order.')


def load_content_books(path):
    path=Path(path)
    if path.is_file():
        return joblib.load(path)
    fallback=ROOT/'data/processed/books_clean.csv'
    require_files([fallback])
    return pd.read_csv(fallback, encoding='utf-8', dtype={'ISBN':str}, keep_default_na=False)


def read_raw(path):
    try:
        return pd.read_csv(path, encoding='utf-8-sig', dtype={'ISBN': str}, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='latin-1', dtype={'ISBN': str}, low_memory=False)


def sparse_user_books(ratings):
    if ratings.empty:
        raise SystemExit('No ratings remain after activity filters. Use more data or lower the thresholds.')
    grouped = ratings.groupby(['User-ID', 'ISBN'], as_index=False)['Book-Rating'].mean()
    users = pd.Index(sorted(grouped['User-ID'].unique()))
    books = pd.Index(sorted(grouped['ISBN'].unique()))
    matrix = csr_matrix((grouped['Book-Rating'].to_numpy(),
                         (users.get_indexer(grouped['User-ID']), books.get_indexer(grouped['ISBN']))),
                        shape=(len(users), len(books)))
    return pd.DataFrame.sparse.from_spmatrix(matrix, index=users, columns=books), matrix


def known_ratings(frame, user_id):
    row = frame.loc[user_id]
    return row[row > 0]


def similar_neighbors(model, matrix, user_index, count=5):
    distances, indices = model.kneighbors(matrix[user_index], n_neighbors=min(count + 1, matrix.shape[0]))
    # Equal vectors can put another reader first, or omit self from the tie set.
    return [(int(i), float(1 - distance)) for i, distance in zip(indices[0], distances[0])
            if i != user_index][:count]
