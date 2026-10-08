"""Data validation, catalog search and recommendation ranking, independent of UI."""

import html
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import issparse

ROOT = Path(__file__).resolve().parents[1]
BOOK_COLUMNS = ["ISBN", "Book-Title", "Book-Author", "Publisher"]
EDITION_LABEL = re.compile(
    r"\((?:paperback|hardcover|hardback|large print|mass market paperback|"
    r"deluxe edition|book club edition|illustrated edition)\)", flags=re.I,
)


class CatalogError(ValueError):
    """An actionable problem with the local catalog or model artifacts."""


def display_text(value):
    if pd.isna(value):
        return "Unknown"
    return re.sub(r"<[^>]*>", "", html.unescape(str(value))).strip() or "Unknown"


def normalize_text(value):
    value = unicodedata.normalize("NFKD", display_text(value).casefold())
    return " ".join("".join(
        c if c.isalnum() else " " for c in value
        if not unicodedata.combining(c)
    ).split())


def work_key(title, author):
    # Remove edition labels, but retain series numbers and meaningful subtitles.
    title = EDITION_LABEL.sub("", display_text(title))
    return canonical_work(normalize_text(title), normalize_text(author))


def canonical_work(title, author):
    # Match punctuation variants in initials and the catalog's known Hobbit aliases.
    author = re.sub(r"\b([a-z])\s+(?=[a-z]\b)", r"\1", author)
    if author in {"jrr tolkien", "john rr tolkien", "john ronald reuel tolkien", "tolkien"}:
        author = "jrr tolkien"
        if title in {"the hobbit", "hobbit", "the hobbit or there and back again", "lo hobbit the hobbit", "el hobbit", "der kleine hobbit",
                     "bilbo le hobbit lord of the rings french", "bilbo le hobbit", "der hobbit",
                     "der kleine hobbit sonderausgabe",
                     "the hobbit the enchanting prelude to the lord of the rings",
                     "the hobbit leatherette collector s edition",
                     "the hobbit young adult edition sis cover",
                     "the hobbit lord of the rings audio",
                     "the annotated hobbit the hobbit or there and back again"}:
            title = "the hobbit"
    return title, author


def is_supplement(title):
    # Specific phrases in the title avoid suppressing books because of publishers
    # or broad words such as "summary" and "analysis".
    return bool(re.search(
        r"\b(?:cliffs\s?notes|sparknotes|york notes|coles notes|barrons? book notes|"
        r"study guides?|student guides?|teacher(?:'s)? guides?|answer keys?|"
        r"test banks?|revision guides?|exam guides?|lesson plans?|workbooks?)\b",
        display_text(title), flags=re.I,
    ))


def read_processed(path):
    # preprocess.py writes UTF-8; ISBNs must remain strings, including leading 0s.
    return pd.read_csv(path, encoding="utf-8", dtype={"ISBN": str}, keep_default_na=False)


def read_catalog(root=ROOT):
    root = Path(root)
    files = [root / "data/processed/books_clean.csv", root / "data/processed/ratings_clean.csv"]
    missing = [str(p.relative_to(root)) for p in files if not p.is_file()]
    if missing:
        raise CatalogError("Missing files: " + ", ".join(missing))
    books, ratings = [read_processed(p) for p in files]
    for table, columns, name in [(books, BOOK_COLUMNS, "books"),
                                  (ratings, ["ISBN", "User-ID", "Book-Rating"], "ratings")]:
        absent = set(columns) - set(table.columns)
        if absent:
            raise CatalogError(f"The {name} file is missing columns: {', '.join(sorted(absent))}")
    if books.empty or books.ISBN.duplicated().any() or books.ISBN.str.strip().eq("").any():
        raise CatalogError("The book catalog must contain unique, nonempty ISBNs.")
    ratings["User-ID"] = pd.to_numeric(ratings["User-ID"], errors="coerce")
    ratings["Book-Rating"] = pd.to_numeric(ratings["Book-Rating"], errors="coerce")
    valid = ratings["User-ID"].notna() & (ratings["User-ID"] % 1 == 0) & ratings["Book-Rating"].between(1, 10)
    ratings = ratings.loc[valid].drop_duplicates(["User-ID", "ISBN"]).copy()
    ratings["User-ID"] = ratings["User-ID"].astype(int)
    return books.reset_index(drop=True), ratings


class Recommender:
    def __init__(self, books, ratings, matrix):
        self.books = books.reset_index(drop=True).copy()
        self.ratings = ratings.copy()
        if not issparse(matrix) or matrix.shape[1] == 0:
            raise CatalogError("The content model must contain a sparse matrix with text features.")
        if matrix.shape[0] != len(books):
            raise CatalogError("The similarity matrix does not match the catalog. Rebuild the content model.")
        self.matrix = matrix.tocsr()
        if not np.isfinite(self.matrix.data).all():
            raise CatalogError("The similarity matrix contains invalid values.")
        self.norms = np.sqrt(self.matrix.multiply(self.matrix).sum(axis=1)).A1
        self.positions = {isbn: i for i, isbn in enumerate(self.books.ISBN.astype(str))}
        self.books["Title"] = self.books["Book-Title"].map(display_text)
        self.books["Author"] = self.books["Book-Author"].map(display_text)
        self.books["_search_title"] = self.books.Title.map(normalize_text)
        authors = {author: normalize_text(author) for author in self.books.Author.unique()}
        self.books["_search_author"] = self.books.Author.map(authors)
        self.books["_work"] = [
            canonical_work(normalize_text(EDITION_LABEL.sub("", title)) if EDITION_LABEL.search(title) else title_key, author_key)
            for title, title_key, author_key in zip(self.books.Title, self.books["_search_title"], self.books["_search_author"])
        ]
        self.books["_eligible"] = self.books.Title.ne("Unknown") & ~self.books.Title.map(is_supplement)
        self.lookup = self.books.set_index("ISBN", drop=False)
        # Store row positions, rather than tens of thousands of tiny DataFrames.
        self.history_indices = self.ratings.groupby("User-ID", sort=False).indices
        self.search_text = self.books["_search_title"] + " " + self.books["_search_author"] + " " + self.books.ISBN

    def history(self, user_id):
        indices = self.history_indices.get(int(user_id))
        return None if indices is None else self.ratings.iloc[indices]

    def search(self, query, limit=24):
        query = normalize_text(query)
        if len(query.replace(" ", "")) < 2:
            return self.books.iloc[:0].copy(), 0
        # Literal search with token matching also handles punctuation and accents.
        mask = pd.Series(True, index=self.books.index)
        for word in query.split():
            mask &= self.search_text.str.contains(word, regex=False)
        found = self.books.loc[mask & self.books["_eligible"]].copy()
        found["_exact"] = found["_search_title"].eq(query)
        found = found.sort_values(["_exact", "Title", "Author", "ISBN"], ascending=[False, True, True, True])
        found = found.drop_duplicates("_work")
        return found.head(limit), len(found)

    def book(self, isbn):
        if str(isbn) not in self.lookup.index:
            return None
        return self.lookup.loc[str(isbn)].to_dict()

    def cosine(self, isbn):
        index = self.positions.get(str(isbn))
        if index is None or self.norms[index] == 0:
            return np.zeros(len(self.books))
        dot = (self.matrix @ self.matrix[index].T).toarray().ravel()
        denominator = self.norms * self.norms[index]
        return np.clip(np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator > 0), 0, 1)

    @staticmethod
    def validate_count(top_n):
        if not isinstance(top_n, int) or not 1 <= top_n <= 20:
            raise ValueError("Request between 1 and 20 recommendations.")

    def _rank(self, scores, excluded, top_n, score_name):
        self.validate_count(top_n)
        excluded_works = {self.books.iloc[self.positions[i]]["_work"] for i in excluded if i in self.positions}
        eligible = self.books["_eligible"] & ~self.books.ISBN.isin(excluded) & ~self.books["_work"].isin(excluded_works)
        result = self.books.loc[eligible].copy()
        result[score_name] = np.asarray(scores)[eligible.to_numpy()]
        result = result[np.isfinite(result[score_name])]
        result = result.sort_values([score_name, "ISBN"], ascending=[False, True]).drop_duplicates("_work")
        return result.head(top_n).to_dict("records")

    def for_preferences(self, preferences, top_n=5):
        self.validate_count(top_n)
        if not preferences or len(preferences) > 5:
            raise ValueError("Choose between 1 and 5 books.")
        try:
            unique = {str(isbn): float(rating) for isbn, rating in preferences}
        except (ValueError, TypeError) as error:
            raise ValueError("Select books and rate each one from 1 to 10.") from error
        selected_works = [self.book(isbn)["_work"] for isbn in unique if self.book(isbn)]
        if len(selected_works) != len(set(selected_works)):
            raise ValueError("Choose one edition of each book for your favorites.")
        for isbn, rating in unique.items():
            if str(isbn) not in self.positions:
                raise ValueError("A selected book is no longer in the catalog. Remove it and search again.")
            if not np.isfinite(rating) or not 1 <= rating <= 10:
                raise ValueError("Rate each book from 1 to 10.")
        positive = [(str(i), (r - 5) / 5) for i, r in unique.items() if r > 5]
        if not positive:
            raise ValueError("Rate at least one book 6 or higher so we know what you enjoy.")
        positive_weight = sum(w for _, w in positive)
        scores = np.zeros(len(self.books))
        strongest = np.zeros(len(self.books))
        reasons = np.full(len(self.books), "", dtype=object)
        for isbn, rating in unique.items():
            similarity = self.cosine(isbn)
            weight = (rating - 5) / 5
            scores += similarity * weight / positive_weight
            if weight > 0:
                improved = similarity * weight > strongest
                reasons[improved] = self.book(isbn)["Title"]
                strongest = np.maximum(strongest, similarity * weight)
        # No evidence of a match is an empty result, not an arbitrary ranking.
        scores[scores <= 1e-9] = np.nan
        result = self._rank(scores, set(map(str, unique)), top_n, "Similarity")
        for row in result:
            row["Reason"] = f"Shares title, author or publisher words with {reasons[self.positions[row['ISBN']]]}."
        return result

    def for_reader(self, user_id, model, users, book_indices, top_n=5):
        self.validate_count(top_n)
        if int(user_id) not in users:
            raise ValueError("That reader is not in the trained model. Try discovery instead.")
        history = self.history(int(user_id))
        excluded = set() if history is None else set(history.ISBN.astype(str))
        candidates = [i for i in self.positions if i in book_indices and i not in excluded]
        if not candidates:
            return []
        user_inputs = np.full((len(candidates), 1), users[int(user_id)], dtype=np.int32)
        book_inputs = np.array([book_indices[i] for i in candidates], dtype=np.int32).reshape(-1, 1)
        predicted = np.asarray(model.predict([user_inputs, book_inputs], batch_size=1024, verbose=0)).ravel()
        if len(predicted) != len(candidates) or not np.isfinite(predicted).all():
            raise CatalogError("The reader model returned invalid predictions.")
        scores = np.full(len(self.books), np.nan)
        for isbn, rating in zip(candidates, np.clip(predicted, 1, 10)):
            scores[self.positions[isbn]] = rating
        return self._rank(scores, excluded, top_n, "Predicted rating")


def load_catalog(root=ROOT):
    books, ratings = read_catalog(root)
    manifest = Path(root) / "models/content_manifest.json"
    if manifest.is_file():
        expected = json.loads(manifest.read_text())["catalog_sha256"]
        actual = hashlib.sha256((Path(root) / "data/processed/books_clean.csv").read_bytes()).hexdigest()
        if actual != expected:
            raise CatalogError("The catalog changed after content training. Rebuild the content model to align its rows.")
    path = Path(root) / "models/tfidf_matrix.joblib"
    if not path.is_file():
        raise CatalogError("Missing models/tfidf_matrix.joblib. Run the content training script.")
    return Recommender(books, ratings, joblib.load(path))
