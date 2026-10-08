"""Bookwise: a local book discovery interface."""
import csv
import html
import io
import logging
from pathlib import Path

import joblib
import streamlit as st

from src.recommender import CatalogError, ROOT, display_text, load_catalog, work_key

st.set_page_config(page_title="Bookwise · Find your next read", page_icon="📖", layout="wide", initial_sidebar_state="collapsed")
st.html(ROOT / 'assets/styles.css')


@st.cache_resource(show_spinner="Opening the library…")
def catalog():
    return load_catalog()


@st.cache_resource(show_spinner="Loading reader recommendations…")
def reader_model():
    from src.model_loader import load_recommender
    paths = [ROOT / "models" / name for name in ("neural_recommender.keras", "user_to_index.joblib", "book_to_index.joblib")]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise CatalogError("Missing reader model files: " + ", ".join(missing))
    model = load_recommender(paths[0])
    users = {int(k): int(v) for k, v in joblib.load(paths[1]).items()}
    books = {str(k): int(v) for k, v in joblib.load(paths[2]).items()}
    for mapping, layer in ((users, "user_embedding"), (books, "book_embedding")):
        size = model.get_layer(layer).input_dim
        if len(set(mapping.values())) != len(mapping) or any(v < 0 or v >= size for v in mapping.values()):
            raise CatalogError("The reader mappings do not match the model. Retrain the neural model.")
    return model, users, books


@st.cache_data(show_spinner=False, max_entries=32)
def recommendations_for_reader(user_id, count):
    model, users, books = reader_model()
    return catalog().for_reader(user_id, model, users, books, count)


def escape(value):
    return html.escape(display_text(value), quote=True)


def add_preference(isbn):
    row = catalog().book(isbn)
    if row is None:
        return
    selected_works = {catalog().book(i)["_work"] for i in st.session_state.preferences if catalog().book(i)}
    if row["_work"] in selected_works:
        return
    if isbn not in st.session_state.preferences and len(st.session_state.preferences) < 5:
        st.session_state.preferences[isbn] = 8
        st.session_state.discovery_results = None


def remove_preference(isbn):
    st.session_state.preferences.pop(isbn, None)
    st.session_state.pop(f"rating_{isbn}", None)
    st.session_state.discovery_results = None


def clear_preferences():
    for isbn in list(st.session_state.preferences):
        remove_preference(isbn)


def update_rating(isbn):
    st.session_state.preferences[isbn] = st.session_state[f"rating_{isbn}"]
    st.session_state.discovery_results = None


def set_search(query):
    st.session_state.book_search = query
    st.session_state.search_page = 0


def reset_search_page():
    st.session_state.search_page = 0


def save_book(row):
    if any(work_key(saved["Title"], saved["Author"]) == work_key(row["Title"], row["Author"])
           for saved in st.session_state.saved.values()):
        return
    st.session_state.saved[row["ISBN"]] = {k: row[k] for k in ("ISBN", "Title", "Author", "Publisher")}


def remove_saved(isbn):
    st.session_state.saved.pop(isbn, None)


def result_cards(rows, source):
    if rows is None:
        return
    if not rows:
        st.info("No suitable matches yet. Try a different book or author, or adjust your ratings.")
        return
    st.subheader("Your next reads")
    st.caption(f"{len(rows)} recommendations · {source}")
    for rank, row in enumerate(rows, 1):
        with st.container(key=f"result_{source}_{row['ISBN']}", border=True):
            detail, action = st.columns([5, 1], vertical_alignment="center")
            with detail:
                st.html(
                    f'<article class="result"><span class="rank" aria-hidden="true">{rank:02}</span>'
                    f'<div><h3 id="result-{escape(row["ISBN"])}">{escape(row["Title"])}</h3><p>{escape(row["Author"])}</p>'
                    f'<p class="metadata">{escape(row["Publisher"])} · ISBN {escape(row["ISBN"])}</p></div></article>',
                )
                if "Similarity" in row:
                    st.caption(f'{row["Similarity"]:.0%} metadata similarity · {row.get("Reason", "")}')
                else:
                    st.caption(f'Estimated rating {row["Predicted rating"]:.1f} / 10 · Based on this dataset reader’s past ratings.')
            with action:
                saved = any(work_key(book["Title"], book["Author"]) == work_key(row["Title"], row["Author"])
                            for book in st.session_state.saved.values())
                st.button("Saved" if saved else "Save book", key=f"save_{source}_{row['ISBN']}",
                          disabled=saved, on_click=save_book, args=(row,),
                          help=f'Save {row["Title"]} to your reading list', width="stretch")
    st.caption("Similarity measures shared metadata words; it is not a probability of enjoyment." if "Similarity" in rows[0]
               else "Estimated ratings are model predictions, not reviews or guarantees.")


for key, value in {"preferences": {}, "saved": {}, "discovery_results": None,
                   "reader_results": None, "reader_signature": None, "search_page": 0}.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.html('<a class="skip-link" href="#discover">Skip to book discovery</a>')
st.html('<header class="masthead"><span class="wordmark">bookwise<span aria-hidden="true">.</span></span>'
            '<span class="masthead-note">A little help finding a good book</span></header>')
st.html('<div class="hero"><p class="eyebrow">THE NEXT CHAPTER</p>'
            '<h1>Find a book you’ll<br class="desktop-break"> want to get lost in.</h1>'
            '<p class="hero-copy">Start with a book you love. Build a small shelf of favorites, '
            'and discover what to read next.</p></div>')

st.html('<div id="discover" tabindex="-1"></div>')
mode = st.radio("Explore the library", ["Discover", "Reader demo", "Reading list"], horizontal=True,
                label_visibility="collapsed", key="mode")

try:
    library = catalog()
except Exception as error:
    logging.exception("Catalog startup failed")
    st.error(f"The library could not open. {error}")
    st.info("Restore the processed CSV files and TF-IDF matrix, then refresh this page. See README.md for setup.")
    st.stop()

st.html(f'<p class="library-note">{len(library.books):,} books in the local catalog '
            '· No account needed · Your selections stay in this browser session</p>')

if mode == "Discover":
    search_column, shelf_column = st.columns([1.55, 1], gap="large")
    with search_column:
        st.header("1. Find your favorites")
        st.write("Search by title, author or ISBN. Add up to five books to your shelf.")
        query = st.text_input("Search the library", placeholder="Try a title, author or ISBN", key="book_search",
                              on_change=reset_search_page, max_chars=120)
        if not query.strip():
            st.caption("A few places to start")
            examples = st.columns(3)
            for column, sample in zip(examples, ["The Hobbit", "Jane Austen", "Toni Morrison"]):
                column.button(sample, key=f"example_{sample}", on_click=set_search, args=(sample,), width="stretch")
            st.html('<div class="empty-search"><span class="book-mark" aria-hidden="true">B</span>'
                        '<h3>Good books lead to more good books.</h3><p>Search for something you’ve read and enjoyed. '
                        'Your shelf will stay here as you explore.</p></div>')
        elif len(query.strip()) < 2:
            st.info("Enter at least two letters or digits to search.")
        else:
            results, total = library.search(query, limit=1000)
            if total == 0:
                st.info("No books found. Check the spelling or try an author’s name. Your shelf is unchanged.")
            else:
                page_size = 6
                page_count = (min(total, 1000) + page_size - 1) // page_size
                st.session_state.search_page = min(st.session_state.search_page, page_count - 1)
                start = st.session_state.search_page * page_size
                visible = results.iloc[start:start + page_size]
                st.caption(f"{total:,} matching titles · Showing {start + 1}–{start + len(visible)}" +
                           (" · Narrow your search to see matches beyond the first 1,000" if total > 1000 else ""))
                for _, row in visible.iterrows():
                    with st.container(border=True, key=f"search_{row.ISBN}"):
                        text, action = st.columns([4, 1], vertical_alignment="center")
                        with text:
                            st.html(f'<div class="search-book"><h3 id="search-{escape(row.ISBN)}">{escape(row.Title)}</h3>'
                                        f'<p>{escape(row.Author)}</p><p class="metadata">ISBN {escape(row.ISBN)}</p></div>')
                        with action:
                            selected_works = {library.book(i)["_work"] for i in st.session_state.preferences if library.book(i)}
                            added = row["_work"] in selected_works
                            full = len(st.session_state.preferences) >= 5
                            st.button("Added" if added else "Add book", key=f"add_{row.ISBN}", disabled=added or full,
                                      on_click=add_preference, args=(row.ISBN,), help=f"Add {row.Title} to your favorites", width="stretch")
                if page_count > 1:
                    previous, page_label, following = st.columns([1, 2, 1], vertical_alignment="center")
                    if previous.button("Previous", disabled=start == 0, key="previous_page", width="stretch"):
                        st.session_state.search_page -= 1
                        st.rerun()
                    page_label.caption(f"Page {st.session_state.search_page + 1} of {page_count}")
                    if following.button("Next", disabled=st.session_state.search_page + 1 == page_count, key="next_page", width="stretch"):
                        st.session_state.search_page += 1
                        st.rerun()
    with shelf_column:
        with st.container(border=True, key="preference_shelf"):
            st.header("2. Make it yours")
            st.caption(f"{len(st.session_state.preferences)} of 5 books added")
            if not st.session_state.preferences:
                st.html('<div class="shelf-empty"><h3>Your favorites belong here.</h3>'
                            '<p>Add a book from search, then tell us how much you liked it.</p></div>')
            else:
                st.caption("6–10: enjoyed it · 5: neutral · 1–4: prefer less like this")
                for isbn, rating in list(st.session_state.preferences.items()):
                    row = library.book(isbn)
                    if row is None:
                        st.warning("A book is no longer in the catalog.")
                        st.button("Remove unavailable book", key=f"remove_{isbn}", on_click=remove_preference, args=(isbn,))
                        continue
                    st.html(f'<h3 class="shelf-title" id="favorite-{escape(isbn)}">{escape(row["Title"])}</h3>'
                                f'<p class="metadata">{escape(row["Author"])}</p>')
                    st.slider(f'Your rating for {row["Title"]}', 1, 10, value=rating, key=f"rating_{isbn}",
                              on_change=update_rating, args=(isbn,))
                    st.button("Remove book", key=f"remove_{isbn}", on_click=remove_preference, args=(isbn,),
                              help=f'Remove {row["Title"]} from your favorites')
                st.button("Clear favorites", key="clear_preferences", on_click=clear_preferences)
            count = st.selectbox("How many recommendations?", options=[5, 10], format_func=lambda n: f"{n} books", key="discovery_count")
            if st.button("Find my next read", type="primary", key="generate_new", width="stretch"):
                try:
                    with st.spinner("Finding books with similar metadata…"):
                        st.session_state.discovery_results = library.for_preferences(list(st.session_state.preferences.items()), count)
                    st.session_state.discovery_signature = (tuple(st.session_state.preferences.items()), count)
                except ValueError as error:
                    st.session_state.discovery_results = None
                    st.error(str(error))
                except Exception:
                    logging.exception("Discovery failed")
                    st.error("Recommendations could not load. Try again; your favorites are still here.")
            st.caption("No sign up. No personal details. Pick at least one book you rate 6 or higher.")
    signature = (tuple(st.session_state.preferences.items()), st.session_state.discovery_count)
    if st.session_state.discovery_results is not None:
        if signature == st.session_state.get("discovery_signature"):
            result_cards(st.session_state.discovery_results, "Your favorites")
        else:
            st.info("Your preferences changed. Choose ‘Find my next read’ to update your recommendations.")

elif mode == "Reader demo":
    st.header("Explore a dataset reader")
    st.write("Preview recommendations for an anonymous reader in the training dataset. These IDs are demo profiles; choose Discover to use your own favorites.")
    try:
        model, users, book_indices = reader_model()
    except Exception as error:
        logging.exception("Reader model startup failed")
        st.error(f"The reader demo is unavailable. {error}")
        st.info("You can still use Discover and your reading list.")
        st.stop()
    controls, history_column = st.columns([1, 1.55], gap="large")
    with controls:
        with st.container(border=True):
            user_id = st.number_input("Dataset reader ID", min_value=1, value=8, step=1, key="reader_id")
            st.caption(f"{len(users):,} reader IDs in the model. Try 8, or another ID from your dataset.")
            count = st.selectbox("Number of recommendations", options=[5, 10], format_func=lambda n: f"{n} books", key="reader_count")
            valid = int(user_id) in users
            if not valid:
                st.warning("This ID is not in the trained dataset. Try 8 or use Discover.")
            if st.button("Find reader recommendations", type="primary", key="generate_existing", width="stretch"):
                try:
                    with st.spinner("Finding books for this reader…"):
                        st.session_state.reader_results = recommendations_for_reader(int(user_id), count)
                    st.session_state.reader_signature = (int(user_id), count)
                except ValueError as error:
                    st.session_state.reader_results = None
                    st.error(str(error))
                except Exception:
                    logging.exception("Reader recommendation failed")
                    st.error("The reader model could not generate recommendations. Try discovery instead.")
    with history_column:
        st.subheader("Their reading history")
        history = library.history(int(user_id))
        if history is not None:
            joined = history.merge(library.books[["ISBN", "Title", "Author"]], on="ISBN", how="inner")
            st.caption(f"{len(joined)} rated books with catalog metadata")
            if not joined.empty:
                st.dataframe(joined.sort_values("Book-Rating", ascending=False)[["Title", "Author", "Book-Rating"]].head(10),
                             hide_index=True, width="stretch", column_config={"Book-Rating": "Their rating"})
            else:
                st.info("The ratings for this reader have no matching book details.")
        else:
            st.info("Choose a valid dataset reader to see their history.")
    if st.session_state.reader_results is not None:
        if st.session_state.reader_signature == (int(user_id), count):
            result_cards(st.session_state.reader_results, f"Dataset reader {int(user_id)}")
        else:
            st.info("The reader or result count changed. Generate recommendations for the new selection.")

else:
    st.header("Your reading list")
    st.write("A place for the books you want to come back to. Download your list to keep it beyond this browser session.")
    if not st.session_state.saved:
        st.info("Your list is waiting for its first book. Find recommendations in Discover and choose Save book.")
        st.button("Explore books", on_click=lambda: st.session_state.update(mode="Discover"), key="go_discover")
    else:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["ISBN", "Title", "Author", "Publisher"])
        writer.writeheader()
        # Neutralize spreadsheet formulas in the optional CSV export.
        for row in st.session_state.saved.values():
            writer.writerow({k: "'" + str(v) if str(v).lstrip().startswith(("=", "+", "-", "@")) else v for k, v in row.items()})
        st.download_button("Download reading list", output.getvalue(), "bookwise-reading-list.csv", "text/csv", key="download_list")
        st.caption(f"{len(st.session_state.saved)} saved {'book' if len(st.session_state.saved) == 1 else 'books'} · Session only")
        for isbn, row in list(st.session_state.saved.items()):
            with st.container(border=True):
                text, action = st.columns([5, 1], vertical_alignment="center")
                text.html(f'<div class="search-book"><h3 id="saved-{escape(isbn)}">{escape(row["Title"])}</h3><p>{escape(row["Author"])}</p>'
                              f'<p class="metadata">ISBN {escape(isbn)}</p></div>')
                action.button("Remove", key=f"unsave_{isbn}", on_click=remove_saved, args=(isbn,), help=f'Remove {row["Title"]} from your reading list', width="stretch")

with st.expander("How the recommendations work"):
    st.write("Discovery compares words in titles, author names and publishers. This catalog does not contain plot descriptions or genre labels. Low ratings reduce matches to books you disliked. Different editions of the same title by the same author are grouped.")
    st.write("The reader demo uses a trained neural model to estimate ratings for books an anonymous dataset reader has not rated. The dataset is historical; it does not reflect current releases or availability.")
    st.caption("Book files may contain spelling or encoding errors. Saved books and preferences are stored only in this Streamlit session, with no login or permanent profile.")
st.html('<footer class="footer"><span>bookwise.</span><span>A good read starts with a little curiosity.</span></footer>')
