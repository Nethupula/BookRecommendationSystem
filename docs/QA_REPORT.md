# Project QA and UI/UX rework

Date: 2026-10-08 · Environment: macOS ARM64, Python 3.9.6, Streamlit 1.50.0, TensorFlow 2.19.0.

## Result and scope

The main application has been reworked in the existing Streamlit stack. Discovery is now the default task, favorites survive unrelated actions, results can be saved, and anonymous dataset readers are explicitly presented as a demo. Runtime recommendation logic is separated from presentation and tested independently.

QA covered every project Python module through source review and import/syntax checks; the supplied catalog and model artifacts; application state and recommendation flows; offline pipelines using temporary synthetic data; and browser layout/keyboard checks. This is an evidence-based audit, not a guarantee that every possible bug has been found. Original-scale model retraining, independent ranking-quality validation, multiple browser engines and a screen-reader certification were not performed.

The supplied datasets and trained models were not replaced. Tests that train or write pipeline outputs use temporary directories. Artifact fingerprints are recorded in `qa-data.json`.

## Confirmed application and setup findings

| ID | Severity | Original problem and evidence | Resolution |
|---|---|---|---|
| A01 | High | Changing `book_search` changed the multiselect's options and cleared the selection. Reproduced: one selected ISBN became `[]` after changing from Harry Potter to Tolkien. | Fixed: favorites live in an independent session dictionary; search results use explicit Add actions. Regression tested across search, empty queries and navigation. |
| A02 | High | Recommendations existed only inside the button's `if` block. Changing an unrelated name input changed five displayed metrics to zero. | Fixed: results are retained in session state; signatures prevent showing them as current after preferences change. |
| A03 | High | A rating of 1 still had a positive recommendation weight and returned five recommendations. | Fixed: ratings below 5 reduce related matches; at least one rating above 5 is required. Neutral ratings add no positive evidence. |
| A04 | Medium | A second ISBN for the selected work was recommended; reproduced with The Hobbit. Reader exclusions also considered only ISBNs. | Fixed: exclude the seed/history work across recognized editions. Also prevent adding/saving another recognized edition. |
| A05 | Medium | Deduplication used only normalized titles. Different books by different authors could collapse into one. | Fixed: work identity includes normalized author. Synthetic tests retain same-title works by different authors. |
| A06 | Medium | Normalization erased all non-ASCII letters and entire parenthesized subtitles/series labels. `東京物語` became empty. | Fixed: Unicode normalization retains letters, accents are searchable, and only specific edition labels are removed. Series numbers are preserved. |
| A07 | Medium | Neighbor code blindly discarded the first neighbor. With ties, the seed is not guaranteed to be first. | Fixed in the app: direct sparse cosine scores and explicit ISBN/work exclusion; no positional assumption. |
| A08 | Medium | Similarity scores were unnormalized sums and grew with the number of favorites; labels implied a comparable content score. | Fixed: weighted scores are bounded for positive results, penalties are explicit, and the UI explains that this is metadata similarity. |
| A09 | Medium | Broad filtering words such as “summary” matched legitimate titles and publisher/author text. | Fixed: specific guide phrases in the title only; plural guide names and Coles/Barrons notes are recognized. |
| A10 | High | A missing file or malformed artifact stopped startup with a traceback. Both model types loaded unconditionally. | Fixed: actionable catalog errors and lazy neural loading. Missing neural artifacts do not disable discovery. Missing-catalog behavior is tested. |
| A11 | High | The original dependency file mixed unrelated web-server packages and versions incompatible with the local interpreter; dependency installation could not reproduce the supplied environment. | Fixed: short verified runtime pins plus separate offline-training dependencies. Installation and `pip check` pass locally. Other platforms still need independent verification. |
| A12 | High | TensorFlow 2.20 hung on import on this Mac. The supplied Keras model contains newer configuration fields and Python-version-specific lambda bytecode. | Mitigated: TensorFlow 2.19 and the compatibility loader. All eight learned weight tensors match the saved artifact exactly; the rating scale is verified at 0, 0.5 and 1. New training uses Rescaling directly. |
| A13 | Medium | The web app deserialized scikit-learn estimators saved with 1.9.1 under installed 1.6.1, emitting incompatibility warnings. | Fixed for the main app: discovery uses the sparse TF-IDF matrix directly. Experimental scripts using the original estimators should rebuild them in the documented environment. |
| A14 | Medium | Processed CSV files are UTF-8 but app/CLI readers used Latin-1, adding further corruption. ISBN type was left to inference. | Fixed: processed readers use UTF-8 and string ISBNs. Raw preprocessing tries UTF-8 then Latin-1. Existing source corruption remains a data issue. |
| A15 | Medium | The matrix/catalog row relationship was assumed, with no readiness validation or future fingerprint. | Fixed: shape and value validation, a fingerprint for new content training, and a complete audit of the provided matrix against all catalog rows. |
| A16 | Medium | Invalid counts, missing IDs, empty/invalid preference sets, nonfinite scores and zero vectors were not consistently guarded. | Fixed: explicit validation and empty-result handling; boundary and malformed-input tests pass. |
| A17 | Medium | Each rerun rebuilt catalog data and performed repeated text/row processing. The first version of the rework took 15.574 seconds to initialize under test load. | Improved: cached resources, shared search text, reused normalized authors, compact history row indices and bounded reader-result caching. Final measured cold load: 3.516 seconds. |
| A18 | Medium | Dataset text was interpolated into raw HTML without escaping; encoded formatting tags also leaked into titles. | Fixed: display text is decoded/stripped, dynamic HTML text is escaped, and presentation uses Streamlit's sanitized `st.html`. CSV export also neutralizes spreadsheet formula prefixes. This does not certify arbitrary untrusted pickle/model files as safe. |

## UI and accessibility findings

| ID | Severity | Original problem | Rework |
|---|---|---|---|
| U01 | High | The default task required choosing one of 77,805 anonymous user IDs. Most visitors have no meaningful ID to choose. | Discovery comes first; the dataset-reader flow is a clearly described separate demo with a numeric ID input and validation. |
| U02 | Medium | Long technical sidebar, dataset statistics and architecture descriptions preceded the book task. | Removed the sidebar/technical dashboard from the primary task. Model information is in an optional disclosure. |
| U03 | Medium | A name field added a step but did not affect recommendations. | Removed. No personal details are requested to discover books. |
| U04 | Medium | Emoji-heavy copy, oversized bold headings, repeated explanations and stacked score metrics made results hard to scan. | A consistent neutral theme, Manrope typography, concise copy and compact editorial book rows. Scores include their meaning and uncertainty. |
| U05 | Medium | There was no way to retain a recommendation in a useful reading list. | Session reading list with save/remove actions and CSV download. Session-only persistence is explicitly explained. |
| U06 | Medium | Section titles were styled divs rather than headings; page-level heading/skip navigation were missing. | One H1, section headings, native labeled controls, a focusable skip target and visible 2 px focus outlines. Custom HTML IDs are unique. |
| U07 | Medium | Search returned an unbounded multiselect, and empty/search/error states did little to explain the next action. | Literal title/author/ISBN search, six rows per page, explicit match counts, actionable empty states and a persistent five-book shelf. Broad queries are capped at 1,000 displayed matches with an explanation. |
| U08 | Medium | Desktop-first composition and small controls offered no reviewed mobile behavior. | Tested 390 px and 320 px reflow without horizontal overflow; input text is 16 px, actions use 44 px minimum height, and the two-column layout stacks. |
| U09 | Medium | Results implied AI/content understanding beyond the available data. README claimed descriptions, filters and dynamic previews that were not implemented. | UI and README accurately describe title/author/publisher word similarity, estimated ratings, a historical catalog and no genre/availability data. |
| U10 | Low | Count sliders for only two choices added a mechanical control with little benefit. | Simple labeled dropdowns for 5 or 10 recommendations; rating sliders remain where an ordered scale helps. |

## Offline-pipeline findings

| ID | Severity | Problem | Resolution / limitation |
|---|---|---|---|
| P01 | High | Importing research modules immediately trained models, evaluated data, wrote files or opened plots. | Fixed: explicit `main()` entry points. Import safety is tested with artifact reads/writes and CSV reads forbidden during import. |
| P02 | Medium | Paths depended on the current working directory. Missing prerequisites produced low-level file failures. | Fixed: project-relative paths and prerequisite messages. The neural CLI was run successfully from `/private/tmp`. |
| P03 | High | Rating evaluation reconstructed user/book encodings from CSV order rather than the saved embedding mappings. Reordered data could evaluate the wrong identities. | Fixed: load persisted mappings; refuse unknown IDs. New neural training also records the exact split and an input fingerprint. |
| P04 | High | Collaborative and hybrid pipelines materialized dense pivots before converting to sparse matrices. | Fixed: build CSR directly and retain sparse DataFrame metadata. Duplicate interactions are averaged as in pivot_table; only positive stored ratings count as known. Synthetic tests verify both. |
| P05 | Medium | Neighbor counts exceeded available rows for small fixtures. | Fixed: bound neighbor counts by the appropriate catalog/user matrix. Content/collaborative/hybrid/evaluation pass on 12-book/12-user fixtures. |
| P06 | Medium | Experimental hybrid content candidates could include books already rated by the target user. | Fixed: exclude known ISBNs from the combined candidate set. |
| P07 | Medium | Experimental evaluation selected the first 100 eligible IDs and tail rows as holdouts, creating order-dependent samples; no eligible users caused concat errors. | Improved: seeded random user/holdout samples and an explicit no-eligible-data message. This remains an active-user evaluation, not population-wide evidence. |
| P08 | Medium | Raw preprocessing assumed all optional image columns existed, did not coerce malformed rating inputs, and kept repeated user/book records. | Fixed: optional columns, numeric/coherent user IDs and rating filtering, and one record per user/book. Synthetic Unicode/invalid-row fixtures pass. |
| P09 | Medium | Neural CLI used a separate incompatible loader, inconsistent keys and different duplicate/filter logic; it printed the author twice. | Fixed: shared catalog/ranking/compatibility code and typed mappings; CLI returns five results for reader 8. |
| P10 | Medium | Original neural training stored Python lambda bytecode and did not save a reproducible split or input provenance. | Improved for future runs: portable Rescaling, random seed, exact split indices and ratings fingerprint. No original-scale retraining was done. |

| P11 | Medium | Equal-distance neighbors could cause the first result to be skipped instead of the actual target user. | Fixed: exclude self by index, preserve tied neighbors and cap counts; regression test passes. |

## Remaining data and model limitations

1. **Unmatched metadata — high impact:** 49,829 of 433,671 rating interactions reference ISBNs absent from the catalog (about 11.5%). The UI only displays/recommends books with metadata. Repair this by reconciling the source catalog and retraining compatible artifacts; inventing titles or dropping rows silently would obscure the problem.
2. **Suspect encoding — medium impact:** 6,634 titles contain `Ã`, `Â` or replacement characters when read correctly as UTF-8. This is a diagnostic count of suspect rows, not an infallible encoding detector. Some strings already contain lost characters. Restore a clean source and rebuild content artifacts rather than applying guessed substitutions.
3. **Original model provenance — high impact:** the supplied neural model has no original split file or input fingerprint. The current CSV and the repository's seeded split recipe produce 43,368 test rows, including 4,933 rows whose users are absent from training and 14,178 whose books are absent from training. These are counts for the reconstructed recipe, not proof of the supplied model's actual training split. Report warm/cold subsets separately and use a preserved split before claiming independent model quality.
4. **Representation and ranking quality — medium impact:** content features have no plot/genre information, use English stop words in the training script, and rely on title/author/publisher vocabulary. Neural predictions have not been calibrated against a fresh independent evaluation in this task. Algorithmic correctness and a polished UI do not establish recommendation usefulness or fairness.
5. **Heuristic work identity — medium impact:** spelling variants, translations and author aliases cannot reliably be unified without authoritative work IDs. Edition filtering and scoped Tolkien/Hobbit aliases reduce straightforward repeats but do not provide bibliographic identity resolution. Browser QA still found differently worded Jane Austen complete-novel collections ranked together; authoritative work/edition metadata is needed to resolve such cases without conflating distinct collections.
6. **Missing offline inputs — verification gap:** raw Books/Ratings/Users CSVs, the original collaborative artifact, content-books snapshot and neural history are not supplied. Synthetic workflows pass, but an original-scale raw-data preprocessing, collaborative training/evaluation and original-history plotting run cannot be certified. An empty notebook placeholder provides no experiment provenance.
7. **Persistence and deployment — product scope:** saved books are session-only and downloadable; there is no account/database sync. Only local hosting is configured. This work does not establish multi-user deployment/load behavior.
8. **Environment/accessibility breadth — verification gap:** browser checks use the Codex in-app browser with viewport emulation, not physical devices. No Safari/Firefox run, VoiceOver/NVDA session or complete WCAG conformance audit was performed. The optional Google-hosted font needs network access; fallback typography remains available offline. System Python emits nonfatal LibreSSL/TensorFlow diagnostics, and Matplotlib dependencies emit deprecation warnings.

## Verification evidence

- **Automated tests:** **26 tests passed in 18.944 seconds**, recorded in `qa-tests.txt`. Tests cover app flows, invalid/stale inputs, a five-book limit, paging, missing-file recovery, ranking/data boundaries, model tensors, import safety, sparse construction and temporary offline workflows.
- **Original neural weights:** all eight learned tensors match the archived HDF5 arrays exactly after compatibility loading. Predictions are finite and in the 1–10 range. The rating-scale transform returns 1, 5.5 and 10 for inputs 0, 0.5 and 1.
- **Full matrix/catalog alignment:** all 271,360 rows were transformed using the provided vectorizer and compared with the supplied matrix. Zero mismatched rows at a 1e-8 row-error threshold; maximum row error 1.5681900222830336e-15.
- **Dataset integrity:** zero duplicate catalog ISBNs and zero duplicate user/book pairs in the supplied processed files; stored ratings range from 1 to 10. Unmatched/suspect counts above remain open.
- **Final measured performance:** cold catalog construction 3.516 s; sampled search queries 0.0001–0.0969 s; ten content recommendations 0.175 s. These are single-run local measurements, not production benchmarks. Earlier load measurements were under different concurrent test load, so the apparent speedup is directional rather than a controlled benchmark.
- **Dependency/setup checks:** `pip install -r requirements-training.txt`, `pip check`, Python AST parsing and `git diff --check` pass. The neural CLI works from outside the project directory.
- **Browser:** desktop, 390 px and 320 px layouts; visible labeled controls and heading outline; keyboard focus verified as a white 2 px solid outline; search/add/generate/save paths checked in the running interface. Screenshots are in `screenshots/`.
- **Contrast:** white on black is 21:1; `#9B9B9B` on black is approximately 7.56:1 and on `#181818` approximately 6.39:1. These cover the main custom text colors; they are not a claim that every framework-generated state was exhaustively measured.

## Suggested next work

Prioritize a clean, licensed source catalog and reproducible model evaluation over adding more decoration. Then decide whether permanent reading-list storage is needed. After those decisions, cross-browser and assistive-technology testing can close the remaining release verification gaps.
