"""Preview the same reader recommendations as the web app from the terminal."""
import argparse
import joblib

try:
    from .recommender import ROOT, load_catalog, CatalogError
    from .model_loader import load_recommender
    from .pipeline_support import require_files
except ImportError:
    from recommender import ROOT, load_catalog, CatalogError
    from model_loader import load_recommender
    from pipeline_support import require_files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--user', type=int, default=8)
    parser.add_argument('--count', type=int, default=5)
    args=parser.parse_args()
    require_files([ROOT/'models/neural_recommender.keras', ROOT/'models/user_to_index.joblib', ROOT/'models/book_to_index.joblib'])
    try:
        library=load_catalog()
        model=load_recommender(ROOT/'models/neural_recommender.keras')
        users={int(k):int(v) for k,v in joblib.load(ROOT/'models/user_to_index.joblib').items()}
        books={str(k):int(v) for k,v in joblib.load(ROOT/'models/book_to_index.joblib').items()}
        rows=library.for_reader(args.user, model, users, books, args.count)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    for rank,row in enumerate(rows,1):
        print(f"{rank}. {row['Title']} — {row['Author']} ({row['Predicted rating']:.2f}/10)")
    print(f'{len(rows)} recommendations for dataset reader {args.user}.')


if __name__=='__main__':
    main()
