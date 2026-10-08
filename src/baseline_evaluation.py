"""Compare a supplied neural model with the mean-rating baseline.

Without the saved original split, the seeded split below is a reproducibility
check of the repository's recipe, not independently verified held-out evidence.
"""
import joblib
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from .recommender import ROOT, read_processed
    from .model_loader import load_recommender
    from .pipeline_support import require_files
except ImportError:
    from recommender import ROOT, read_processed
    from model_loader import load_recommender
    from pipeline_support import require_files


def main():
    paths=[ROOT/'data/processed/ratings_clean.csv',ROOT/'models/neural_recommender.keras',
           ROOT/'models/user_to_index.joblib',ROOT/'models/book_to_index.joblib']
    require_files(paths)
    ratings=read_processed(paths[0])
    ratings['Book-Rating']=pd.to_numeric(ratings['Book-Rating'],errors='coerce')
    ratings=ratings[ratings['Book-Rating'].between(1,10)].copy()
    users={str(k):int(v) for k,v in joblib.load(paths[2]).items()}
    books={str(k):int(v) for k,v in joblib.load(paths[3]).items()}
    # Saved mappings define embedding identity. CSV row order must not redefine it.
    ratings['user_index']=ratings['User-ID'].astype(str).map(users)
    ratings['book_index']=ratings.ISBN.map(books)
    if ratings[['user_index','book_index']].isna().any().any():
        raise SystemExit('The dataset contains IDs absent from the trained mappings; evaluate against the original training data.')
    x=ratings[['user_index','book_index']].to_numpy(dtype=np.int32)
    y=ratings['Book-Rating'].to_numpy(dtype=np.float32)
    if len(y)<10:
        raise SystemExit('At least 10 valid ratings are needed for this split.')
    manifest=ROOT/'models/neural_manifest.json'
    split=ROOT/'models/neural_split.npz'
    verified_split=manifest.is_file() and split.is_file()
    if verified_split:
        expected=json.loads(manifest.read_text())['ratings_sha256']
        if hashlib.sha256(paths[0].read_bytes()).hexdigest()!=expected:
            raise SystemExit('The ratings file changed after training; restore the original dataset or retrain.')
        with np.load(split) as indices:
            y_train=y[indices['train']]
            x_test=x[indices['test']]
            y_test=y[indices['test']]
    else:
        x_train,x_temp,y_train,y_temp=train_test_split(x,y,test_size=.2,random_state=42)
        _,x_test,_,y_test=train_test_split(x_temp,y_temp,test_size=.5,random_state=42)
    model=load_recommender(paths[1])
    predictions=np.asarray(model.predict([x_test[:,0:1],x_test[:,1:2]],batch_size=1024,verbose=0)).ravel()
    if not np.isfinite(predictions).all():
        raise SystemExit('Model predictions contain nonfinite values.')
    rows=[]
    for name,pred in [('Average rating baseline',np.full(len(y_test),y_train.mean())),('Neural model',predictions)]:
        rows.append({'Model':name,'MAE':mean_absolute_error(y_test,pred),'RMSE':np.sqrt(mean_squared_error(y_test,pred))})
    result=pd.DataFrame(rows)
    if not verified_split:
        print('CAUTION: the original training split and artifact provenance are unavailable; this seeded reconstruction cannot certify held-out quality.')
    print(result.to_string(index=False))
    result.to_csv(ROOT/'data/neural_network_evaluation.csv',index=False)


if __name__=='__main__':
    main()
