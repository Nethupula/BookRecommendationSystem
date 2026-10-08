"""Exercise offline pipelines only on temporary synthetic data."""
import contextlib
import importlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

MODULES=['preprocess','content_based','collaborative','hybrid','evaluation','neural_network','baseline_evaluation','neural_recommend','plot_training']


class PipelineTests(unittest.TestCase):
    def test_sparse_matrix_preserves_only_known_ratings(self):
        from src.pipeline_support import sparse_user_books, known_ratings
        ratings=pd.DataFrame({'User-ID':[8,9,8], 'ISBN':['001','002','001'], 'Book-Rating':[8,7,10]})
        frame,matrix=sparse_user_books(ratings)
        self.assertEqual(matrix.shape,(2,2))
        self.assertEqual(matrix.nnz,2)
        self.assertEqual(known_ratings(frame,8).to_dict(),{'001':9.0})
        with self.assertRaisesRegex(SystemExit,'No ratings'):
            sparse_user_books(ratings.iloc[:0])

    def test_neighbor_ties_exclude_actual_self(self):
        import numpy as np
        from scipy.sparse import csr_matrix
        from src.pipeline_support import similar_neighbors
        class TiedModel:
            def kneighbors(self, vector, **kwargs):
                return np.array([[0.,0.,0.]]),np.array([[2,0,1]])
        self.assertEqual(similar_neighbors(TiedModel(),csr_matrix(np.ones((3,2))),0,2),[(2,1.),(1,1.)])
    def test_imports_do_not_train_load_artifacts_or_write_data(self):
        for name in MODULES:
            with self.subTest(module=name), patch('joblib.load', side_effect=AssertionError('Import loaded an artifact')), patch('joblib.dump', side_effect=AssertionError('Import wrote an artifact')), patch.object(pd, 'read_csv', side_effect=AssertionError('Import read data')):
                module=importlib.import_module('src.'+name)
                importlib.reload(module)
                self.assertTrue(callable(module.main))

    def test_synthetic_preprocess_content_collaborative_and_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'data').mkdir(); (root/'models').mkdir()
            books=pd.DataFrame({'ISBN':[f'{i:010}' for i in range(12)],
                                'Book-Title':[f'River tale {i}' for i in range(12)],
                                'Book-Author':['García' if i%2 else 'Tolkien' for i in range(12)],
                                'Publisher':['Library']*12, 'Year-Of-Publication':[2000]*12})
            ratings=pd.DataFrame([{'ISBN':f'{book:010}', 'User-ID':user, 'Book-Rating':8+(user+book)%3} for user in range(12) for book in range(12)])
            ratings=pd.concat([ratings,pd.DataFrame([{'ISBN':'0000000000','User-ID':'invalid','Book-Rating':'invalid'}])],ignore_index=True)
            books.to_csv(root/'data/Books.csv',index=False); ratings.to_csv(root/'data/Ratings.csv',index=False)
            pd.DataFrame({'User-ID':list(range(12)),'Location':['Test']*12,'Age':[30]*12}).to_csv(root/'data/Users.csv',index=False)
            for name in ['preprocess','content_based','collaborative','hybrid','evaluation']:
                module=importlib.import_module('src.'+name)
                with self.subTest(pipeline=name), patch.object(module,'ROOT',root), contextlib.redirect_stdout(io.StringIO()):
                    module.main()
            clean=pd.read_csv(root/'data/processed/books_clean.csv',dtype={'ISBN':str})
            self.assertEqual(clean.iloc[0].ISBN,'0000000000')
            self.assertIn('García',clean['Book-Author'].tolist())
            self.assertEqual(len(pd.read_csv(root/'data/processed/ratings_clean.csv')),144)
            result=pd.read_csv(root/'data/evaluation_results.csv')
            self.assertEqual(len(result),3)
            self.assertTrue(result[['Precision@5','Recall@5']].ge(0).all().all())
            self.assertTrue(result[['Precision@5','Recall@5']].le(1).all().all())

    def test_missing_prerequisites_have_actionable_errors(self):
        module=importlib.import_module('src.preprocess')
        with tempfile.TemporaryDirectory() as directory, patch.object(module,'ROOT',Path(directory)):
            with self.assertRaisesRegex(SystemExit,'Missing prerequisites'):
                module.main()

class NeuralPipelineTests(unittest.TestCase):
    def test_temporary_neural_training_evaluation_and_plotting(self):
        import numpy as np
        from src.model_loader import load_recommender
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'data/processed').mkdir(parents=True); (root/'models').mkdir()
            ratings=pd.DataFrame([{'ISBN':f'{book:010}', 'User-ID':user, 'Book-Rating':6+(user+book)%5} for user in range(4) for book in range(6)])
            ratings.to_csv(root/'data/processed/ratings_clean.csv',index=False)
            for name in ['neural_network','baseline_evaluation','plot_training']:
                module=importlib.import_module('src.'+name)
                with self.subTest(pipeline=name), patch.object(module,'ROOT',root), contextlib.redirect_stdout(io.StringIO()), patch('matplotlib.pyplot.show'):
                    module.main()
            model=load_recommender(root/'models/neural_recommender.keras')
            self.assertEqual(model.get_layer('rating_scale').__class__.__name__,'Rescaling')
            self.assertTrue((root/'models/training_validation_loss.png').exists())
            result=pd.read_csv(root/'data/neural_network_evaluation.csv')
            self.assertEqual(len(result),2)
            self.assertTrue(np.isfinite(result[['MAE','RMSE']].to_numpy()).all())


if __name__=='__main__':
    unittest.main()
