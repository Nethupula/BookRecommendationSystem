import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.recommender import CatalogError, Recommender, display_text, normalize_text, read_catalog, load_catalog, work_key, is_supplement


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.books = pd.DataFrame({
            'ISBN': ['01', '02', '03', '04', '05', '06', '07'],
            'Book-Title': ['The River', 'The River (Paperback)', 'The River', 'Woodland', 'Space', 'River Study Guide', 'Empty'],
            'Book-Author': ['A', 'A', 'B', 'A', 'C', 'Guide author', 'D'],
            'Publisher': ['Pub'] * 7,
        })
        self.ratings = pd.DataFrame({'ISBN': ['01'], 'User-ID': [8], 'Book-Rating': [8]})
        self.matrix = csr_matrix([[1,0,0], [1,0,0], [.9,.1,0], [.6,.8,0], [0,1,0], [1,0,0], [0,0,0]], dtype=float)
        self.engine = Recommender(self.books, self.ratings, self.matrix)

    def test_excludes_seed_editions_and_supplements_but_keeps_other_authors(self):
        rows = self.engine.for_preferences([('01', 8)])
        self.assertEqual(rows[0]['ISBN'], '03')
        self.assertFalse({'01','02','06','07'} & {r['ISBN'] for r in rows})
        self.assertEqual(len({(r['Title'], r['Author']) for r in rows}), len(rows))

    def test_low_ratings_are_not_positive_preferences(self):
        with self.assertRaisesRegex(ValueError, '6 or higher'):
            self.engine.for_preferences([('01', 1)])
        before = self.engine.for_preferences([('01', 10)], top_n=5)
        after = self.engine.for_preferences([('01', 10), ('05', 1)], top_n=5)
        old = next(r['Similarity'] for r in before if r['ISBN']=='04')
        new = [r['Similarity'] for r in after if r['ISBN']=='04']
        self.assertTrue(not new or new[0] < old)

    def test_scores_are_bounded_and_finite(self):
        rows = self.engine.for_preferences([('01', 10), ('04', 9)])
        self.assertTrue(all(0 < r['Similarity'] <= 1 for r in rows))
        self.assertEqual(rows, self.engine.for_preferences([('01', 10), ('04', 9)]))

    def test_zero_vectors_are_safe(self):
        self.assertTrue(np.equal(self.engine.cosine('07'), 0).all())
        self.assertEqual(self.engine.for_preferences([('07', 8)]), [])

    def test_invalid_preferences_and_counts(self):
        for prefs in ([], [('missing',8)], [('01',11)], [('01',float('nan'))], [('01',8),('02',8)], [('01','invalid')]):
            with self.subTest(prefs=prefs), self.assertRaises(ValueError):
                self.engine.for_preferences(prefs)
        for count in (0, -1, 21, '5'):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.engine.for_preferences([('01',8)], count)

    def test_search_is_literal_and_keeps_distinct_authors(self):
        result, count = self.engine.search('River')
        self.assertEqual(count, 2)
        self.assertEqual(set(result.Author), {'A','B'})
        self.assertEqual(self.engine.search('[')[1], 0)
        self.assertEqual(self.engine.search('01')[0].iloc[0].ISBN, '01')
        self.assertEqual(self.engine.search('zzzz')[1], 0)

    def test_reader_excludes_known_work_and_missing_metadata(self):
        class Predictor:
            def predict(self, inputs, **kwargs):
                self.inputs = inputs
                return np.full((len(inputs[0]), 1), 8.2)
        predictor = Predictor()
        rows = self.engine.for_reader(8, predictor, {8:0}, {i:n for n,i in enumerate(['01','02','03','04','05','06','absent'])})
        self.assertFalse({'01','02','06','absent'} & {r['ISBN'] for r in rows})
        self.assertEqual(predictor.inputs[0].shape[1], 1)
        self.assertTrue(all(1<=r['Predicted rating']<=10 for r in rows))

    def test_reader_rejects_invalid_outputs_and_users(self):
        class BadPredictor:
            def predict(self, inputs, **kwargs):
                return np.full((len(inputs[0]),1), np.nan)
        with self.assertRaises(CatalogError):
            self.engine.for_reader(8, BadPredictor(), {8:0}, {'03':0})
        with self.assertRaises(ValueError):
            self.engine.for_reader(99, BadPredictor(), {8:0}, {'03':0})

    def test_matrix_validation(self):
        with self.assertRaises(CatalogError):
            Recommender(self.books, self.ratings, self.matrix.toarray())
        with self.assertRaises(CatalogError):
            Recommender(self.books, self.ratings, self.matrix[:2])
        bad = self.matrix.copy(); bad.data[0] = np.nan
        with self.assertRaises(CatalogError):
            Recommender(self.books, self.ratings, bad)


class TextAndDataTests(unittest.TestCase):
    def test_manifest_detects_catalog_replacement(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); processed=root/'data/processed'; processed.mkdir(parents=True); (root/'models').mkdir()
            pd.DataFrame({'ISBN':['001'],'Book-Title':['Title'],'Book-Author':['A'],'Publisher':['P']}).to_csv(processed/'books_clean.csv',index=False)
            pd.DataFrame({'ISBN':['001'],'User-ID':[8],'Book-Rating':[8]}).to_csv(processed/'ratings_clean.csv',index=False)
            (root/'models/content_manifest.json').write_text(json.dumps({'catalog_sha256':'incorrect'}))
            with self.assertRaisesRegex(CatalogError,'catalog changed'):
                load_catalog(root)
    def test_unicode_and_edition_identity(self):
        self.assertEqual(normalize_text('東京物語'), '東京物語')
        self.assertEqual(normalize_text('García Márquez'), 'garcia marquez')
        self.assertNotEqual(work_key('The River','A'), work_key('The River','B'))
        self.assertNotEqual(work_key('Story (Book 1)','A'), work_key('Story (Book 2)','A'))
        self.assertEqual(work_key('Story (Paperback)','A'), work_key('Story','A'))

    def test_known_hobbit_aliases(self):
        original = work_key('The Hobbit', 'J. R. R. Tolkien')
        for title in ['Hobbit', 'The Hobbit: Or There and Back Again', 'Lo Hobbit / The Hobbit', 'El Hobbit', 'Der Kleine Hobbit',
                      'The Hobbit (Illustrated Edition)',
                      'The Annotated Hobbit: The Hobbit, Or, There and Back Again']:
            self.assertEqual(original, work_key(title, 'J.R.R. Tolkien'))
        self.assertNotEqual(original, work_key('Poems from The Hobbit', 'J.R.R. Tolkien'))

    def test_markup_is_text_and_filter_is_specific(self):
        self.assertEqual(display_text('Title &lt;i&gt;edition&lt;/i&gt; &amp; more'), 'Title edition & more')
        self.assertFalse(is_supplement('A summary of my life'))
        self.assertTrue(is_supplement('The River Study Guide'))
        self.assertTrue(is_supplement('The River Study Guides'))
        self.assertTrue(is_supplement('The Hobbit (Coles Notes)'))

    def test_files_are_validated_and_utf8_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); processed=root/'data/processed'; processed.mkdir(parents=True)
            with self.assertRaisesRegex(CatalogError,'Missing files'):
                read_catalog(root)
            pd.DataFrame({'ISBN':['001'], 'Book-Title':['García'], 'Book-Author':['A'], 'Publisher':['P']}).to_csv(processed/'books_clean.csv',index=False)
            pd.DataFrame({'ISBN':['001','001','001'], 'User-ID':[8,8,'invalid'], 'Book-Rating':[8,8,99]}).to_csv(processed/'ratings_clean.csv',index=False)
            books, ratings=read_catalog(root)
            self.assertEqual(books.iloc[0].ISBN,'001')
            self.assertEqual(books.iloc[0]['Book-Title'],'García')
            self.assertEqual(len(ratings),1)
            pd.DataFrame({'ISBN':['001']}).to_csv(processed/'books_clean.csv',index=False)
            with self.assertRaisesRegex(CatalogError,'missing columns'):
                read_catalog(root)


if __name__ == '__main__':
    unittest.main()
