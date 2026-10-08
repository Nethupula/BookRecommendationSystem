"""Integration tests use the supplied local artifacts; they do not train or overwrite them."""
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless((ROOT/'models/tfidf_matrix.joblib').exists(), 'Local model artifacts required')
class AppFlows(unittest.TestCase):
    def app(self):
        app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=120)
        self.assertEqual(list(app.exception), [])
        return app

    def search_and_add(self, app, term='The Hobbit'):
        app.text_input(key='book_search').set_value(term).run(timeout=90)
        additions=[b for b in app.button if b.key and b.key.startswith('add_') and not b.disabled]
        self.assertTrue(additions)
        isbn=additions[0].key[4:]
        additions[0].click().run(timeout=90)
        return isbn

    def test_selection_survives_search_empty_queries_paging_and_navigation(self):
        app=self.app(); isbn=self.search_and_add(app)
        for term in ('Jane Austen', 'zzzz_no_book', '', '[', 'a'):
            app.text_input(key='book_search').set_value(term).run(timeout=90)
            self.assertIn(isbn, app.session_state.preferences)
            self.assertEqual(list(app.exception), [])
        app.radio(key='mode').set_value('Reading list').run(timeout=90)
        app.button(key='go_discover').click().run(timeout=90)
        self.assertEqual(app.radio(key='mode').value,'Discover')
        self.assertIn(isbn,app.session_state.preferences)
        app.button(key=f'remove_{isbn}').click().run(timeout=90)
        self.assertNotIn(isbn,app.session_state.preferences)

    def test_validation_results_persist_save_export_and_remove(self):
        app=self.app()
        app.button(key='generate_new').click().run(timeout=90)
        self.assertTrue(app.error)
        isbn=self.search_and_add(app)
        app.slider(key=f'rating_{isbn}').set_value(1).run(timeout=90)
        app.button(key='generate_new').click().run(timeout=90)
        self.assertIn('6 or higher',app.error[0].value)
        app.slider(key=f'rating_{isbn}').set_value(9).run(timeout=90)
        app.button(key='generate_new').click().run(timeout=90)
        self.assertEqual(list(app.exception),[])
        rows=app.session_state.discovery_results
        self.assertEqual(len(rows),5)
        app.text_input(key='book_search').set_value('Toni Morrison').run(timeout=90)
        self.assertEqual(app.session_state.discovery_results,rows)
        save=[b for b in app.button if b.key and b.key.startswith('save_') and not b.disabled][0]
        save.click().run(timeout=90)
        self.assertEqual(len(app.session_state.saved),1)
        self.assertEqual(app.session_state.discovery_results,rows)
        app.radio(key='mode').set_value('Reading list').run(timeout=90)
        self.assertEqual(len(app.get('download_button')),1)
        [b for b in app.button if b.key and b.key.startswith('unsave_')][0].click().run(timeout=90)
        self.assertFalse(app.session_state.saved)
        app.radio(key='mode').set_value('Discover').run(timeout=90)
        app.button(key='clear_preferences').click().run(timeout=90)
        self.assertFalse(app.session_state.preferences)
        self.assertIsNone(app.session_state.discovery_results)

    def test_reader_invalid_valid_and_stale_results(self):
        app=self.app(); app.radio(key='mode').set_value('Reader demo').run(timeout=120)
        app.number_input(key='reader_id').set_value(999999).run(timeout=90)
        app.button(key='generate_existing').click().run(timeout=90)
        self.assertTrue(app.error)
        app.number_input(key='reader_id').set_value(8).run(timeout=90)
        app.button(key='generate_existing').click().run(timeout=90)
        self.assertEqual(list(app.exception),[])
        self.assertEqual(len(app.session_state.reader_results),5)
        app.selectbox(key='reader_count').set_value(10).run(timeout=90)
        self.assertTrue(any('changed' in i.value for i in app.info))
        app.button(key='generate_existing').click().run(timeout=90)
        self.assertEqual(len(app.session_state.reader_results),10)
        self.assertEqual(list(app.exception),[])

    def test_five_book_limit_and_pagination(self):
        app=self.app()
        app.text_input(key='book_search').set_value('Jane Austen').run(timeout=90)
        app.button(key='next_page').click().run(timeout=90)
        self.assertEqual(app.session_state.search_page,1)
        app.button(key='previous_page').click().run(timeout=90)
        for _ in range(5):
            additions=[b for b in app.button if b.key and b.key.startswith('add_') and not b.disabled]
            self.assertTrue(additions)
            additions[0].click().run(timeout=90)
        self.assertEqual(len(app.session_state.preferences),5)
        self.assertTrue(all(b.disabled for b in app.button if b.key and b.key.startswith('add_')))
        app.button(key='clear_preferences').click().run(timeout=90)
        self.assertEqual(len(app.session_state.preferences),0)

    def test_z_catalog_failure_is_actionable(self):
        from unittest.mock import patch
        import streamlit as st
        from src.recommender import CatalogError
        st.cache_resource.clear()
        with patch('src.recommender.load_catalog',side_effect=CatalogError('Missing files: data/processed/books_clean.csv')):
            app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=90)
        self.assertEqual(list(app.exception),[])
        self.assertTrue(app.error)
        self.assertIn('Missing files',app.error[0].value)
        st.cache_resource.clear()


if __name__=='__main__':
    unittest.main()
