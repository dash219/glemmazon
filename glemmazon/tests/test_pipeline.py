# -*- coding: utf-8 -*-
import unittest

import pandas as pd

from glemmazon import pipeline

PT_MODEL_DIR = 'models/lemmatizer/pt'


class TestPipeline(unittest.TestCase):

    def setUp(self):
        df = pd.DataFrame({'word': ['amo', 'amaram', 'amaram'],
                           'tense': ['pres', 'past', 'plus'],
                           'person': ['1', '3', '3']})
        self.ldict = pipeline.LookupDictionary(df)

    def test_lookup(self):
        self.assertEqual(
            self.ldict.lookup(word='amaram', tense='past'),
            [{'word': 'amaram', 'tense': 'past', 'person': '3'}])

        self.assertEqual(
            self.ldict.lookup(word='amo', keep_cols=['tense', 'word']),
            [{'word': 'amo', 'tense': 'past'}])

        self.assertEqual(
            self.ldict.lookup(word='amo', omit_cols=['tense']),
            [{'word': 'amo', 'person': '1'}])

    def test_invalid_lookup(self):
        with self.assertRaises(KeyError):
            self.ldict.lookup(word='invalid_value')

        with self.assertRaises(ValueError):
            self.ldict.lookup(invalid_key='amo')

        with self.assertRaises(ValueError):
            self.ldict.lookup()

    def test_add_entry(self):
        new_entry = {'word': 'amaria', 'tense': 'cnd', 'person': '1'}
        self.ldict.add_entry(**new_entry)
        self.assertEqual([new_entry], self.ldict.lookup(word='amaria'))

    def test_add_entry_empty_df(self):
        new_entry = {'word': 'amaria', 'tense': 'cnd', 'person': '1'}
        ldict = pipeline.LookupDictionary(columns=['word', 'tense',
                                                   'person'])
        ldict.add_entry(**new_entry)

    def test_invalid_add_entry(self):
        with self.assertRaises(ValueError):
            self.ldict.add_entry(word='fail')  # Missing other columns

        with self.assertRaises(ValueError):
            new_entry = {'word': 'amaria', 'tense': 'cnd',
                         'person': '1', 'extra_field': 'X'}
            self.ldict.add_entry(**new_entry)  # Extra column

    def test__build_query(self):
        self.assertEqual(
            "word == 'teste' and pos == 'NOUN'",
            self.ldict._build_query(word='teste', pos='NOUN'))

    def test_lookup(self):
        self.assertEqual(
            self.ldict.lookup(word='amaram', tense='past'),
            [{'word': 'amaram', 'tense': 'past', 'person': '3'}])

    def test_load(self):
        self.assertTrue(pipeline.Lemmatizer.load(PT_MODEL_DIR))


if __name__ == '__main__':
    unittest.main()
