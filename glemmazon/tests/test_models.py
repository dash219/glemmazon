# -*- coding: utf-8 -*-
import unittest

from glemmazon import constants as k
from glemmazon import Analyzer, Inflector, Lemmatizer

# Analyzers
ANA_PT_MODEL = 'models/analyzer/pt'

# Lemmatizers
LEM_EN_MODEL = 'models/lemmatizer/en'
LEM_PT_MODEL = 'models/lemmatizer/pt'
LEM_NL_MODEL = 'models/lemmatizer/nl'

# Inflectors
# TODO(gustavoauma): Enable tests for the Inflector
INF_PT_MODEL = 'models/inflector/pt'


class TestModels(unittest.TestCase):
    def test_analyzer_pt(self):
        analyzer = Analyzer.load(ANA_PT_MODEL)
        expected = {
            'mood': 'ind', 'number': 'sing', 'person': '3',
            'pos': 'VERB', 'tense': 'imp', 'verbform': 'fin'
        }
        for attr in (
                'polarity', 'prontype', 'reflex', 'voice', 'case',
                'definite', 'degree', 'foreign', 'gender', 'numtype'
        ):
            expected[attr] = k.UNSPECIFIED_TAG
        self.assertEqual(analyzer(word='amava', pos='VERB'), expected)

    def test_lemmatizer_en(self):
        lemmatizer = Lemmatizer.load(LEM_EN_MODEL)
        self.assertEqual(lemmatizer(word='loves', pos='VERB'), 'love')

    def test_lemmatizer_pt(self):
        lemmatizer = Lemmatizer.load(LEM_PT_MODEL)
        self.assertEqual(lemmatizer(word='carros', pos='NOUN'), 'carro')

    def test_lemmatizer_nl(self):
        lemmatizer = Lemmatizer.load(LEM_NL_MODEL)
        self.assertEqual(lemmatizer(word='maand', pos='VERB'),
                         'maanden')


if __name__ == '__main__':
    unittest.main()
