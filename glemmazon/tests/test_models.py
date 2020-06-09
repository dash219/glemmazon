# -*- coding: utf-8 -*-
import unittest

from glemmazon import Analyzer, Inflector, Lemmatizer
from glemmazon import constants as k
from glemmazon.pipeline import Result, Source

# Analyzers
ANA_PT_MODEL = 'models/analyzer/pt'

# Lemmatizers
LEM_EN_MODEL = 'models/lemmatizer/en'
LEM_PT_MODEL = 'models/lemmatizer/pt'
LEM_NL_MODEL = 'models/lemmatizer/nl'

# Inflectors
# TODO(gustavoauma): Enable tests for the Inflector
INF_EN_MODEL = 'models/inflector/en'
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
        self.assertEqual(analyzer(word='amava', pos='VERB').data,
                         expected)

    def test_inflector_en(self):
        inflector = Inflector.load(INF_EN_MODEL)
        self.assertEqual(inflector.get_word(
            lemma='love', tense='past', pos='VERB', fill_na=True),
            'loved')

    def test_inflector_pt(self):
        inflector = Inflector.load(INF_PT_MODEL)
        self.assertEqual(inflector.get_word(
            lemma='amar', tense='imp', pos='VERB', person='1',
            fill_na=True), 'amava')

    def test_lemmatizer_en(self):
        lemmatizer = Lemmatizer.load(LEM_EN_MODEL)
        self.assertEqual(lemmatizer.get_lemma(word='loves', pos='VERB'),
                         'love')
        self.assertEqual(
            lemmatizer(word='loves', pos='VERB'),
            Result(data={'lemma': 'love'}, source=Source.MODEL))

    def test_lemmatizer_pt(self):
        lemmatizer = Lemmatizer.load(LEM_PT_MODEL)
        self.assertEqual(
            lemmatizer.get_lemma(word='carros', pos='NOUN'), 'carro')

    def test_lemmatizer_nl(self):
        lemmatizer = Lemmatizer.load(LEM_NL_MODEL)
        self.assertEqual(lemmatizer.get_lemma(word='maand', pos='VERB'),
                         'maanden')


if __name__ == '__main__':
    unittest.main()
