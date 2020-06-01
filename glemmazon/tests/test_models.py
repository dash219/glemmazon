# -*- coding: utf-8 -*-
import unittest

from glemmazon import Inflector, Lemmatizer

# Lemmatizers
LEM_EN_MODEL = 'models/lemmatizer/en'
LEM_PT_MODEL = 'models/lemmatizer/pt'
LEM_NL_MODEL = 'models/lemmatizer/nl'

# Inflectors
# TODO(gustavoauma): Enable tests for the Inflector
INF_PT_MODEL = 'models/inflector/pt'


class TestModels(unittest.TestCase):
    def setUp(self):
        self.l_en = Lemmatizer.load(LEM_EN_MODEL)

        self.l_pt = Lemmatizer.load(LEM_PT_MODEL)

        self.l_nl = Lemmatizer.load(LEM_NL_MODEL)

        # self.i_pt = Inflector.load(INF_PT_MODEL)

    def test_en(self):
        self.assertEqual(self.l_en(word='loves', pos='VERB'), 'love')

    def test_pt(self):
        self.assertEqual(self.l_pt(word='carros', pos='NOUN'), 'carro')

    def test_nl(self):
        self.assertEqual(self.l_nl(word='maand', pos='VERB'),
                         'maanden')


if __name__ == '__main__':
    unittest.main()
