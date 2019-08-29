# -*- coding: utf-8 -*-
import unittest

from glemmazon import Lemmatizer

# Lemmatizers
EN_MODEL = 'models/lemmatizer/en'
PT_MODEL = 'models/lemmatizer/pt'
NL_MODEL = 'models/lemmatizer/nl'


class TestModels(unittest.TestCase):
    def setUp(self):
        self.l_en = Lemmatizer.load(EN_MODEL)

        self.l_pt = Lemmatizer.load(PT_MODEL)

        self.l_nl = Lemmatizer.load(NL_MODEL)

    def test_en(self):
        self.assertEqual(self.l_en('loves', 'VERB'), 'love')

    def test_pt(self):
        self.assertEqual(self.l_pt('amam', 'VERB'), 'amar')

    def test_nl(self):
        self.assertEqual(self.l_nl('sprong', 'VERB'), 'springen')


if __name__ == '__main__':
    unittest.main()
