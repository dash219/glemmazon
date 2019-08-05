# -*- coding: utf-8 -*-
import os
import unittest

from glemmazon import Lemmatizer

TEST_DIR = os.getcwd()

# Inflectors
PT_INFLEC_MD = 'pt_inflec_md.pkl'

# Lemmatizers
EN_MODEL = 'models/en'
PT_MODEL = 'models/pt'
NL_MODEL = 'models/nl'


class TestModels(unittest.TestCase):
    def setUp(self):
        self.l_en = Lemmatizer()
        self.l_en.load(EN_MODEL)

        self.l_pt = Lemmatizer()
        self.l_pt.load(PT_MODEL)

        self.l_nl = Lemmatizer()
        self.l_nl.load(NL_MODEL)

    def test_en(self):
        self.assertEqual(self.l_en('loves', 'VERB'), 'love')

    def test_pt(self):
        self.assertEqual(self.l_pt('amam', 'VERB'), 'amar')

    def test_nl(self):
        self.assertEqual(self.l_nl('sprong', 'VERB'), 'springen')


if __name__ == '__main__':
    unittest.main()
