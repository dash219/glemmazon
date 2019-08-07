# -*- coding: utf-8 -*-
import numpy as np
import unittest

from glemmazon import encoder as t


class TestEncoder(unittest.TestCase):
    def testDenseTag(self):
        enc_tag = t.DenseTag(['_UNK', 'A', 'B', 'C'])
        np.testing.assert_array_equal(enc_tag('_UNK'),
                                      np.array([1., 0., 0., 0.]))
        np.testing.assert_array_equal(enc_tag('B'),
                                      np.array([0., 0., 1., 0.]))
        self.assertEqual(enc_tag.output_shape, (4,))

    def testDenseWordSuffix(self):
        enc_suf = t.DenseWordSuffix(['a', 'b', 'c'], 2, '_UNK')
        np.testing.assert_array_equal(
            enc_suf('ab'), np.array([0., 1., 0., 0.,
                                     0., 0., 1., 0.]))
        # Note that '_UNK' is [1., 0. , 0., 0.]
        np.testing.assert_array_equal(
            enc_suf('dd'), np.array([1., 0., 0., 0.,
                                     1., 0., 0., 0.]))
        self.assertEqual(enc_suf.output_shape, (8,))

    def testDictFeatureEncoder(self):
        dict_enc = t.DictFeatureEncoder({
            'pos': t.DenseTag(['NOUN', 'VERB']),
            'tense': t.DenseTag(['PRES', 'FUT', 'PAST']),
            'word': t.DenseWordSuffix(['a', 'b'], 2, '_UNK'),
        })
        # Note that '_UNK' is [1., 0. , 0., 0.]
        np.testing.assert_array_equal(
            dict_enc({'pos': 'NOUN', 'tense': 'PRES', 'word': 'bb'}),
            np.array([
                0., 1., 0.,  # NOUN
                0., 1., 0., 0.,  # PRES
                0., 0., 1., 0., 0., 1.  # bb
            ]))
        self.assertEqual(dict_enc.output_shape, (13,))
        self.assertCountEqual(dict_enc.scope, {'pos', 'tense', 'word'})

    # -------------------------------------------------------------------
    # Sequence feature encoders
    # -------------------------------------------------------------------
    def testSeqWordSuffix(self):
        seq_word_enc = t.SeqWordSuffix(['a', 'b', 'c'], suffix_length=3)
        np.testing.assert_array_equal(
            seq_word_enc('abbc'),
            np.array([[0., 0., 1., 0.],  # b
                      [0., 0., 1., 0.],  # b
                      [0., 0., 0., 1.],  # c
                      ]))
        self.assertEqual(seq_word_enc.output_shape, (3, 4))

    def testSeqFeatureEncoder(self):
        dict_enc = t.SeqFeatureEncoder(
            'word', t.SeqWordSuffix(['a', 'b', 'c', 'd'],
                                    suffix_length=4),
            t.DictFeatureEncoder({
                'pos': t.DenseTag(['NOUN', 'VERB']),
                'tense': t.DenseTag(['PRES', 'FUT', 'PAST']),
            }))

        # Note that '_UNK' is [1., 0. , 0., 0.]
        np.testing.assert_array_equal(
            dict_enc({'word': 'bcc', 'tense': 'FUT', 'pos': 'NOUN'}),
            np.array([
                [
                    1., 0., 0., 0., 0.,  # _UNK
                    0., 1., 0.,  # NOUN
                    0., 0., 1., 0.  # FUT
                ], [
                    0., 0., 1., 0., 0.,  # b
                    0., 1., 0.,  # NOUN
                    0., 0., 1., 0.  # FUT
                ], [
                    0., 0., 0., 1., 0.,  # c
                    0., 1., 0.,  # NOUN
                    0., 0., 1., 0.  # FUT
                ], [
                    0., 0., 0., 1., 0.,  # c
                    0., 1., 0.,  # NOUN
                    0., 0., 1., 0.  # FUT
                ]]))
        self.assertEqual(dict_enc.output_shape, (4, 12))
        self.assertCountEqual(dict_enc.scope, {'pos', 'tense', 'word'})

    # -------------------------------------------------------------------
    # Label encoders
    # -------------------------------------------------------------------
    def testLabelEncoder(self):
        le = t.LabelEncoder(['_UNK', 'A', 'B', 'C'])
        np.testing.assert_array_equal(le('_UNK'),
                                      np.array([1., 0., 0., 0.]))
        self.assertEqual(le.decode(np.array([1., 0., 0., 0.])), '_UNK')
        self.assertEqual(le.output_shape, (4,))

    def testDictLabelEncoder(self):
        dict_le = t.DictLabelEncoder({
            'lemma_suffix': t.LabelEncoder(['ed', 's', '']),
            'lemma_index': t.LabelEncoder(['1', '2', '3']),
        })
        np.testing.assert_array_equal(
            dict_le({'lemma_suffix': 'ed', 'lemma_index': '2'}),
            np.array([[0., 0., 1., 0.],
                      [0., 1., 0., 0.]]))
        np.testing.assert_array_equal(
            dict_le({'lemma_index': '2', 'lemma_suffix': 'ed'}),
            np.array([[0., 0., 1., 0.],
                      [0., 1., 0., 0.]]))
        self.assertEqual(dict_le.decode(
            [np.array([0., 0., 1., 0.]),
             np.array([0., 1., 0., 0.])]),
            {'lemma_suffix': 'ed', 'lemma_index': '2'})
        self.assertCountEqual(dict_le.scope,
                              {'lemma_suffix', 'lemma_index'})


if __name__ == '__main__':
    unittest.main()
