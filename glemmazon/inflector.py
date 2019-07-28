"""Main module for the morphology inflector."""

__all__ = ['Inflector']

from typing import Dict

import re

import numpy as np
import pickle

from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from pandas import DataFrame

from glemmazon import constants as k
from glemmazon import utils


def _query_from_kwargs(lemma, **kwargs):
    """Turn a dictionary with features into a DataFrame query."""
    kwargs[k.LEMMA_COL] = lemma
    return ' and '.join(["%s == '%s'" % (key, re.escape(val))
                         for key, val in kwargs.items()])


class Inflector(object):
    """Class to represent an inflector."""

    def __init__(self):
        """Initialize the class."""
        self.index_model = None
        self.suffix_model = None
        self.index_to_ix = None
        self.suffix_to_ix = None
        self.feature_to_ix = None

        self.maxlen = None
        self.tokenizer = None
        self.exceptions = None

        self.ix_to_suffix = None
        self.ix_to_index = None

    def __call__(self, lemma: str, **kwargs) -> str:
        try:
            return self._lookup(lemma, **kwargs)
        # TODO(gustavoauma): Make this exception less broad.
        except IndexError:
            op = (self._predict_index(lemma, **kwargs),
                  self._predict_suffix(lemma, **kwargs))
            return utils.apply_suffix_op(lemma, op)

    def load(self, path: str):
        """Load the model from a pickle file."""
        with open(path, 'rb') as reader:
            self.set_model(**pickle.load(reader))

    def save(self, path: str):
        """Save the model as a pickle file."""
        with open(path, 'wb') as writer:
            pickle.dump(self.__dict__, writer)

    def set_model(self,
                  index_model: Sequential,
                  suffix_model: Sequential,
                  index_to_ix: Dict[str, int],
                  suffix_to_ix: Dict[str, int],
                  feature_to_ix: Dict[str, Dict[str, int]],
                  tokenizer: Tokenizer,
                  maxlen: int,
                  exceptions: DataFrame = None,
                  ix_to_suffix: Dict[str, int] = None,
                  ix_to_index: Dict[str, int] = None):
        self.index_model = index_model
        self.suffix_model = suffix_model
        self.index_to_ix = index_to_ix
        self.suffix_to_ix = suffix_to_ix
        self.feature_to_ix = feature_to_ix
        self.maxlen = maxlen
        self.tokenizer = tokenizer
        self.exceptions = exceptions

        self.ix_to_index = (ix_to_index or
                            utils.revert_dictionary(index_to_ix))
        self.ix_to_suffix = (ix_to_suffix or
                             utils.revert_dictionary(suffix_to_ix))

    def _lookup(self, lemma: str, **kwargs) -> str:
        try:
            return self.exceptions.query(_query_from_kwargs(
                lemma, **kwargs)).iloc[0].values[0]
        except IndexError:
            raise IndexError(
                "Could not find entry in the exceptions '%s' (%s)" %
                (lemma, kwargs))

    def _predict_index(self, lemma: str,
                       **kwargs) -> int:
        ix_pred = self.index_model.predict(self._extract_features(
            lemma, **kwargs)).argmax()
        # Convert to int, as it is originally a string.
        return int(self.ix_to_index[ix_pred])

    def _predict_suffix(self,
                        lemma: str,
                        **kwargs) -> str:
        ix_pred = self.suffix_model.predict(self._extract_features(
            lemma, **kwargs)).argmax()
        return self.ix_to_suffix[ix_pred]

    def _extract_features(self, lemma: str,
                          unknown=k.UNKNOWN_TAG,
                          **kwargs) -> np.array:
        """Extract features from a list of words and pos."""
        # TODO(gustavoauma): Check that this assertion is not too slow.
        # If slow, explicitly define the arguments for each language, 
        # instead of using kwargs.
        assert set(kwargs).issubset(self.feature_to_ix)

        # Convert the morphological tags into vectors.
        tag_vec = []
        for tag, ix in self.feature_to_ix.items():
            if tag in kwargs:
                tag_value = kwargs[tag]
            else:
                tag_value = unknown
            tag_vec.extend(utils.encode_labels([
                tag_value], self.feature_to_ix[tag])[0])

        tag_vecs = np.array([np.repeat([tag_vec], self.maxlen, axis=0)])

        # Extract character vectors.
        char_vecs = pad_sequences([to_categorical(
            self.tokenizer.texts_to_sequences(lemma),
            len(self.tokenizer.word_index) + 1)], self.maxlen)
        return np.concatenate([char_vecs, tag_vecs], axis=2)

    def load_exceptions(self, df: DataFrame):
        self.validate_exceptions(df)
        df = df.set_index([c for c in df.columns if c != k.WORD_COL])
        self.exceptions = df

    def validate_exceptions(self, df: DataFrame):
        # Check that the features are compatible with the model.
        _features_to_ix = {}
        for col in df.columns:
            # TODO(gustavouma): Refactor this. The load function should
            #  not create a DataFrame with new columns.
            if col in [k.WORD_COL, k.LEMMA_COL, k.SUFFIX_COL,
                       k.INDEX_COL, k.WORD_SUFFIX_COL,
                       k.WORD_INDEX_COL]:
                continue
            _features_to_ix[col] = utils.build_index_dict(
                getattr(df, col))
        if not all(item in self.feature_to_ix.items() for item in
                   _features_to_ix.items()):
            raise TypeError('Exceptions DataFrame columns do not match '
                            ' the model. Expected: %s, found: %s.' %
                            (self.feature_to_ix, _features_to_ix))
