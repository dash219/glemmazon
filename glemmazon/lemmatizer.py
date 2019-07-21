"""Main module for the lemmatizer."""

__all__ = ['Lemmatizer']

import pickle
from typing import Dict, Tuple

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier

from glemmazon import preprocess
from glemmazon import string_ops

LEMMA_SUFFIX_COL = '_lemma_suffix'
LEMMA_INDEX_COL = '_lemma_index'


class Lemmatizer(object):
    """Class to represent a lemmatizer."""

    def __init__(self):
        """Initialize the class."""
        self.clf_index = None
        self.clf_suffix = None
        self.vec = None
        self.exceptions = None

    def __call__(self, word: str, pos: str) -> str:
        try:
            return self._lookup(word, pos)
        except KeyError:
            op = self._predict_suffix_op(word, pos)
            return string_ops.apply_suffix_op(word, op)

    def load(self, path: str):
        """Load the model from a pickle file."""
        with open(path, 'rb') as reader:
            self.set_model(**pickle.load(reader))

    def save(self, path: str):
        """Save the model as a pickle file."""
        with open(path, 'wb') as writer:
            pickle.dump(self.__dict__, writer)

    def set_model(self,
                  clf_index: DecisionTreeClassifier,
                  clf_suffix: DecisionTreeClassifier,
                  vec: DictVectorizer,
                  exceptions: Dict[Tuple[str, str], str] = None):
        self.clf_index = clf_index
        self.clf_suffix = clf_suffix
        self.vec = vec
        self.exceptions = exceptions or dict()

    def _lookup(self, word: str, pos: str) -> str:
        return self.exceptions[(word, pos)]

    def _predict_index(self, word: str, pos: str) -> int:
        features = preprocess.extract_features(word, pos)
        return self.clf_index.predict(
            self.vec.transform(features))[0]

    def _predict_suffix(self, word: str, pos: str) -> str:
        features = preprocess.extract_features(word, pos)
        return self.clf_suffix.predict(
            self.vec.transform(features))[0]

    def _predict_suffix_op(self, word: str, pos: str) -> tuple:
        return (self._predict_index(word, pos),
                self._predict_suffix(word, pos))

    @staticmethod
    def _validate_attribute(attr, attr_type):
        if not isinstance(attr, attr_type):
            raise AttributeError(
                'Model attribute "%s" does not have type "%s".' % (
                    attr, attr_type.__name__))

    def load_exceptions(self, path: str,
                        word_col: str = 'word',
                        pos_col: str = 'pos',
                        lemma_col: str = 'lemma'):
        df = pd.read_csv(path)
        df = df[[word_col, pos_col, lemma_col]].set_index([
            word_col, pos_col])
        self.set_exceptions(df[lemma_col].to_dict())

    def set_exceptions(self, exceptions: Dict[Tuple[str, str], str]):
        self.exceptions = exceptions
