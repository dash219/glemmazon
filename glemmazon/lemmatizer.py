"""Main module for the lemmatizer."""

__all__ = ['Lemmatizer']

from typing import Dict, Iterator, List, Tuple

import numpy as np
import pickle

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.utils import to_categorical

from glemmazon import preprocess
from glemmazon import string_ops


def _build_index_dict(iterable):
    index_dict = {}
    for e in iterable:
        if e not in index_dict:
            index_dict[e] = len(index_dict)
    return index_dict


def _encode_labels(labels, labels_dict):
    return to_categorical(
        [labels_dict[l] for l in labels],
        len(labels_dict))


def _revert_dictionary(dic):
    return {val: key for key, val in dic.items()}


class Lemmatizer(object):
    """Class to represent a lemmatizer."""

    def __init__(self):
        """Initialize the class."""
        self.index_model = None
        self.suffix_model = None
        self.index_to_ix = None
        self.pos_to_ix = None
        self.suffix_to_ix = None
        self.maxlen = None
        self.tokenizer = None
        self.exceptions = None

        self.ix_to_suffix = None
        self.ix_to_index = None

    def __call__(self,
                 word_list: List[str],
                 pos_list: List[str]) -> List[str]:
        lemmas = []
        # TODO(gustavoauma): Refactor this method. The loop here is
        # inefficient. This really should take advantage of the fact
        # that predict() receives an array as input.
        for word, pos in zip(word_list, pos_list):
            try:
                lemmas.append(self._lookup(word, pos))
            except KeyError:
                op = next(self._yield_suffix_op([word], [pos]))
                lemmas.append(string_ops.apply_suffix_op(word, op))
        return lemmas

    def yield_lemmas(self,
                     word_list: List[str],
                     pos_list: List[str]) -> List[str]:
        for i, op in enumerate(self._yield_suffix_op(word_list,
                                                     pos_list)):
            yield string_ops.apply_suffix_op(word_list[i], op)

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
                  pos_to_ix: Dict[str, int],
                  tokenizer: Tokenizer,
                  maxlen: int,
                  exceptions: Dict[Tuple[str, str], str] = None,
                  ix_to_suffix: Dict[str, int] = None,
                  ix_to_index: Dict[str, int] = None):
        self.index_model = index_model
        self.suffix_model = suffix_model
        self.index_to_ix = index_to_ix
        self.suffix_to_ix = suffix_to_ix
        self.pos_to_ix = pos_to_ix
        self.maxlen = maxlen
        self.tokenizer = tokenizer
        self.exceptions = exceptions or dict()

        self.ix_to_index = (ix_to_index or
                            _revert_dictionary(index_to_ix))
        self.ix_to_suffix = (ix_to_suffix or
                             _revert_dictionary(suffix_to_ix))

    def _lookup(self, word: str, pos: str) -> str:
        return self.exceptions[(word, pos)]

    def _predict_index(self, word_list: List[str],
                       pos_list: List[str]):
        return [self.ix_to_index[vec.argmax()] for vec in
                self.index_model.predict(self._extract_features(
                    word_list, pos_list))]

    def _predict_suffix(self,
                        word_list: List[str],
                        pos_list: List[str]) -> List[str]:
        return [self.ix_to_suffix[vec.argmax()] for vec in
                self.suffix_model.predict(self._extract_features(
                    word_list, pos_list))]

    def _yield_suffix_op(self,
                         word_list: List[str],
                         pos_list: List[str]) -> Iterator:
        for op in zip(self._predict_index(word_list, pos_list),
                      self._predict_suffix(word_list, pos_list)):
            yield op

    def _extract_features(self,
                          word_list: List[str],
                          pos_list: List[str]) -> np.array:
        """Extract features from a list of words and pos."""
        word_features = pad_sequences(
            self.tokenizer.texts_to_sequences(word_list), self.maxlen)
        pos_features = _encode_labels(pos_list, self.pos_to_ix)
        return np.concatenate([word_features, pos_features], axis=1)

    @staticmethod
    def _validate_attribute(attr, attr_type):
        if not isinstance(attr, attr_type):
            raise AttributeError(
                'Model attribute "%s" does not have type "%s".' % (
                    attr, attr_type.__name__))

    def load_exceptions(self, path: str):
        self.set_exceptions(preprocess.exceptions_to_dict(path))

    def set_exceptions(self, exceptions: Dict[Tuple[str, str], str]):
        self.exceptions = exceptions
