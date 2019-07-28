"""Main module for the lemmatizer."""

__all__ = ['Lemmatizer']

from typing import Dict, Tuple

import numpy as np
import pickle

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.utils import to_categorical

from glemmazon import preprocess
from glemmazon import utils


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

    def __call__(self, word: str, pos: str) -> str:
        try:
            return self._lookup(word, pos)
        except KeyError:
            op = (self._predict_index(word, pos),
                  self._predict_suffix(word, pos))
            return utils.apply_suffix_op(word, op)

    def load(self, path: str):
        """Load the model from a pickle file."""
        with open(path, 'rb') as reader:
            self.set_model(**pickle.load(reader))

    def load_exceptions(self, path: str):
        """Load exceptions from a .csv file with "word, pos, lemma"."""
        self.exceptions = preprocess.exceptions_to_dict(path)

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
                            utils.revert_dictionary(index_to_ix))
        self.ix_to_suffix = (ix_to_suffix or
                             utils.revert_dictionary(suffix_to_ix))

    def _lookup(self, word: str, pos: str) -> str:
        return self.exceptions[(word, pos)]


    def _predict_index(self, lemma: str, pos: str) -> int:
        ix_pred = self.index_model.predict(self._extract_features(
            lemma, pos)).argmax()
        # Convert to int, as it is originally a string.
        return int(self.ix_to_index[ix_pred])

    def _predict_suffix(self, lemma: str, pos: str) -> str:
        ix_pred = self.suffix_model.predict(self._extract_features(
            lemma, pos)).argmax()
        return self.ix_to_suffix[ix_pred]

    def _extract_features(self, lemma: str, pos: str) -> np.array:
        """Extract features from a list of words and pos."""
        # Convert the morphological tags into vectors.
        tag_vec = utils.encode_labels([pos], self.pos_to_ix)[0]
        tag_vecs = np.array([np.repeat([tag_vec], self.maxlen, axis=0)])

        # Extract character vectors.
        char_vecs = pad_sequences([to_categorical(
            self.tokenizer.texts_to_sequences(lemma),
            len(self.tokenizer.word_index) + 1)], self.maxlen)
        return np.concatenate([char_vecs, tag_vecs], axis=2)

