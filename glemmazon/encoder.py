"""Feature and label encoders."""

__all__ = [
    'DenseTag',
    'DenseWordSuffix',
    'DictFeatureEncoder',
    'DictLabelEncoder',
    'LabelEncoder',
    'SeqWordSuffix',
    'SeqFeatureEncoder',
]

from typing import Dict, Iterator, Iterable, List, Tuple, Union

import collections
import logging
import numpy as np

from abc import ABC
from abc import abstractmethod

from tensorflow.keras.utils import to_categorical

from glemmazon import utils

logger = logging.getLogger(__name__)


class FeatureEncoder(ABC):
    """Abstract feature encoder."""

    def __init__(self):
        """Initialize the class."""

    def __call__(self, x) -> np.array:
        # TODO(gustavoauma): For efficiency, refactor _process_input and
        # use operations over numpy arrays instead of list comprehension.
        return x

    @property
    @abstractmethod
    def output_shape(self) -> Tuple[int]:
        raise NotImplementedError


# -----------------------------------------------------------------------
# Dense feature encoders
# -----------------------------------------------------------------------
class DenseTag(FeatureEncoder):
    """Feature encoder for tags"""

    def __init__(self,
                 tags: Iterator[str] = None,
                 unknown: str = '_UNK'):
        super(DenseTag, self).__init__()
        self.tag_to_ix = utils.build_index_dict(tags, unknown)
        self.unknown = unknown

    def __call__(self, tag: str) -> np.array:
        try:
            _tag = self.tag_to_ix[tag]
        except KeyError:
            logger.error('Tag "%s" is not valid. Accepted values: "%s".'
                         'Assigning "%s".' % (
                             tag, list(self.tag_to_ix), self.unknown))
            _tag = self.tag_to_ix[self.unknown]
        return to_categorical([_tag], len(self.tag_to_ix))[0]

    @property
    def output_shape(self) -> Tuple[int]:
        return len(self.tag_to_ix),


class DenseWordSuffix(FeatureEncoder):
    def __init__(self, chars: Iterator[str] = None,
                 suffix_length: int = 3, unknown: str = '_UNK'):
        super(DenseWordSuffix).__init__()
        self.char_to_ix = utils.build_index_dict(chars, unknown)
        self.suffix_length = suffix_length
        self.unknown = unknown

    def __call__(self, word: str) -> np.array:
        char_list = _pad_word_suffix(word, self.suffix_length,
                                     self.unknown)
        self._force_valid_chars(char_list, word)
        return np.concatenate(to_categorical([
            self.char_to_ix[ch] for ch in char_list],
            len(self.char_to_ix)), axis=0)

    @property
    def output_shape(self) -> Tuple[int]:
        return self.suffix_length * len(self.char_to_ix),

    def _force_valid_chars(self, char_list, word):
        for i, ch in enumerate(char_list):
            if ch not in self.char_to_ix:
                logger.error('Char "%s" from word "%s" is not valid. '
                             'Accepted values: "%s". Assigning "%s".' % (
                                 ch, word, list(self.char_to_ix),
                                 self.unknown))
                char_list[i] = self.unknown


class DictFeatureEncoder(FeatureEncoder):
    def __init__(self, encoders: Dict[str, Union[
        DenseTag, DenseWordSuffix]]):
        super(DictFeatureEncoder).__init__()
        self.encoders = collections.OrderedDict()

        # Not ideal, but required for Python 3.5
        # https://stackoverflow.com/questions/47273260
        for p_name, encoder in sorted(encoders.items()):
            self.encoders[p_name] = encoder
        self.scope = set(self.encoders)

    def __call__(self, feature_dict: Dict[str, str]) -> np.array:
        if set(feature_dict) != self.scope:
            raise ValueError(
                'Example tags don\'t match the feature encoders. '
                'Expected features for: %s. Found: %s.' % (
                    sorted(list(self.scope)),
                    sorted(list(feature_dict))))

        dense = []
        for p_name, encoder in self.encoders.items():
            dense.append(encoder(feature_dict[p_name]))
        return np.concatenate(dense, axis=0)

    @property
    def output_shape(self) -> Tuple[int]:
        length = 0
        for p_name in self.encoders.values():
            length += p_name.output_shape[0]
        return length,


# -----------------------------------------------------------------------
# Sequence feature encoders
# -----------------------------------------------------------------------
class SeqWordSuffix(FeatureEncoder):
    def __init__(self, chars: Iterable[str] = None,
                 suffix_length: int = 3, unknown: str = '_UNK'):
        super(SeqWordSuffix).__init__()
        self.char_to_ix = utils.build_index_dict(chars, unknown)
        self.suffix_length = suffix_length
        self.unknown = unknown

    def __call__(self, word: str) -> np.array:
        char_list = _pad_word_suffix(word, self.suffix_length,
                                     self.unknown)
        self._force_valid_chars(char_list, word)
        return to_categorical([
            self.char_to_ix[ch] for ch in char_list],
            len(self.char_to_ix))

    @property
    def output_shape(self) -> Tuple[int, int]:
        return self.suffix_length, len(self.char_to_ix)

    def _force_valid_chars(self, char_list, word):
        for i, ch in enumerate(char_list):
            if ch not in self.char_to_ix:
                logger.error('Char "%s" from word "%s" is not valid. '
                             'Accepted values: "%s". Assigning "%s".' %
                             (ch, word, list(self.char_to_ix),
                              self.unknown))
                char_list[i] = self.unknown


class SeqFeatureEncoder(FeatureEncoder):
    def __init__(self, seq_name: str, seq_encoder: SeqWordSuffix,
                 dense_encoders: DictFeatureEncoder):
        super(SeqFeatureEncoder).__init__()
        self.seq_name = seq_name
        self.seq_encoder = seq_encoder
        self.dense_encoders = dense_encoders
        self.scope = {seq_name} | self.dense_encoders.scope

    def __call__(self, feature_dict: Dict[str, str]) -> np.array:
        d = dict(feature_dict)

        if set(feature_dict) != self.scope:
            raise ValueError(
                'Example tags don\'t match the feature feature_encoders. '
                'Expected features for: %s. Found: %s.' % (
                    sorted(list(self.scope)),
                    sorted(list(feature_dict))))

        seq_vec = self.seq_encoder(feature_dict[self.seq_name])

        del d[self.seq_name]
        dense_vec = self.dense_encoders(d)

        # Repeat and append the 1d dense array into 2d sequence array.
        #
        # e.g. (4, 39) + (2,) -> (4, 41)
        return np.append(seq_vec, np.repeat(
            [dense_vec], seq_vec.shape[0], axis=0), axis=1)

    @property
    def output_shape(self) -> Tuple[int, int]:
        return (self.seq_encoder.output_shape[0],
                self.seq_encoder.output_shape[1] +
                self.dense_encoders.output_shape[0])


def _pad_word_suffix(word: str,
                     maxlen: int,
                     unknown: str = '_UNK') -> List[str]:
    return [unknown] * (maxlen - len(word)) + list(word)[-maxlen:]


# -----------------------------------------------------------------------
# Label encoders
# -----------------------------------------------------------------------
class LabelEncoder(DenseTag):
    def __init__(self, tags: Iterator[str] = None,
                 unknown: str = '_UNK'):
        super(LabelEncoder).__init__()
        self.tag_to_ix = utils.build_index_dict(tags, unknown)
        self.ix_to_tag = utils.revert_dictionary(self.tag_to_ix)
        self.unknown = unknown

    def decode(self, ar: np.array) -> str:
        return self.ix_to_tag[ar.argmax()]


class DictLabelEncoder(LabelEncoder):
    def __init__(self, encoders: Dict[str, LabelEncoder]):
        super(DictLabelEncoder).__init__()
        self.encoders = collections.OrderedDict()
        # Not ideal, but required for Python 3.5
        # https://stackoverflow.com/questions/47273260
        for p_name, encoder in sorted(encoders.items()):
            self.encoders[p_name] = encoder
        self.scope = set(self.encoders)

    def __call__(self, label_dict: Dict[str, str]) -> List[np.array]:
        if set(label_dict) != self.scope:
            raise ValueError(
                'Example label doesn\'t match the label encoders. '
                'Expected: %s. Found: %s.' % (
                    list(self.scope), list(label_dict)))

        dense = []
        for e_name, encoder in self.encoders.items():
            dense.append(encoder(label_dict[e_name]))
        return dense

    def to_single_vec(self, label_dict: Dict[str, str]) -> np.array:
        """Encode labels into a single vector with shape (X,)."""
        return np.hstack(self.__call__(label_dict))

    def from_single_vec(self, vec: np.array) -> Dict[str, str]:
        """Decode labels from a single vector with shape (X,)."""
        return self.decode(self._split_single_vec(vec))

    def _split_single_vec(self, vec: np.array) -> List[np.array]:
        """Split a single vector into a list of feature vectors."""
        arrays = []
        prev_idx = 0
        for enc in self.encoders.values():
            vec_length = enc.output_shape[0]
            arrays.append(vec[prev_idx:prev_idx + vec_length])
            prev_idx += vec_length
        return arrays

    @property
    def output_shape(self) -> Tuple[int]:
        length = 0
        for p_name in self.encoders.values():
            length += p_name.output_shape[0]
        return length,

    def decode(self, ars: List[np.array]) -> Dict[str, str]:
        if len(ars) != len(self.encoders):
            raise ValueError(
                'Array list doesn\'t match the number of encoders.'
                'Expected: %s. Found: %s.' % (
                    len(self.encoders), len(ars)))

        labels = {}
        for ar, (e_name, encoder) in zip(ars, self.encoders.items()):
            labels[e_name] = encoder.decode(ar)
        return labels
