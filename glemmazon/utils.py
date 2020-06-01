"""Functions for manipulating string suffixes."""

__all__ = [
    'apply_suffix_op',
    'build_index_dict',
    'encode_labels',
    'get_logger',
    'get_suffix_op',
    'revert_dictionary'
]

from typing import List, Tuple

import logging
import os
import numpy as np

from pandas import DataFrame

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.utils import Sequence

from glemmazon import constants as k_
from glemmazon.encoder import DictFeatureEncoder, DictLabelEncoder


class BatchGenerator(Sequence):
    def __init__(self,
                 df: DataFrame,
                 feature_encoders: DictFeatureEncoder,
                 label_encoders: DictLabelEncoder,
                 batch_size=16):
        # Copy and random sample the dataframe "in-place".
        # http://stackoverflow.com/q/29576430
        self.df = df.sample(frac=1).reset_index(drop=True)

        self.feature_encoders = feature_encoders
        self.label_encoders = label_encoders
        self.batch_size = batch_size

        self.n = 0
        self.max = self.__len__()

    def __len__(self):
        return int(np.ceil(len(self.df) / float(self.batch_size)))

    def __getitem__(self, idx: int) -> Tuple[np.array, List[np.array]]:
        batch_df = self.df[
                   idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = []
        batch_y = []
        for _, row in batch_df.iterrows():
            row_dict = dict(row)
            x_dict = {k: v for k, v in row_dict.items() if
                      k in self.feature_encoders.scope}
            y_dict = {k: v for k, v in row_dict.items() if
                      k in self.label_encoders.scope}
            batch_x.append(self.feature_encoders(x_dict))
            batch_y.append(self.label_encoders(y_dict))
        return np.stack(batch_x), [np.vstack(e) for e in zip(*batch_y)]

    def __next__(self):
        if self.n >= self.max:
            self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result


def apply_suffix_op(word: str, op: Tuple[int, str]) -> str:
    r_index, suffix = op

    # Check that the r-index is not larger than the word.
    if r_index * (-1) > len(word):
        raise ValueError('R-index cannot be larger than word length',
                         r_index, len(word), word, op)

    # Case: (0, '')
    if not r_index and not suffix:
        return word
    # Case: (0, 'a')
    elif not r_index and suffix:
        return word + suffix
    # Case: (1, 'a')
    else:
        return word[:r_index * (-1)] + suffix


def build_index_dict(iterable, unknown='_UNK'):
    index_dict = {unknown: 0}
    for e in iterable:
        if e not in index_dict:
            index_dict[e] = len(index_dict)
    return index_dict


def encode_labels(labels, labels_dict):
    return to_categorical([labels_dict[l] for l in labels],
                          len(labels_dict))


def get_suffix_op(a: str, b: str) -> Tuple[int, str]:
    # Case: abc -> abc
    if a == b:
        return 0, ''

    common_prefix = os.path.commonprefix([a, b])
    # Case: abc -> abcd
    if common_prefix == a:
        return 0, b.replace(common_prefix, '', 1)

    # Case: abcd -> abc
    elif common_prefix == b:
        return len(a) - len(common_prefix), ''

    # Case: abc -> abd
    else:
        return (len(a) - len(common_prefix),
                b.replace(common_prefix, '', 1))


def get_logger(model):
    if not os.path.exists(model):
        os.makedirs(model)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(model, k_.LOG_FILE), mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def revert_dictionary(d):
    return {v: k for k, v in d.items()}
