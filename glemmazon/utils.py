"""Functions for manipulating string suffixes."""

__all__ = [
    'apply_suffix_op',
    'build_index_dict',
    'encode_labels',
    'get_suffix_op',
    'revert_dictionary'
]

from typing import Tuple

import os

from keras.utils import to_categorical


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


def revert_dictionary(d):
    return {v: k for k, v in d.items()}
