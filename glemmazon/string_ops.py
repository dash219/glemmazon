"""Functions for manipulating string suffixes."""

__all__ = ['apply_suffix_op', 'get_suffix_op']

import os

from typing import Tuple


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
