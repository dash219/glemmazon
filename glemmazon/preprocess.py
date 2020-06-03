"""Functions for preprocessing data."""

__all__ = [
    'add_lemmatizer_info',
    'add_inflector_info',
    'conllu_to_df',
]

from typing import Callable, Dict, Set

import pyconll
import tqdm
from pandas import DataFrame
from pyconll.unit import Token

from glemmazon import cleanup
from glemmazon import constants as k
from glemmazon import utils


def add_lemmatizer_info(df: DataFrame,
                        word_col: str = k.WORD_COL,
                        lemma_col: str = k.LEMMA_COL,
                        suffix_col: str = k.SUFFIX_COL,
                        index_col: str = k.INDEX_COL) -> DataFrame:
    # Extract lemma suffix and r_index
    idxs = []
    lemmas = []
    for row in df.itertuples():
        op = utils.get_suffix_op(getattr(row, word_col),
                                 getattr(row, lemma_col))
        idxs.append(op[0])
        lemmas.append(op[1])

    df[suffix_col] = lemmas
    df[index_col] = idxs
    return df


def add_inflector_info(df: DataFrame,
                       word_col: str = k.WORD_COL,
                       lemma_col: str = k.LEMMA_COL,
                       suffix_col: str = k.SUFFIX_COL,
                       index_col: str = k.INDEX_COL) -> DataFrame:
    # Extract lemma suffix and r_index
    idxs = []
    lemmas = []
    for row in df.itertuples():
        op = utils.get_suffix_op(getattr(row, word_col),
                                 getattr(row, lemma_col))
        idxs.append(op[0])
        lemmas.append(op[1])

    df[suffix_col] = lemmas
    df[index_col] = idxs
    return df


def conllu_to_df(path: str,
                 clean_up: Callable = cleanup.dummy,
                 lemma_suffix_col: str = k.SUFFIX_COL,
                 min_count: int = 3,
                 lemmatizer_info: bool = False,
                 inflector_info: bool = False) -> DataFrame:
    entries = _conllu_to_tokens(path)
    df = DataFrame(entries)
    df = clean_up(df)

    if lemmatizer_info:
        df = add_lemmatizer_info(df)

        # Exclude inflection patterns that occur only once.
        df = df.groupby(lemma_suffix_col).filter(
            lambda r: r[lemma_suffix_col].count() > min_count)

    if inflector_info:
        # Fields are intentionally opposite.
        df = add_lemmatizer_info(
            df, word_col=k.LEMMA_COL, lemma_col=k.WORD_COL,
            suffix_col=k.WORD_SUFFIX_COL, index_col=k.WORD_INDEX_COL)

    return df


class _HashableDict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


def _conllu_to_tokens(path: str) -> Set[Dict[str, str]]:
    """Return the annotated tokens from a CoNLL-U file."""

    tokens = set()
    for sentence in tqdm.tqdm(pyconll.load_from_file(path)):
        for token in sentence:
            tokens.add(_HashableDict(_flatten_token(token)))
    return tokens


# noinspection PyProtectedMember
def _flatten_token(token: Token) -> Dict[str, str]:
    """Flatten a CoNLL-U token annotation: {a: {b}} -> {a: b}."""
    flattened = {}
    for feat, val in token.feats.items():
        flattened[feat.lower()] = list(val)[0].lower()

    # Some lemmas are missing from the UD corpora. If that is the case,
    # assume that it coincides with the word form.
    flattened[k.LEMMA_COL] = (token.lemma.lower() if token.lemma
                              else token._form.lower())
    flattened[k.POS_COL] = token.upos or k.UNKNOWN_POS

    # TODO(gustavoauma): Check whether there is a public attribute.
    flattened[k.WORD_COL] = token._form.lower()
    return flattened
