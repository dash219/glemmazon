"""Functions for preprocessing data."""

__all__ = ['conllu_to_df', 'unimorph_to_df']

from typing import Callable, Dict, Set

import tqdm
import pandas as pd
import pyconll

from pandas import DataFrame
from pyconll.unit import Token

from glemmazon import cleanup
from glemmazon import string_ops
from glemmazon import constants as k

# Note: the reason for having a hard-coded dictionary here, instead of
# reusing the available mapping from UniMorph, is because there are
# ambiguous cases, e.g. 'V' (unimorph) -> 'AUX,VERB' (UD); and we need
# to set explicit preferences.
_POS_UNIMORPH2UD = {
    'ADV': 'ADV',
    'PRO': 'PRON',
    'V': 'VERB',
    'ADP': 'ADP',
    'DET': 'DET',
    'N': 'NOUN',
    'ADJ': 'ADJ',
    'CONJ': 'CCONJ',
    'PUNCT': 'X',
    'NUM': 'NUM',
    'PROPN': 'PROPN',
    'PART': 'PART',
    'INTJ': 'INTJ',
    'V.PTCP': 'VERB',
}


def conllu_to_df(path: str,
                 clean_up: Callable = cleanup.dummy,
                 lemma_suffix_col: str = k.SUFFIX_COL,
                 min_count: int = 3) -> DataFrame:
    entries = _conllu_to_tokens(path)
    df = DataFrame(entries)
    df = clean_up(df)
    df = _add_lemma_info(df)

    # Exclude inflection patterns that occur only once.
    df = df.groupby(lemma_suffix_col).filter(
        lambda r: r[lemma_suffix_col].count() > min_count)

    return df


def unimorph_to_df(path: str,
                   clean_up: Callable = cleanup.dummy,
                   lemma_suffix_col: str = k.SUFFIX_COL,
                   min_count: int = 3) -> DataFrame:
    df = pd.read_csv(path, delimiter='\t', names=[
        k.LEMMA_COL, k.WORD_COL, k.MORPH_FEATURES_COL],
                     keep_default_na=False)

    df[k.POS_COL] = df[k.MORPH_FEATURES_COL].apply(
        lambda x: _POS_UNIMORPH2UD[x.split(';')[0]])

    df = df.drop(k.MORPH_FEATURES_COL, axis=1)

    df = clean_up(df)
    print(df.head())
    df = _add_lemma_info(df)

    # Exclude inflection patterns that occur only once.
    df = df.groupby(lemma_suffix_col).filter(
        lambda r: r[lemma_suffix_col].count() > min_count)

    return df


class _HashableDict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


def _add_lemma_info(df: DataFrame,
                    word_col: str = k.WORD_COL,
                    lemma_col: str = k.LEMMA_COL,
                    lemma_suffix_col: str = k.SUFFIX_COL,
                    lemma_index_col: str = k.INDEX_COL) -> DataFrame:
    # Extract lemma suffix and r_index
    idxs = []
    lemmas = []
    for row in df.itertuples():
        op = string_ops.get_suffix_op(getattr(row, word_col),
                                      getattr(row, lemma_col))
        idxs.append(op[0])
        lemmas.append(op[1])

    df[lemma_suffix_col] = lemmas
    df[lemma_index_col] = idxs
    return df


def _conllu_to_tokens(path: str) -> Set[Dict[str, str]]:
    """Return the annotated tokens from a CoNLL-U file."""

    tokens = set()
    for sentence in tqdm.tqdm(pyconll.load_from_file(path)):
        for token in sentence:
            tokens.add(_HashableDict(_flatten_token(token)))
    return tokens


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
