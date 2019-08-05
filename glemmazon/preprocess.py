"""Functions for preprocessing data."""

__all__ = [
    'add_lemmatizer_info',
    'add_inflector_info',
    'conllu_to_df',
    'exceptions_to_dict',
    'unimorph_to_df',
]

from typing import Callable, Dict, Tuple, Set

import tqdm
import pandas as pd
import pyconll

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
                 min_count: int = 3) -> DataFrame:
    entries = _conllu_to_tokens(path)
    df = DataFrame(entries)
    df = clean_up(df)
    df = add_lemmatizer_info(df)

    # Exclude inflection patterns that occur only once.
    df = df.groupby(lemma_suffix_col).filter(
        lambda r: r[lemma_suffix_col].count() > min_count)

    return df


def exceptions_to_dict(path: str) -> Dict[Tuple[str,str], str]:
    df = pd.read_csv(path)
    df = df[[k.WORD_COL, k.POS_COL, k.LEMMA_COL]].set_index([k.WORD_COL,
                                                             k.POS_COL])
    return df[k.LEMMA_COL].to_dict()


def unimorph_to_df(path: str,
                   mapping_path: str,
                   clean_up: Callable = None,
                   lemmatizer_cols: bool = False,
                   inflector_cols: bool = False,
                   unknown: str = '_UNK',
                   ) -> DataFrame:
    """Read a UniMorph file as a DataFrame."""
    df = pd.read_csv(path, delimiter='\t', names=[
        k.LEMMA_COL, k.WORD_COL, k.MORPH_FEATURES_COL],
                     keep_default_na=False)

    # Adapt UniMorph tags to UniversalDependency.
    #
    # Note: it is actually much faster to build a new DataFrame, in
    # comparison with modifying an existing one in-place.
    if mapping_path:
        mapping_df = pd.read_csv(mapping_path)
        mapping_df = mapping_df.set_index('unimorph')
        entries = []
        for _, row in tqdm.tqdm(df.iterrows()):
            entry_feats = {'lemma': row['lemma'], 'word': row['word']}
            for morph_tag in row[k.MORPH_FEATURES_COL].split(';'):
                mapping_df.loc[morph_tag].to_list()
                ud_feature, ud_tag = mapping_df.loc[morph_tag].to_list()
                entry_feats[ud_feature] = ud_tag.upper()
            entries.append(entry_feats)
        df = pd.DataFrame(entries)

    if clean_up:
        df = clean_up(df)

    if lemmatizer_cols:
        df = add_lemmatizer_info(df)

    if inflector_cols:
        # Fields are intentionally opposite.
        df = add_lemmatizer_info(
            df, word_col=k.LEMMA_COL, lemma_col=k.WORD_COL,
            suffix_col=k.WORD_SUFFIX_COL, index_col=k.WORD_INDEX_COL)

    if unknown:
        df = df.fillna(unknown)

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
