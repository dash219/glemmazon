"""Functions for preprocessing data."""

__all__ = ['conllu_to_df', 'extract_features', 'prepare_data']

from typing import Callable, Dict, List, Set

import tqdm
import numpy as np
import pyconll

from pandas import DataFrame
from pyconll.unit import Token
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

from glemmazon import cleanup
from glemmazon import string_ops

# Columns from UniversalDependencies.
_WORD_COL = 'word'
_POS_COL = 'pos'
_LEMMA_COL = 'lemma'

# Columns added for the Lemmatizer.
_SUFFIX_COL = '_lemma_suffix'
_INDEX_COL = '_lemma_index'

_UNKNOWN_POS = '_UNKNOWN_POS'


def conllu_to_df(path: str,
                 clean_up: Callable = cleanup.dummy,
                 lemma_suffix_col: str = _SUFFIX_COL,
                 min_count: int = 3) -> DataFrame:
    entries = _conllu_to_tokens(path)
    df = DataFrame(entries)
    df = clean_up(df)
    df = _add_lemma_info(df)

    # Exclude inflection patterns that occur only once.
    df = df.groupby(lemma_suffix_col).filter(
        lambda r: r[lemma_suffix_col].count() > min_count)

    return df


def extract_features(word: str, pos: str) -> dict:
    return {
        '_suffix_1': word[-1:],
        '_suffix_2': word[-2:],
        '_suffix_3': word[-3:],
        '_len_word': len(word),
        '_pos': pos,
    }


# noinspection PyPep8Naming
def prepare_data(df: DataFrame,
                 feature_cols: List[str],
                 lemma_index_col: str = _INDEX_COL,
                 lemma_suffix_col: str = _SUFFIX_COL,
                 test_size: float = 0.2):
    train, test = train_test_split(df, test_size=test_size)

    # Use DataFrame() around "train" and "test" here, otherwise
    # extracting features from them later too slow.
    #
    # TODO(gustavoauma): Find out why iterating over slices is slow.
    X_train = DataFrame(train[feature_cols])
    X_val = DataFrame(test[feature_cols])

    y1_train = train[[lemma_index_col]]
    y1_val = test[[lemma_index_col]]

    y2_train = train[[lemma_suffix_col]]
    y2_val = test[[lemma_suffix_col]]

    # Change the shape of the array: (N, 1) -> (N,).
    y1_train = np.array(y1_train).reshape(-1)
    y1_val = np.array(y1_val).reshape(-1)

    y2_train = np.array(y2_train).reshape(-1)
    y2_val = np.array(y2_val).reshape(-1)

    # Iterate over the rows and extract the features for both training
    # and eval (this might be very slow).
    X_train = _extract_features_df(X_train)
    X_val = _extract_features_df(X_val)

    # Vectorize the features
    vec = DictVectorizer()
    X_train = vec.fit_transform(X_train.T.to_dict().values()).toarray()
    X_val = vec.transform(X_val.T.to_dict().values()).toarray()

    return vec, X_train, y1_train, y2_train, X_val, y1_val, y2_val


class _HashableDict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


def _add_lemma_info(df: DataFrame,
                    word_col: str = _WORD_COL,
                    lemma_col: str = _LEMMA_COL,
                    lemma_suffix_col: str = _SUFFIX_COL,
                    lemma_index_col: str = _INDEX_COL) -> DataFrame:
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
    flattened[_LEMMA_COL] = (token.lemma.lower() if token.lemma
                             else token._form.lower())
    flattened[_POS_COL] = token.upos or _UNKNOWN_POS

    # TODO(gustavoauma): Check whether there is a public attribute.
    flattened[_WORD_COL] = token._form.lower()
    return flattened


def _extract_features_df(df: DataFrame,
                         word_col: str = _WORD_COL,
                         pos_col: str = _POS_COL) -> DataFrame:
    # Add the feature columns to the DataFrame.
    #
    # This is hacky step, but necessary. The feature list is only
    # maintained extract_features, to avoid redundancy. So we cannot
    # really extract this information from anywhere else.
    for k in extract_features('', '').keys():
        df[k] = np.nan

    for i in tqdm.tqdm(df.index):
        word, pos = df.loc[i, [word_col, pos_col]]
        features = extract_features(word, pos)

        # Iterate over each feature from the dict and add its value to
        # the DataFrame cell.
        df.loc[i, features.keys()] = features.values()
    df = df.drop([word_col], axis=1)
    return df
