"""Main module for the morphology inflector."""

__all__ = ['Inflector']

from typing import Dict, Tuple

import os
import re

import numpy as np
import pickle

from tensorflow.keras.models import load_model, Model
from pandas import DataFrame

from glemmazon import constants as k
from glemmazon.encoder import DictFeatureEncoder, DictLabelEncoder
from glemmazon import utils


def _query_from_kwargs(lemma, **kwargs):
    """Turn a dictionary with features into a DataFrame query."""
    kwargs[k.LEMMA_COL] = lemma
    return ' and '.join(["%s == '%s'" % (key, re.escape(val))
                         for key, val in kwargs.items()])


class Inflector(object):
    """Class to represent an inflector."""

    def __init__(self):
        """Initialize the class."""
        self.model = None
        self.feature_enc = None
        self.label_enc = None
        self.exceptions = None or DataFrame()

    def __call__(self, lemma: str, **kwargs: str) -> str:
        try:
            raise KeyError
        # TODO(gustavoauma): Make this exception less broad.
        except (IndexError, KeyError):
            return utils.apply_suffix_op(lemma, self._predict_op(
                lemma, **kwargs))

    def load(self, folder: str):
        """Load the model from a folder."""
        with open(os.path.join(folder, k.PARAMS_FILE), 'rb') as reader:
            self.set_model(**{
                **{'model': load_model(
                    os.path.join(folder, k.MODEL_FILE))},
                **pickle.load(reader)})

    def save(self, folder: str):
        """Save the model to a folder."""
        if not os.path.exists(folder):
            os.mkdir(folder)
        self.model.save(os.path.join(folder, k.MODEL_FILE))
        with open(os.path.join(folder, k.PARAMS_FILE), 'wb') as writer:
            pickle.dump({
                'exceptions': self.exceptions,
                'feature_enc': self.feature_enc,
                'label_enc': self.label_enc,
            }, writer)

    def set_model(self,
                  model: Model,
                  feature_enc: DictFeatureEncoder,
                  label_enc: DictLabelEncoder,
                  exceptions: Dict[Tuple[str, str], str] = None):
        self.model = model
        self.feature_enc = feature_enc
        self.label_enc = label_enc
        self.exceptions = (exceptions if not exceptions.empty else
                           DataFrame(columns=feature_enc.scope | {
                           k.WORD_COL}))

    def _lookup(self, lemma: str, **kwargs) -> str:
        try:
            return self.exceptions.query(_query_from_kwargs(
                lemma, **kwargs)).iloc[0].values[0]
        except IndexError:
            raise IndexError(
                "Could not find entry in the exceptions for '%s' (%s)" %
                (lemma, kwargs))

    def _predict_op(self,
                    lemma: str,
                    fill_na=False,
                    **kwargs: str) -> Tuple[int, str]:
        if fill_na:
            for feature in self.feature_enc.scope:
                if feature != k.LEMMA_COL and feature not in kwargs:
                    kwargs[feature] = k.UNKNOWN_TAG

        features = [self.feature_enc({k.LEMMA_COL: lemma, **kwargs})]
        y_pred_dict = self.label_enc.decode(self.model.predict(
            np.array(features)))
        return int(y_pred_dict[k.WORD_INDEX_COL]), y_pred_dict[
            k.WORD_SUFFIX_COL]

    def load_exceptions(self, df: DataFrame):
        self.validate_exceptions(df)
        df = df.set_index([c for c in df.columns if c != k.WORD_COL])
        self.exceptions = df

    def validate_exceptions(self, df: DataFrame):
        # Check that the features are compatible with the model.
        _features_to_ix = {}
        for col in df.columns:
            # TODO(gustavouma): Refactor this. The load function should
            #  not create a DataFrame with new columns.
            if col in [k.WORD_COL, k.LEMMA_COL, k.SUFFIX_COL,
                       k.INDEX_COL, k.WORD_SUFFIX_COL,
                       k.WORD_INDEX_COL]:
                continue
            _features_to_ix[col] = utils.build_index_dict(
                getattr(df, col))
        if not all(item in self.feature_to_ix.items() for item in
                   _features_to_ix.items()):
            raise TypeError('Exceptions DataFrame columns do not match '
                            ' the model. Expected: %s, found: %s.' %
                            (self.feature_to_ix, _features_to_ix))
