"""Main module for the morphology inflector."""

__all__ = ['Inflector']

from typing import Dict, Tuple

import numpy as np
from pandas import DataFrame
from tensorflow.keras.models import Model

from glemmazon import constants as k
from glemmazon import preprocess
from glemmazon import utils
from glemmazon.encoder import DictFeatureEncoder, DictLabelEncoder
from glemmazon.pipeline import BasePipeline, LookupDictionary


class Inflector(BasePipeline):
    """Class to represent an inflector."""

    def __init__(self,
                 model: Model,
                 feature_enc: DictFeatureEncoder,
                 label_enc: DictLabelEncoder,
                 exceptions: LookupDictionary = None,
                 first_call: bool = True):
        """Initialize the class.

        Args:
            model (Model): The model to be loaded.
            feature_enc (DictFeatureEncoder): Feature encoder.
            label_enc (DictLabelEncoder): Label encoder.
            exceptions (LookupDictionary): Exceptions dictionary.
            first_call (bool): If True, make a first call to the model
                when it is loaded, to avoid latency issues. The first
                iteration of predict() is slower due to caching:
                https://stackoverflow.com/questions/55577711
        """
        self.model = model
        self.feature_enc = feature_enc
        self.label_enc = label_enc
        self.exceptions = exceptions or LookupDictionary()

        # Make a fake first call to predict to avoid latency issues.
        if first_call:
            self.annotate(**self.dummy_example)

    def __call__(self, lemma: str, **kwargs: str) -> str:
        try:
            raise KeyError
        # TODO(gustavoauma): Make this exception less broad.
        except (IndexError, KeyError):
            return utils.apply_suffix_op(lemma, self._predict_op(
                lemma, **kwargs))

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
                    fill_na: bool = False,
                    **kwargs: Dict[str, str]) -> Tuple[int, str]:
        """Return the string operation for the lemma as (index, str).

        Args:
            lemma (str): The lemma.
            fill_na (bool): If True, will fill all unspecified
                morphological features with k.UNKNOWN_TAG.
            **kwargs (Dict[str, str]): Keyword arguments with the
                morphological features.
        """
        if fill_na:
            for feature in self.feature_enc.scope:
                if feature != k.LEMMA_COL and feature not in kwargs:
                    kwargs[feature] = k.UNKNOWN_TAG

        features = [self.feature_enc({k.LEMMA_COL: lemma, **kwargs})]
        y_pred_dict = self.label_enc.decode(self.model.predict(
            np.array(features)))
        return (int(y_pred_dict[k.WORD_INDEX_COL]),
                y_pred_dict[k.WORD_SUFFIX_COL])

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

    @property
    def dummy_example(self) -> Dict[str, str]:
        return ['amar', {'aspect': 'IMP'}]

    def annotate(self, word: str, pos: str):
        """Annotate a single example (using the model)."""
        return utils.apply_suffix_op(word, self._predict_op(word, pos))

    def load_exceptions(self, path: str):
        """Load exceptions from a .csv file with "word, pos, lemma"."""
        self.exceptions = preprocess.exceptions_to_dict(path)
