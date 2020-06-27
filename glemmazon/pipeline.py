"""Module containing pipelines."""

__all__ = [
    'Analyzer',
    'BasePipeline',
    'Inflector',
    'LookupDictionary',
    'Lemmatizer',
    'Result',
    'Source',
]

import os
import pickle
import re
from abc import abstractmethod, ABC
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model, Model

from glemmazon import constants as k
from glemmazon import utils
from glemmazon.encoder import DictFeatureEncoder, DictLabelEncoder


class Source(Enum):
    DICTIONARY = 1
    MODEL = 2


class Result(object):
    def __init__(self, data: Dict[str, str], source: Source):
        self.data = data
        self.source = source

    def __eq__(self, other):
        return self.data == other.data and self.source == other.source

    def __repr__(self):
        return str(self.__dict__)

    def __str__(self):
        return str(locals())

    def to_dict(self):
        return self.__dict__


class LookupDictionary(object):
    """Class to represent a lookup dictionary."""

    def __init__(self, df=None, columns=None):
        if df is None:
            if columns is None:
                raise ValueError('Argument "columns" must be specified '
                                 'when there is no input DataFrame.')
            df = pd.DataFrame(columns=columns)
        self.df = df

    def __bool__(self):
        return not self.df.empty

    def __len__(self):
        return len(self.df)

    @classmethod
    def from_csv(cls, path: str):
        df = pd.read_csv(path)
        return LookupDictionary(df)

    def to_csv(self, path: str):
        self.df.to_csv(path, index=False)

    def lookup(self, keep_cols=None, omit_cols=None, **kwargs
               ) -> List[Dict[str, str]]:
        """Lookup key in the exceptions dictionary.

        Raises:
            KeyError: if the entry is not found.
            ValueError: if the argument passed is not a column in the
                DataFrame.

        Returns:
            List of Python dictionaries with keys and values, e.g.
            [
              {'word': 'amar', 'pos': 'VERB'},
              {'word': 'carro', 'pos': 'NOUN'}
            ]
        """
        q = self._build_query(**kwargs)

        try:
            result_df = self.df.query(q)
        except pd.core.computation.ops.UndefinedVariableError as e:
            raise ValueError(e)
        if result_df.empty:
            raise KeyError(
                f'Could not find entry with attributes: {kwargs}')

        # Filters
        if omit_cols:
            result_df = result_df.drop(omit_cols, axis=1)
        if keep_cols:
            result_df = result_df[keep_cols]
        return result_df.to_dict('records')

    def add_entry(self, **kwargs):
        try:
            self.df.loc[len(self.df)] = kwargs
        except ValueError as e:
            raise ValueError(
                f'Cannot add entry. {kwargs}'
                f' Expected columns: {sorted(list(self.df.columns))}.')

    def _build_query(self, **kwargs):
        return ' and '.join([
            "%s == '%s'" % (key, re.escape(val.replace("'", '')))
            for key, val in kwargs.items()])


class BasePipeline(ABC):
    """Abstract class to represent pipeline with model + exceptions."""

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
        self.exceptions = exceptions

        # Make a fake first call to predict to avoid latency issues.
        if first_call:
            self.predict(**self._dummy_example)

    def __call__(self, **kwargs) -> Result:
        if self.exceptions:
            try:
                return Result(data=self.lookup(**kwargs),
                              source=Source.DICTIONARY)
            except KeyError:
                pass
        return Result(data=self.predict(**kwargs), source=Source.MODEL)

    @classmethod
    def load(cls, path: str):
        """Load the pipeline from a directory."""
        exceptions = LookupDictionary.from_csv(
            os.path.join(path, k.EXCEPTIONS_FILE))
        model = load_model(os.path.join(path, k.MODEL_FILE))
        with open(os.path.join(path, k.PARAMS_FILE), 'rb') as reader:
            return cls(**{
                **{'model': model},
                **pickle.load(reader),
                **{'exceptions': exceptions}
            })

    def save(self, path: str):
        """Save the pipeline to a directory."""
        if not os.path.exists(path):
            os.mkdir(path)

        self.model.save(os.path.join(path, k.MODEL_FILE))
        self.exceptions.to_csv(os.path.join(path, k.EXCEPTIONS_FILE))
        with open(os.path.join(path, k.PARAMS_FILE), 'wb') as writer:
            pickle.dump({
                'feature_enc': self.feature_enc,
                'label_enc': self.label_enc,
            }, writer)

    @property
    @abstractmethod
    def _dummy_example(self):
        """Dummy example to be used in the first call of the model.

        Note: the first iteration of predict() in tensorflow s slower
        due to caching: https://stackoverflow.com/questions/55577711.
        """

    def lookup(self, **kwargs) -> Dict[str, str]:
        """Lookup a single key and value."""
        return self.exceptions.lookup(**kwargs)[0]  # Return first.

    @property
    @abstractmethod
    def predict(self, *args, **kwargs) -> Dict[str, str]:
        """Annotate a single example (using the model)."""


class Lemmatizer(BasePipeline):
    """Class to represent a lemmatizer."""

    def get_lemma(self, **kwargs):
        try:
            return self.__call__(**kwargs).data[k.LEMMA_COL]
        except KeyError as e:
            raise KeyError(
                f'Couldn\'t get lemma for kwargs: {kwargs}.', e)

    def lookup(self, **kwargs) -> Dict[str, str]:
        return super().lookup(keep_cols=[k.LEMMA_COL], **kwargs)

    def predict(self, word: str, pos: str) -> Dict[str, str]:
        """Annotate a single example (using the model)."""
        return {k.LEMMA_COL: utils.apply_suffix_op(
            word, self._predict_op(word, pos))}

    @property
    def _dummy_example(self) -> Dict[str, str]:
        return {'word': '', 'pos': k.UNKNOWN_TAG}

    def _predict_op(self, word: str, pos: str) -> Tuple[int, str]:
        """Return the string operation for the lemma as (index, str)."""
        features = [self.feature_enc({k.WORD_COL: word,
                                      k.POS_COL: pos})]
        y_pred_dict = self.label_enc.decode(self.model.predict(
            np.array(features)))
        return int(y_pred_dict[k.INDEX_COL]), y_pred_dict[k.SUFFIX_COL]


class Analyzer(BasePipeline):
    """Class to represent an analyser."""

    def lookup(self, **kwargs) -> Dict[str, str]:
        return super().lookup(omit_cols=[k.WORD_COL], **kwargs)

    def predict(self, word: str, pos: str) -> Dict[str, str]:
        """Annotate a single example (using the model)."""
        features = [self.feature_enc({k.WORD_COL: word,
                                      k.POS_COL: pos})]
        y_pred = self.model.predict(np.array(features))
        y_pred_dict = self.label_enc.decode(y_pred)
        return y_pred_dict

    @property
    def _dummy_example(self) -> Dict[str, str]:
        return {'word': '', 'pos': k.UNKNOWN_TAG}


class Inflector(BasePipeline):
    """Class to represent an inflector."""

    def get_word(self, **kwargs):
        try:
            return self.__call__(**kwargs).data[k.WORD_COL]
        except KeyError as e:
            raise KeyError(
                f'Couldn\'t get word for kwargs: {kwargs}.', e)

    def lookup(self, **kwargs) -> Dict[str, str]:
        return super().lookup(keep_cols=[k.WORD_COL], **kwargs)

    def predict(self, lemma: str, **kwargs: str) -> Dict[str, str]:
        """Annotate a single example (using the model)."""
        return {k.WORD_COL: utils.apply_suffix_op(
            lemma, self._predict_op(lemma, **kwargs))}

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
                    kwargs[feature] = k.UNSPECIFIED_TAG

        features = [self.feature_enc({k.LEMMA_COL: lemma, **kwargs})]
        y_pred_dict = self.label_enc.decode(self.model.predict(
            np.array(features)))
        return (int(y_pred_dict[k.WORD_INDEX_COL]),
                y_pred_dict[k.WORD_SUFFIX_COL])

    @property
    def _dummy_example(self) -> Dict[str, str]:
        return {
            **{'lemma': 'teste', 'pos': 'NOUN'},
            **{key: k.UNSPECIFIED_TAG for key in self.feature_enc.scope
               if key not in ('lemma', 'pos')}
        }
