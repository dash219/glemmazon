"""Module containing pipelines."""

__all__ = ['Lemmatizer']

import os
import pickle
from abc import abstractmethod, ABC
from typing import Dict, Tuple

import numpy as np
from tensorflow.keras.models import load_model, Model

from glemmazon import constants as k
from glemmazon import preprocess
from glemmazon import utils
from glemmazon.encoder import DictFeatureEncoder, DictLabelEncoder


class BasePipeline(ABC):
    """Abstract class to represent pipeline with model + exceptions."""

    def __init__(self,
                 model: Model,
                 feature_enc: DictFeatureEncoder,
                 label_enc: DictLabelEncoder,
                 exceptions: Dict[str, str] = None,
                 first_call: bool = True):
        """Initialize the class.

        Args:
            model (Model): The model to be loaded.
            feature_enc (DictFeatureEncoder): Feature encoder.
            label_enc (DictLabelEncoder): Label encoder.
            exceptions (dict): Exceptions dictionary.
            first_call (bool): If True, make a first call to the model
                when it is loaded, to avoid latency issues. The first
                iteration of predict() is slower due to caching:
                https://stackoverflow.com/questions/55577711
        """
        self.model = model
        self.feature_enc = feature_enc
        self.label_enc = label_enc
        self.exceptions = exceptions or dict()

        # Make a fake first call to predict to avoid latency issues.
        if first_call:
            self.annotate(**self.dummy_example)

    def __call__(self, *args) -> str:
        try:
            return self.exceptions[args]
        except KeyError:
            return self.annotate(*args)

    @classmethod
    def load(cls, path: str):
        """Load the pipeline from a directory."""
        with open(os.path.join(path, k.PARAMS_FILE),
                  'rb') as reader:
            return cls(**{**{'model': load_model(
                os.path.join(path, k.MODEL_FILE))},
                          **pickle.load(reader)})

    def save(self, path: str):
        """Save the pipeline to a directory."""
        if not os.path.exists(path):
            os.mkdir(path)

        self.model.save(os.path.join(path, k.MODEL_FILE))
        with open(os.path.join(path, k.PARAMS_FILE), 'wb') as writer:
            pickle.dump({
                'exceptions': self.exceptions,
                'feature_enc': self.feature_enc,
                'label_enc': self.label_enc,
            }, writer)

    @property
    @abstractmethod
    def dummy_example(self):
        """Dummy example to be used in the first call of the model.

        Note: the first iteration of predict() in tensorflow s slower
        due to caching: https://stackoverflow.com/questions/55577711.
        """

    @property
    @abstractmethod
    def annotate(self, *args, **kwargs):
        """Annotate a single example (using the model)."""


class Lemmatizer(BasePipeline):
    """Class to represent a lemmatizer."""

    @property
    def dummy_example(self) -> Dict[str, str]:
        return {'word': '', 'pos': k.UNKNOWN_TAG}

    def annotate(self, word: str, pos: str):
        """Annotate a single example (using the model)."""
        return utils.apply_suffix_op(word, self._predict_op(word, pos))

    def load_exceptions(self, path: str):
        """Load exceptions from a .csv file with "word, pos, lemma"."""
        self.exceptions = preprocess.exceptions_to_dict(path)

    def _predict_op(self, word: str, pos: str) -> Tuple[int, str]:
        """Return the string operation for the lemma as (index, str)."""
        features = [self.feature_enc({k.WORD_COL: word,
                                      k.POS_COL: pos})]
        y_pred_dict = self.label_enc.decode(self.model.predict(
            np.array(features)))
        return int(y_pred_dict[k.INDEX_COL]), y_pred_dict[k.SUFFIX_COL]
