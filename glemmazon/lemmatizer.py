"""Main module for the lemmatizer."""

__all__ = ['Lemmatizer']

from typing import Dict, Tuple

import os

import numpy as np
import pickle

from tensorflow.keras.models import load_model, Model

from glemmazon import constants as k
from glemmazon import preprocess
from glemmazon import utils
from glemmazon.encoder import DictFeatureEncoder, DictLabelEncoder


class Lemmatizer(object):
    """Class to represent a lemmatizer."""

    def __init__(self,
                 model: Model,
                 feature_enc: DictFeatureEncoder,
                 label_enc: DictLabelEncoder,
                 exceptions: Dict[Tuple[str, str], str] = None):
        self.model = model
        self.feature_enc = feature_enc
        self.label_enc = label_enc
        self.exceptions = exceptions or dict()

        # Make a fake first call to predict: the first iteration of
        # predict() is slower due to caching:
        # https://stackoverflow.com/questions/55577711
        self._predict_op('', k.UNKNOWN_TAG)

    def __call__(self, word: str, pos: str) -> str:
        try:
            return self.exceptions[(word, pos)]
        except KeyError:
            return utils.apply_suffix_op(word, self._predict_op(
                word, pos))

    def predict(self, instances, **kwargs):
        labels = []
        for instance in instances:
            labels.append(self.__call__(**instance))
        return labels

    @classmethod
    def from_path(cls, model_dir: str):
        """Load the model from a model_dir."""
        with open(os.path.join(model_dir, k.PARAMS_FILE),
                  'rb') as reader:
            return cls(**{**{'model': load_model(
                os.path.join(model_dir, k.MODEL_FILE))},
                          **pickle.load(reader)})
    @classmethod
    def load(cls, model_dir: str):
        """Load the model from a model_dir."""
        return cls.from_path(model_dir)

    def load_exceptions(self, path: str):
        """Load exceptions from a .csv file with "word, pos, lemma"."""
        self.exceptions = preprocess.exceptions_to_dict(path)

    def save(self, model_dir: str):
        """Save the model to a model_dir."""
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        self.model.save(os.path.join(model_dir, k.MODEL_FILE))
        with open(os.path.join(model_dir, k.PARAMS_FILE), 'wb') as writer:
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
        self.exceptions = exceptions or dict()

    def _predict_op(self, word: str, pos: str) -> Tuple[int, str]:
        features = [self.feature_enc({k.WORD_COL: word,
                                      k.POS_COL: pos})]
        y_pred_dict = self.label_enc.decode(self.model.predict(
            np.array(features)))
        return int(y_pred_dict[k.INDEX_COL]), y_pred_dict[k.SUFFIX_COL]
