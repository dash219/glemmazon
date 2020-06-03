# -*- coding: utf-8 -*-
r"""Module for training a new model of the inflector.

Basic usage:
python -m glemmazon.train_inflector \
  --conllu data/en_ewt-ud-train.conllu \
  --model models/inflector/en
"""

import logging
import os
import pickle

import pandas as pd
import tqdm
from absl import app
from absl import flags
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Input,
    LSTM,
    Bidirectional)
from tensorflow.keras.models import Model

from glemmazon import cleanup
from glemmazon import constants as k_
from glemmazon import preprocess
from glemmazon import utils
from glemmazon.encoder import (
    DenseTag,
    DictFeatureEncoder,
    DictLabelEncoder,
    LabelEncoder,
    SeqFeatureEncoder,
    SeqWordSuffix)
from glemmazon.pipeline import LookupDictionary, Inflector

FLAGS = flags.FLAGS
flags.DEFINE_string("conllu", None, "Path to a CoNLL-U file.")
flags.DEFINE_string("model", None,
                    "Path to store the Pickle file with the model.")
flags.DEFINE_string("exceptions", None,
                    "Path to a CSV with lemma exceptions [columns: "
                    "'word', 'pos', 'lemma'].")
flags.DEFINE_string("cleanup_conllu", "basic",
                    "Name of the clean-up function to be used. Use "
                    "'dummy' for no clean-up.")
flags.DEFINE_integer("min_count", 3,
                     "The minimum number of counts a lemma suffix need "
                     "to have for it to be included for training.")
flags.DEFINE_integer("max_features", 256,
                     "The maximum number of characters to be "
                     "considered in the vocabulary.")
flags.DEFINE_boolean("no_losses", False,
                     "If True, losses from training data will be added "
                     "to the model's exception dictionary (not to the "
                     ".csv file though).")
flags.DEFINE_integer("embedding_size", 16, "Embedding size.")
flags.DEFINE_integer("batch_size", 16, "Mini-batch size.")
flags.DEFINE_integer("maxlen", 10,
                     "The max length of the suffix to be extracted.")
flags.DEFINE_integer("epochs", 25, "Epochs for training.")

flags.mark_flag_as_required('model')
flags.mark_flag_as_required('conllu')

logger = logging.getLogger(__name__)


def _build_encoders(df, dense_cols):
    ch_list = {ch for lemma in df[k_.LEMMA_COL].apply(lambda x: list(x))
               for ch in lemma}
    sfe = SeqFeatureEncoder(
        seq_name=k_.LEMMA_COL,
        seq_encoder=SeqWordSuffix(ch_list, suffix_length=6),
        dense_encoders=DictFeatureEncoder({
            col: DenseTag(df[col].unique()) for col in dense_cols}))

    label_encoders = {
        k_.WORD_SUFFIX_COL: LabelEncoder(
            df[k_.WORD_SUFFIX_COL].unique()),
        k_.WORD_INDEX_COL: LabelEncoder(df[k_.WORD_INDEX_COL].unique()),
    }
    dle = DictLabelEncoder(label_encoders)

    return sfe, dle


# TODO(gustavoauma): Subclass layers.Model, like the cool kids.
def _build_model(input_shape, dle):
    inputs = Input(shape=input_shape)
    deep = Bidirectional(LSTM(64))(inputs)
    deep = Dropout(0.3)(deep)
    deep = Dense(64)(deep)
    out_index = Dense(
        dle.encoders[k_.WORD_INDEX_COL].output_shape[0],
        activation='softmax', name=k_.WORD_INDEX_COL)(deep)
    out_suffix = Dense(
        dle.encoders[k_.WORD_SUFFIX_COL].output_shape[0],
        activation='softmax', name=k_.WORD_SUFFIX_COL)(deep)
    return Model(inputs, [out_index, out_suffix])


def _get_losses_df(inflec, df):
    exceptions = []
    for _, row in tqdm.tqdm(df.iterrows()):
        kwargs = dict([(key, val) for key, val in row.items()
                       if key in inflec.feature_to_ix])
        word_pred = inflec(row[k_.LEMMA_COL], **kwargs)
        if word_pred[1] != row[k_.WORD_COL]:
            logger.info(
                'Added exception: "%s" -> "%s" [pred: "%s"]' % (
                    row[k_.LEMMA_COL], row[k_.WORD_COL], word_pred[1]))
            exceptions.append({**{k_.WORD_COL: row[k_.WORD_COL],
                                  k_.LEMMA_COL: row[k_.LEMMA_COL]},
                               **kwargs})
    return pd.DataFrame(exceptions)


# noinspection PyPep8Naming
def main(_):
    logger.info('Reading sentences from CoNLL-U...')
    df = preprocess.conllu_to_df(
        FLAGS.conllu, getattr(cleanup, FLAGS.cleanup_conllu),
        min_count=FLAGS.min_count, inflector_info=True)

    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df[k_.WORD_INDEX_COL] = df[k_.WORD_INDEX_COL].astype('str')

    # Make a copy of the original DataFrame, without the aggregation, so
    # that exceptions are kept in the data.
    if FLAGS.no_losses:
        orig_df = pd.DataFrame(df)

    # Exclude inflection patterns that occur only once.
    df = df.groupby(k_.WORD_SUFFIX_COL).filter(
        lambda r: r[k_.WORD_SUFFIX_COL].count() > FLAGS.min_count)

    logger.info('Data sample:')
    logger.info(df.head())

    logger.info('Splitting between training, test and val...')
    train, test = train_test_split(df, test_size=0.2)
    logger.info('# Training examples: %d' % len(train))
    logger.info('# Test examples: %d' % len(test))

    logger.info('Preparing training data and feature/label encoders...')
    dense_cols = [col for col in df.columns if col not in [
        k_.WORD_SUFFIX_COL, k_.WORD_INDEX_COL, k_.WORD_COL,
        k_.LEMMA_COL]]
    sfe, dle = _build_encoders(df, dense_cols)

    logger.info('Preparing batch generators...')
    batch_generator = utils.BatchGenerator(df, sfe, dle)

    logger.info('Building the model...')
    model = _build_model(sfe.output_shape, dle)
    model.summary()

    logger.info('Running training...')
    model.compile('adam', 'categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit_generator(batch_generator, epochs=FLAGS.epochs, verbose=2)

    if FLAGS.exceptions:
        logger.info('Loading exceptions...')
        exceptions = LookupDictionary.from_csv(FLAGS.exceptions)
    else:
        exceptions = LookupDictionary(columns=list(sfe.scope |
                                                   dle.scope))

    logger.info('Persisting the model...')
    inflector = Inflector(model=model, feature_enc=sfe, label_enc=dle,
                          exceptions=exceptions)
    inflector.save(FLAGS.model)

    # Add losses to the exception dictionary, so that they can be
    # labeled correctly, if specified by the caller.
    if FLAGS.no_losses:
        raise NotImplementedError

    logger.info('Model successfully saved in: %s.' % FLAGS.model)


if __name__ == '__main__':
    app.run(main)
