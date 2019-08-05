r"""Module for training a new model of the lemmatizer.

Basic usage:
python -m glemmazon.train_lemmatizer \
  --conllu data/en_ewt-ud-train.conllu \
  --model models/lemmatizer/en

Combine CoNLL-U and UniMorph data:
python -m glemmazon.train_lemmatizer \
  --conllu data/en_ewt-ud-train.conllu \
  --unimorph data/eng \
  --mapping data/tag_mapping.csv \
  --model models/lemmatizer/en
"""
import os
import logging
import sys

import pandas as pd
import pickle
import numpy as np
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
from tensorflow.keras.models import Sequential, Model

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
from glemmazon.lemmatizer import Lemmatizer

FLAGS = flags.FLAGS
flags.DEFINE_string("conllu", None, "Path to a CoNLL-U file.")
flags.DEFINE_string("unimorph", None, "Path to a UniMorph file.")
flags.DEFINE_string("mapping", None, "Path to a file containing tag "
                                     "mappings from UniMorph to UniversalDependencies.")
flags.DEFINE_string("model", None,
                    "Path to store the Pickle file with the model.")

flags.DEFINE_string("exceptions", None,
                    "Path to a CSV with lemma exceptions [columns: "
                    "'word', 'pos', 'lemma'].")
flags.DEFINE_string("cleanup_conllu", "dummy",
                    "Name of the clean-up function to be used. Use "
                    "'dummy' for no clean-up.")
flags.DEFINE_string("cleanup_unimorph", "dummy",
                    "Name of the clean-up function to be used. Use "
                    "'dummy' for no clean-up.")
flags.DEFINE_integer("min_count", 3,
                     "The minimum number of counts a lemma suffix need "
                     "to have for it to be included for training.")
flags.DEFINE_integer("max_features", 256,
                     "The maximum number of characters to be "
                     "considered in the vocabulary.")
flags.DEFINE_boolean("no_losses", True,
                     "If True, losses from training data will be added "
                     "to the model's exception dictionary (not to the "
                     ".csv file though).")
flags.DEFINE_integer("embedding_size", 16, "Embedding size.")
flags.DEFINE_integer("batch_size", 16, "Mini-batch size.")
flags.DEFINE_integer("maxlen", 10,
                     "The max length of the suffix to be extracted.")
flags.DEFINE_integer("epochs", 25, "Epochs for training.")

flags.mark_flag_as_required('model')

logger = logging.getLogger(__name__)


def _build_encoders(df):
    ch_list = {ch for word in df.word.apply(lambda x: list(x))
               for ch in word}
    sfe = SeqFeatureEncoder(
        seq_name='word',
        seq_encoder=SeqWordSuffix(ch_list, suffix_length=6),
        dense_encoders=DictFeatureEncoder({'pos': DenseTag(
            df.pos.unique())}))

    label_encoders = {
        'lemma_suffix': LabelEncoder(df.lemma_suffix.unique()),
        'lemma_index': LabelEncoder(df.lemma_index.unique()),
    }
    dle = DictLabelEncoder(label_encoders)

    return sfe, dle


# TODO(gustavoauma): Subclass layers.Model, like the cool kids.
def _build_model(input_shape, dle):
    inputs = Input(shape=input_shape)
    deep = Bidirectional(LSTM(64))(inputs)
    deep = Dropout(0.3)(deep)
    deep = Dense(64)(deep)
    out_suffix = Dense(dle.encoders['lemma_suffix'].output_shape[0],
                       activation='softmax', name='lemma_suffix')(deep)
    out_index = Dense(dle.encoders['lemma_index'].output_shape[0],
                      activation='softmax', name='lemma_index')(deep)
    return Model(inputs, [out_suffix, out_index])


def _add_losses_as_exceptions(l, df):
    for _, row in tqdm.tqdm(df.iterrows()):
        lemma_pred = l(row[k_.WORD_COL], row[k_.POS_COL])
        if lemma_pred != row[k_.LEMMA_COL]:
            logger.info(
                'Added exception: "%s" -> "%s" [pred: "%s"]' % (
                    row[k_.WORD_COL], row[k_.LEMMA_COL],
                    lemma_pred))
            l.exceptions[(row[k_.WORD_COL], row[k_.POS_COL])] = (
                row[k_.LEMMA_COL])


def main(_):
    if not FLAGS.conllu and not FLAGS.unimorph:
        sys.exit('At least one of the flags --conllu or --unimorph '
                 'need to be specified.')
    elif FLAGS.unimorph and not FLAGS.mapping:
        sys.exit('A mapping file for Unimorph tags need to be '
                 'defined with the flag --mapping.')

    df = pd.DataFrame()
    if FLAGS.conllu:
        print('Reading sentences from CoNLL-U...')
        df = df.append(preprocess.conllu_to_df(
            FLAGS.conllu, getattr(cleanup, FLAGS.cleanup_conllu),
            min_count=FLAGS.min_count), sort=True)
    if FLAGS.unimorph:
        print('Reading tokens from UniMorph...')
        df = df.append(preprocess.unimorph_to_df(
            FLAGS.unimorph, FLAGS.mapping,
            clean_up=getattr(cleanup, FLAGS.cleanup_unimorph)),
            lemmatizer_cols=True,
            sort=True)

    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df[k_.INDEX_COL] = df[k_.INDEX_COL].astype('str')

    # Keep only relevant columns
    df = df[[k_.WORD_COL, k_.LEMMA_COL, k_.POS_COL, k_.SUFFIX_COL,
             k_.INDEX_COL]]

    # Make a copy of the original DataFrame, without the aggregation, so
    # that exceptions are kept in the data.
    if FLAGS.no_losses:
        orig_df = pd.DataFrame(df)

    # Exclude inflection patterns that occur only once.
    df = df.groupby(k_.SUFFIX_COL).filter(
        lambda r: r[k_.SUFFIX_COL].count() > FLAGS.min_count)

    print('Data sample:')
    print(df.head())

    print('Splitting between training, test and val...')
    train, test = train_test_split(df, test_size=0.2)
    print(len(train), 'train examples')
    print(len(test), 'test examples')

    print('Preparing training data and feature/label encoders...')
    sfe, dle = _build_encoders(df)

    print('Preparing test data...')
    x_test = np.stack([sfe(dict(r)) for _, r in
                       test[['word', 'pos']].iterrows()])
    y_test = [np.vstack(e) for e in
              zip(*[dle(dict(r)) for _, r in
                    test[['lemma_suffix', 'lemma_index']].iterrows()])]
    print(x_test.shape, y_test[0].shape, y_test[1].shape)

    print('Preparing batch generators...')
    batch_generator = utils.BatchGenerator(df, sfe, dle)

    print('Building the model...')
    model = _build_model(sfe.output_shape, dle)
    model.summary()

    print('Running training...')
    model.compile('adam', 'categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit_generator(batch_generator, epochs=FLAGS.epochs, verbose=2)

    exceptions = {}
    if FLAGS.exceptions:
        print('Loading exceptions...')
        exceptions = preprocess.exceptions_to_dict(FLAGS.exceptions)

    print('Persisting the model...')
    if not os.path.exists(FLAGS.model):
        os.mkdir(FLAGS.model)
    model.save(os.path.join(FLAGS.model, k_.MODEL_FILE))
    with open(os.path.join(FLAGS.model, k_.PARAMS_FILE),
              'wb') as writer:
        pickle.dump({
            'exceptions': exceptions,
            'feature_enc': sfe,
            'label_enc': dle,
        }, writer)

    # Add losses to the exception dictionary, so that they can be
    # labeled correctly, if specified by the caller.
    if FLAGS.no_losses:
        print('Adding losses to the dictionary with exceptions...')
        l = Lemmatizer()
        l.load(FLAGS.model)
        n_start = len(l.exceptions)
        # noinspection PyUnboundLocalVariable
        _add_losses_as_exceptions(l, orig_df)
        print('# Exceptions added: %d' % (len(l.exceptions) - n_start))
        l.save(FLAGS.model)

    print('Model successfully saved in folder: %s.' % FLAGS.model)


if __name__ == '__main__':
    app.run(main)
