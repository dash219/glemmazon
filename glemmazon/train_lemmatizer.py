r"""Module for training a new model of the lemmatizer.

Basic usage:
python -m glemmazon.train_lemmatizer \
  --conllu data/en_ewt-ud-train.conllu \
  --model models/en.pkl

Include a dictionary with exceptions:
python -m glemmazon.train_lemmatizer \
  --conllu data/en_ewt-ud-train.conllu \
  --model models/en.pkl \
  --exceptions data/en_exceptions.csv
"""
from typing import Dict

import logging
import sys

import numpy as np
import pandas as pd
import pickle
import tqdm

from absl import app
from absl import flags
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from tensorflow.python.keras.utils import Sequence

from glemmazon import cleanup
from glemmazon import constants as k
from glemmazon import preprocess
from glemmazon import utils
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


def _build_model(input_shape, labels):
    model = Sequential()
    model.add(Bidirectional(LSTM(16), input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(len(labels), activation='softmax'))

    model.compile('adam', 'categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


# noinspection PyPep8Naming
class batch_generator(Sequence):
    def __init__(self, df, pos_to_ix, label_to_ix, tokenizer,
                 col_name, batch_size):
        self.df = df.sample(frac=1).reset_index(drop=True)
        self.pos_to_ix = pos_to_ix
        self.label_to_ix = label_to_ix
        self.tokenizer = tokenizer
        self.col_name = col_name
        self.batch_size = batch_size
        self.n = 0
        self.max = self.__len__()

    def __len__(self):
        return int(np.ceil(len(self.df) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_df = self.df[
                   idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = []
        batch_y = []
        for _, row in batch_df.iterrows():
            batch_x.append(_extract_features(row[k.WORD_COL],
                                             row[k.POS_COL],
                                             self.pos_to_ix,
                                             self.tokenizer))
            batch_y.append(utils.encode_labels([row[self.col_name]],
                                               self.label_to_ix))
        return np.concatenate(batch_x), np.concatenate(batch_y)

    def __next__(self):
        if self.n >= self.max:
            self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result


def _extract_features(word: str,
                      pos: str,
                      pos_to_ix: Dict[str, str],
                      tokenizer: Tokenizer) -> np.array:
    """Extract features from a list of words and pos."""
    # Convert the morphological tags into vectors.
    tag_vec = utils.encode_labels([pos], pos_to_ix)[0]
    tag_vecs = np.array([np.repeat([tag_vec], FLAGS.maxlen, axis=0)])

    # Extract character vectors.
    char_vecs = pad_sequences(
        [to_categorical(tokenizer.texts_to_sequences(word),
                        len(tokenizer.word_index) + 1)], FLAGS.maxlen)
    return np.concatenate([char_vecs, tag_vecs], axis=2)


def _extract_features_df(df, pos_to_ix, tokenizer):
    features = []
    for _, row in tqdm.tqdm(df.iterrows()):
        features.append(_extract_features(
            row[k.WORD_COL], row[k.POS_COL], pos_to_ix, tokenizer))
    return np.concatenate(features)


def _add_losses_as_exceptions(l, df):
    for _, row in tqdm.tqdm(df.iterrows()):
        lemma_pred = l(row[k.WORD_COL], row[k.POS_COL])
        if lemma_pred != row[k.LEMMA_COL]:
            logger.info(
                'Added exception: "%s" -> "%s" [pred: "%s"]' % (
                    row[k.WORD_COL], row[k.LEMMA_COL],
                    lemma_pred))
            l.exceptions[(row[k.WORD_COL], row[k.POS_COL])] = (
                row[k.LEMMA_COL])


# noinspection PyPep8Naming
def main(_):
    if not FLAGS.conllu and not FLAGS.unimorph:
        sys.exit('At least one of the flags --conllu or --unimorph '
                 'need to be specified.')

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
            sort=True)

    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Make a copy of the original DataFrame, without the aggregation, so
    # that exceptions are kept in the data.
    if FLAGS.no_losses:
        orig_df = pd.DataFrame(df)

    # Exclude inflection patterns that occur only once.
    df = df.groupby(k.SUFFIX_COL).filter(
        lambda r: r[k.SUFFIX_COL].count() > FLAGS.min_count)

    print('Data sample:')
    print(df.head())

    print('Preparing lists of tags/labels...')
    suffix_to_ix = utils.build_index_dict(df[k.SUFFIX_COL].unique())
    index_to_ix = utils.build_index_dict(df[k.INDEX_COL].unique())
    pos_to_ix = utils.build_index_dict(df.pos.unique())

    print('Splitting between training and test...')
    train, test = train_test_split(df, test_size=0.2)

    print('Converting chars to integers...')
    tokenizer = Tokenizer(num_words=FLAGS.max_features, char_level=True)
    tokenizer.fit_on_texts(df.word)

    print('Preparing validation data...')
    x_val = _extract_features_df(test, pos_to_ix, tokenizer)
    ys_val = utils.encode_labels(test[k.SUFFIX_COL], suffix_to_ix)
    yi_val = utils.encode_labels(test[k.INDEX_COL], index_to_ix)
    print(x_val.shape, ys_val.shape, yi_val.shape)

    print('Preparing batch generators...')
    ys_train_gen = batch_generator(df, pos_to_ix, suffix_to_ix,
                                   tokenizer, k.SUFFIX_COL,
                                   FLAGS.batch_size)
    yi_train_gen = batch_generator(df, pos_to_ix, index_to_ix,
                                   tokenizer, k.INDEX_COL,
                                   FLAGS.batch_size)

    print('Building suffix model...')
    sample_x = _extract_features(test.iloc[0][k.WORD_COL],
                                 test.iloc[0][k.POS_COL],
                                 pos_to_ix, tokenizer)
    input_shape = (FLAGS.maxlen, sample_x.shape[2])

    suffix_model = _build_model(input_shape, suffix_to_ix)
    suffix_model.fit_generator(ys_train_gen,
                               epochs=FLAGS.epochs,
                               steps_per_epoch=len(
                                   df) // FLAGS.batch_size,
                               verbose=2,
                               validation_data=[x_val, ys_val])

    print('Building index model...')
    index_model = _build_model(input_shape, index_to_ix)
    index_model.fit_generator(yi_train_gen,
                              epochs=FLAGS.epochs,
                              steps_per_epoch=len(
                                  df) // FLAGS.batch_size,
                              verbose=2,
                              validation_data=[x_val, yi_val])

    exceptions = {}
    if FLAGS.exceptions:
        print('Loading exceptions...')
        exceptions = preprocess.exceptions_to_dict(FLAGS.exceptions)

    print('Persisting the model...')
    model = {
        'index_model': index_model,
        'suffix_model': suffix_model,
        'suffix_to_ix': suffix_to_ix,
        'index_to_ix': index_to_ix,
        'pos_to_ix': pos_to_ix,
        'tokenizer': tokenizer,
        'maxlen': FLAGS.maxlen,
        'exceptions': exceptions,
    }

    # Add losses to the exception dictionary, so that they can be
    # labeled correctly, if specified by the caller.
    if FLAGS.no_losses:
        print('Adding losses to the dictionary with exceptions...')
        l = Lemmatizer()
        l.set_model(**model)
        n_start = len(l.exceptions)
        # noinspection PyUnboundLocalVariable
        _add_losses_as_exceptions(l, orig_df)
        print('# Exceptions added: %d' % (len(l.exceptions) - n_start))
        model['exceptions'] = l.exceptions

    pickle.dump(model, open(FLAGS.model, 'wb'))
    print('Model successfully saved in: %s.' % FLAGS.model)


if __name__ == '__main__':
    app.run(main)
