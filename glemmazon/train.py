r"""Module for training a new model of the lemmatizer.

Basic usage:
python glemmazon/train.py \
  --conllu data/en_ewt-ud-train.conllu \
  --model models/en.pkl

Include a dictionary with exceptions:
python glemmazon/train.py \
  --conllu data/en_ewt-ud-train.conllu \
  --model models/en.pkl \
  --exceptions data/en_exceptions.csv
"""

__all__ = ['Lemmatizer']

import sys

import numpy as np
import pandas as pd
import pickle

from absl import app
from absl import flags
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

from glemmazon import cleanup
from glemmazon import constants as k
from glemmazon import preprocess
from glemmazon.lemmatizer import Lemmatizer

FLAGS = flags.FLAGS
flags.DEFINE_string("conllu", None, "Path to a CoNLL-U file.")
flags.DEFINE_string("unimorph", None, "Path to a UniMorph file.")
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
flags.DEFINE_integer("embedding_size", 16, "Embedding size.")
flags.DEFINE_integer("batch_size", 16, "Mini-batch size.")
flags.DEFINE_integer("maxlen", 10,
                     "The max length of the suffix to be extracted.")
flags.DEFINE_integer("epochs", 25, "Epochs for training.")

flags.mark_flag_as_required('model')


def _build_model(max_features, embedding_size, maxlen, labels):
    model = Sequential()
    model.add(Embedding(max_features, embedding_size,
                        input_length=maxlen))
    model.add(Bidirectional(LSTM(16)))
    model.add(Dropout(0.3))
    model.add(Dense(len(labels), activation='softmax'))
    return model


def _build_index_dict(iterable):
    index_dict = {}
    for e in iterable:
        if e not in index_dict:
            index_dict[e] = len(index_dict)
    return index_dict


def _encode_labels(labels, labels_dict):
    return to_categorical([labels_dict[l] for l in labels],
                          len(labels_dict))


def _extract_features(words, pos_list, pos_to_ix, tokenizer,
                      maxlen=10):
    words_feats = pad_sequences(
        tokenizer.texts_to_sequences(words), maxlen)
    pos_feats = _encode_labels(pos_list, pos_to_ix)
    return np.concatenate([words_feats, pos_feats], axis=1)


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
            min_count=FLAGS.min_count), sort=False)

    if FLAGS.unimorph:
        print('Reading tokens from UniMorph...')
        df = df.append(preprocess.unimorph_to_df(
            FLAGS.unimorph, getattr(cleanup, FLAGS.cleanup_unimorph),
            min_count=FLAGS.min_count), sort=False)

    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    print('Data sample:')
    print(df.head())

    print('Preparing lists of tags/labels...')
    suffix_to_ix = _build_index_dict(df[k.SUFFIX_COL].unique())
    index_to_ix = _build_index_dict(df[k.INDEX_COL].unique())
    pos_to_ix = _build_index_dict(df.pos.unique())

    train, test = train_test_split(df, test_size=0.2)

    print('Converting chars to integers...')
    tokenizer = Tokenizer(num_words=FLAGS.max_features, char_level=True)
    tokenizer.fit_on_texts(df.word)

    print('Padding the sequences...')
    x_train = _extract_features(train.word, train.pos, pos_to_ix,
                                tokenizer, FLAGS.maxlen)
    x_test = _extract_features(test.word, test.pos, pos_to_ix,
                               tokenizer, FLAGS.maxlen)

    print('Converting the data to one-hot...')
    ys_train = _encode_labels(train[k.SUFFIX_COL], suffix_to_ix)
    ys_test = _encode_labels(test[k.SUFFIX_COL], suffix_to_ix)

    yi_train = _encode_labels(train[k.INDEX_COL], index_to_ix)
    yi_test = _encode_labels(test[k.INDEX_COL], index_to_ix)

    print('Building the suffix model...')
    suffix_model = _build_model(FLAGS.max_features,
                                FLAGS.embedding_size,
                                x_train.shape[1],
                                suffix_to_ix)
    suffix_model.compile('adam', 'categorical_crossentropy',
                         metrics=['accuracy'])
    print(suffix_model.summary())

    suffix_model.fit(x_train, ys_train,
                     batch_size=FLAGS.batch_size,
                     epochs=FLAGS.epochs,
                     validation_data=[x_test, ys_test])

    print('Building the index model...')
    index_model = _build_model(FLAGS.max_features,
                               FLAGS.embedding_size,
                               x_train.shape[1],
                               index_to_ix)
    index_model.compile('adam', 'categorical_crossentropy',
                        metrics=['accuracy'])
    print(suffix_model.summary())

    index_model.fit(x_train, yi_train,
                    batch_size=FLAGS.batch_size,
                    epochs=FLAGS.epochs,
                    validation_data=[x_test, yi_test])

    print('Persisting the model...')
    pickle.dump({
        'index_model': index_model,
        'suffix_model': suffix_model,
        'suffix_to_ix': suffix_to_ix,
        'index_to_ix': index_to_ix,
        'pos_to_ix': pos_to_ix,
        'tokenizer': tokenizer,
        'maxlen': FLAGS.maxlen,
    }, open(FLAGS.model, 'wb'))
    print('Model successfully saved in: %s.' % FLAGS.model)


if __name__ == '__main__':
    app.run(main)
