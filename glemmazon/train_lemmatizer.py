r"""Module for training a new model of the lemmatizer.

Basic usage:
python -m glemmazon.train_lemmatizer \
  --conllu data/en_ewt-ud-train.conllu \
  --model models/lemmatizer/en
"""

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
from glemmazon.pipeline import Lemmatizer, LookupDictionary

FLAGS = flags.FLAGS
flags.DEFINE_string("conllu", None, "Path to a CoNLL-U file.")
flags.DEFINE_string("model", None,
                    "Path to store the Pickle file with the model.")

flags.DEFINE_string("exceptions", None,
                    "Path to a CSV with lemma exceptions [columns: "
                    "'word', 'pos', 'lemma'].")
flags.DEFINE_string("cleanup_function", "basic",
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


def _build_encoders(df):
    ch_list = {ch for word in df.word.apply(lambda x: list(x))
               for ch in word}
    sfe = SeqFeatureEncoder(
        seq_name='word',
        seq_encoder=SeqWordSuffix(ch_list, suffix_length=6),
        dense_encoders=DictFeatureEncoder({'pos': DenseTag(
            df.pos.unique())}))

    label_encoders = {
        'lemma_index': LabelEncoder(df.lemma_index.unique()),
        'lemma_suffix': LabelEncoder(df.lemma_suffix.unique()),
    }
    dle = DictLabelEncoder(label_encoders)

    return sfe, dle


# TODO(gustavoauma): Subclass layers.Model, like the cool kids.
def _build_model(input_shape, dle):
    inputs = Input(shape=input_shape)
    deep = Bidirectional(LSTM(32))(inputs)
    deep = Dropout(0.3)(deep)
    deep = Dense(64)(deep)
    out_index = Dense(dle.encoders['lemma_index'].output_shape[0],
                      activation='softmax', name='lemma_index')(deep)
    out_suffix = Dense(dle.encoders['lemma_suffix'].output_shape[0],
                       activation='softmax', name='lemma_suffix')(deep)
    return Model(inputs, [out_index, out_suffix])


def _add_losses_as_exceptions(l, df, logger):
    for _, row in tqdm.tqdm(df.iterrows(), initial=1):
        lemma_pred = l(word=row[k_.WORD_COL], pos=row[k_.POS_COL])
        if lemma_pred != row[k_.LEMMA_COL]:
            logger.info(
                'Added exception: "%s" -> "%s" [pred: "%s"]' % (
                    row[k_.WORD_COL], row[k_.LEMMA_COL],
                    lemma_pred))
            l.exceptions.add_entry(
                word=k_.WORD_COL, pos=k_.POS_COL, lemma=k_.LEMMA_COL)


def main(_):
    logger = utils.get_logger(FLAGS.model)
    logger.info('Reading CoNLL-U sentences from "%s"...' % FLAGS.conllu)
    df = preprocess.conllu_to_df(
        FLAGS.conllu, getattr(cleanup, FLAGS.cleanup_function),
        min_count=FLAGS.min_count, lemmatizer_info=True)

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

    logger.info('Data sample:\n %s' % df.head())

    logger.info('Splitting between training, test and val...')
    train, test = train_test_split(df, test_size=0.2)
    logger.info('# Training examples: %d' % len(train))
    logger.info('# Test examples: %d' % len(test))

    logger.info('Preparing training data and feature/label encoders...')
    sfe, dle = _build_encoders(df)

    logger.info('Preparing batch generators...')
    batch_generator = utils.BatchGenerator(df, sfe, dle)

    logger.info('Building the model...')
    model = _build_model(sfe.output_shape, dle)
    model.summary(print_fn=logger.info)

    logger.info('Running training...')
    model.compile('adam', 'categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit_generator(batch_generator, epochs=FLAGS.epochs,
                                  verbose=2)
    for i in range(FLAGS.epochs):
        epoch_metrics = {k: v[i] for k, v in history.history.items()}
        logger.debug('Epoch %d: %s' % (i + 1, sorted(
            epoch_metrics.items())))

    if FLAGS.exceptions:
        logger.info('Loading exceptions...')
        exceptions = LookupDictionary.from_csv(
            FLAGS.exceptions)
    else:
        exceptions = LookupDictionary(columns=[
            k_.WORD_COL, k_.POS_COL, k_.LEMMA_COL])

    logger.info('Persisting the model and parameters...')
    lemmatizer = Lemmatizer(model=model, feature_enc=sfe, label_enc=dle,
                            exceptions=exceptions)
    lemmatizer.save(FLAGS.model)

    # Add losses to the exception dictionary, so that they can be
    # labeled correctly, if specified by the caller.
    if FLAGS.no_losses:
        logger.info(
            'Adding losses to the dictionary with exceptions...')
        n_start = len(lemmatizer.exceptions)
        # noinspection PyUnboundLocalVariable
        _add_losses_as_exceptions(lemmatizer, orig_df, logger)
        logger.info('# Exceptions added: %d' % (
                len(lemmatizer.exceptions) - n_start))
        lemmatizer.save(FLAGS.model)

    logger.info('Model successfully saved in folder: %s.' % FLAGS.model)


if __name__ == '__main__':
    app.run(main)
