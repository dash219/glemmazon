r"""Module for training a new model of the analyzer.

Basic usage:
python -m glemmazon.train_analyzer \
  --conllu data/en_ewt-ud-train.conllu \
  --model models/analyzer/en
"""

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
from glemmazon import constants as k
from glemmazon import preprocess
from glemmazon import utils
from glemmazon.encoder import (
    DenseTag,
    DictFeatureEncoder,
    DictLabelEncoder,
    LabelEncoder,
    SeqFeatureEncoder,
    SeqWordSuffix)
from glemmazon.pipeline import Analyzer, LookupDictionary

FLAGS = flags.FLAGS
flags.DEFINE_string("conllu", None, "Path to a CoNLL-U file.")
flags.DEFINE_string("model", None,
                    "Path to store the Pickle file with the model.")
flags.DEFINE_string("exceptions", None,
                    "Path to a CSV with lemma exceptions [columns: "
                    "'word', 'pos', 'lemma'].")
flags.DEFINE_string("cleanup", "basic",
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
        col: LabelEncoder(df[col].unique()) for col in df.columns
        if col not in (k.WORD_COL, k.LEMMA_COL)
    }
    dle = DictLabelEncoder(label_encoders)
    return sfe, dle


def _build_model(input_shape, dle):
    inputs = Input(shape=input_shape)
    deep = Bidirectional(LSTM(32))(inputs)
    deep = Dropout(0.3)(deep)
    deep = Dense(64)(deep)
    outputs = [
        Dense(dle.encoders[c].output_shape[0], activation='softmax',
              name=c)(deep) for c in dle.encoders
    ]
    return Model(inputs, outputs)


def _add_losses_as_exceptions(l, df, logger):
    for i, row in tqdm.tqdm(df.iterrows(), initial=1):
        feats_dict = {'word': row[k.WORD_COL], 'pos': row[k.POS_COL]}
        y_pred_dict = l(word=row[k.WORD_COL], pos=row[k.POS_COL])
        all_attrs = {**feats_dict, **y_pred_dict}

        row_as_dict = row.to_dict()
        del row_as_dict[k.LEMMA_COL]
        if all_attrs != row_as_dict:
            diff = sorted(set(row_as_dict.items()) ^
                          set(all_attrs.items()))
            logger.info(f'{i}. Added exception: "{row[k.WORD_COL]}" -> '
                        f'"{row_as_dict}" [diff: "{diff}"]. '
                        f'Ratio: {len(l.exceptions) / i}.')
            l.exceptions.add_entry(**all_attrs)


def main(_):
    logger = utils.get_logger(FLAGS.model)
    logger.info('Reading CoNLL-U sentences from "%s"...' % FLAGS.conllu)
    df = preprocess.conllu_to_df(
        FLAGS.conllu, getattr(cleanup, FLAGS.cleanup),
        min_count=FLAGS.min_count,
        lemmatizer_info=False)

    df.drop_duplicates(inplace=True)
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
        exceptions = LookupDictionary.from_csv(FLAGS.exceptions)
    else:
        exceptions = LookupDictionary(columns=list(sfe.scope |
                                                   dle.scope))

    logger.info('Persisting the model and parameters...')
    analyzer = Analyzer(model=model, feature_enc=sfe, label_enc=dle,
                        exceptions=exceptions)
    analyzer.save(FLAGS.model)

    if FLAGS.no_losses:
        logger.info(
            'Adding losses to the dictionary with exceptions...')
        n_start = len(analyzer.exceptions)
        # noinspection PyUnboundLocalVariable
        _add_losses_as_exceptions(analyzer, df, logger)
        logger.info('# Exceptions added: %d' % (
                len(analyzer.exceptions) - n_start))
        analyzer.save(FLAGS.model)

    logger.info('Model successfully saved in folder: %s.' % FLAGS.model)


if __name__ == '__main__':
    app.run(main)
