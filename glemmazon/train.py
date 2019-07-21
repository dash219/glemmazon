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

from absl import app
from absl import flags
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

from glemmazon import cleanup
from glemmazon import preprocess
from glemmazon.lemmatizer import Lemmatizer

LEMMA_SUFFIX_COL = '_lemma_suffix'
LEMMA_INDEX_COL = '_lemma_index'

FLAGS = flags.FLAGS
flags.DEFINE_string("conllu", None, "Path to a CoNLL-U file.")
flags.DEFINE_string("model", None,
                    "Path to store the model .pkl file.")

flags.DEFINE_string("exceptions", None,
                    "Path to a CSV with lemma exceptions [columns: "
                    "'word', 'pos', 'lemma'].")
flags.DEFINE_string("cleanup", "dummy",
                    "Name of the clean-up function to be used. Use "
                    "'dummy' for no clean-up.")
flags.DEFINE_integer("min_count", 3,
                     "The minimum number of counts a lemma suffix need "
                     "to have for it to be included for training.")

flags.mark_flags_as_required(['conllu', 'model'])

# noinspection PyPep8Naming
def _train_clf(X, y, X_val=None, y_val=None):
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    if X_val is not None and y_val is not None:
        y_pred = clf.predict(X_val)
        print(classification_report(y_val, y_pred))
    return clf


# noinspection PyPep8Naming
def main(_):
    print('Reading sentences from CoNLL-U...')
    df = preprocess.conllu_to_df(FLAGS.conllu,
                                 getattr(cleanup, FLAGS.cleanup),
                                 min_count=FLAGS.min_count)

    print('Vectorizing the data...')
    vec, X_train, y1_train, y2_train, X_val, y1_val, y2_val = (
        preprocess.prepare_data(df, ['word', 'pos']))

    print(X_train.shape, y1_train.shape, y2_train.shape)
    print(X_val.shape, y1_val.shape, y2_val.shape)

    print('Training the classifiers...')
    clf1 = _train_clf(X_train, y1_train, X_val, y1_val)
    clf2 = _train_clf(X_train, y2_train, X_val, y2_val)

    print('Persisting the model...')
    l = Lemmatizer()
    l.set_model(clf1, clf2, vec)
    if FLAGS.exceptions:
        l.load_exceptions(FLAGS.exceptions)
    l.save(FLAGS.model)
    print('Model successfully saved in: %s.' % FLAGS.model)


if __name__ == '__main__':
    app.run(main)
