"""Constants used across glemmazon."""

# Columns from UniversalDependencies.
WORD_COL = 'word'
POS_COL = 'pos'
LEMMA_COL = 'lemma'

# Columns from UniMorph.
MORPH_FEATURES_COL = 'morph_features'

# Columns added for the Lemmatizer.
SUFFIX_COL = 'lemma_suffix'
INDEX_COL = 'lemma_index'

# Columns added for the Inflector.
WORD_INDEX_COL = 'word_index'
WORD_SUFFIX_COL = 'word_suffix'

# Fallback
UNKNOWN_POS = '_UNKNOWN_POS'
UNKNOWN_TAG = '_UNK'
UNSPECIFIED_TAG = '_UNSP'

PARAMS_FILE = 'params.pkl'
MODEL_FILE = 'model.h5'
EXCEPTIONS_FILE = 'exceptions.csv'
LOG_FILE = 'train.log'
