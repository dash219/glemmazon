"""Dataset-specific functions for cleaning up data."""

__all__ = [
    'basic',
    'basic_lemmatizer',
    'dummy',
    'en_ewt'
]

from pandas import DataFrame

from glemmazon import constants as k


def _filter_lemmatizer(df):
    return df.filter([k.WORD_COL, k.POS_COL, k.LEMMA_COL])


def en_ewt(df: DataFrame) -> DataFrame:
    """Clean-up the corpus UD_English-EW."""
    df = df[df.abbr.isna()]  # em (<them)
    df = df[df.foreign.isna()]  # reunion
    df = df[df.typo.isna()]  # opinon (<opinion)
    df = df[df.numtype.isna()]  # 4.5
    df = df[~df.pos.isin(['PART', 'SYM', 'PUNCT', 'X'])]  # symbols, etc.
    df = df.drop(['abbr', 'foreign', 'typo', 'numtype'], axis=1)
    return df


def basic(df: DataFrame) -> DataFrame:
    # Filter rows based on the presence of morphological features.
    for morph_feature in [
        'abbr', # em (<them)
        'foreign',  # reunion
        'typo',  # opinon (<opinion)
        'numtype',  # 4.5
    ]:
        if morph_feature in df.columns:
            df = df[df[morph_feature].isna()]

    # Fill NaN
    df = df.fillna(k.UNSPECIFIED_TAG)

    # Filter rows based on POS values.
    for pos in ['PART', 'SYM', 'PUNCT', 'X']:
        if pos in df.columns:
            df = df[~df.pos.isin([pos])]

    return df


def basic_lemmatizer(df: DataFrame) -> DataFrame:
    return _filter_lemmatizer(basic(df))


def dummy(df: DataFrame) -> DataFrame:
    return df
