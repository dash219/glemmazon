"""Dataset-specific functions for cleaning up data."""

__all__ = ['dummy', 'en_ewt']

from pandas import DataFrame

from glemmazon import constants as k


def _filter(df):
    return df.filter([k.WORD_COL, k.POS_COL, k.LEMMA_COL])


def en_ewt(df: DataFrame) -> DataFrame:
    """Clean-up the corpus UD_English-EW."""
    df = df[df.abbr.isna()]  # em (<them)
    df = df[df.foreign.isna()]  # reunion
    df = df[df.typo.isna()]  # opinon (<opinion)
    df = df[df.numtype.isna()]  # 4.5
    df = df.drop(['abbr', 'foreign', 'typo', 'numtype'], axis=1)
    return df


def dummy(df: DataFrame) -> DataFrame:
    return _filter(df)
