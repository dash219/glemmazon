"""Language-specific functions for cleaning up UD data."""

from pandas import DataFrame


def en_ewt(df: DataFrame) -> DataFrame:
    """Clean-up the corpus UD_English-EW."""
    df = df[df.abbr.isna()]  # em (<them)
    df = df[df.foreign.isna()]  # reunion
    df = df[df.typo.isna()]  # opinon (<opinion)
    df = df[df.numtype.isna()]  # 4.5
    df = df.drop(['abbr', 'foreign', 'typo', 'numtype'], axis=1)
    return df


def dummy(df: DataFrame) -> DataFrame:
    return df
