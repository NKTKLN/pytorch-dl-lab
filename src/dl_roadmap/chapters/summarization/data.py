"""Filtering for the IlyaGusev/gazeta summarization corpus."""

from pathlib import Path

import pandas as pd

TEXT_WORDS = (100, 1200)
SUMMARY_WORDS = (5, 80)
MIN_CYRILLIC_RATIO = 0.5


def _cyrillic_ratio(series: pd.Series) -> pd.Series:
    """Returns each string's share of Cyrillic characters.

    Args:
        series: Strings to measure.

    Returns:
        The ratio of Cyrillic characters to total length, per row.
    """
    return series.str.count(r"[а-яА-ЯёЁ]") / series.str.len().clip(lower=1)


def prepare_gazeta(
    df: pd.DataFrame,
    cache_file: Path | None = None,
    force_prepare: bool = False,
) -> pd.DataFrame:
    """Filters noisy gazeta pairs by language ratio, length, and duplicates.

    Args:
        df: DataFrame with raw ``text`` and ``summary`` columns.
        cache_file: Path to a CSV cache of the filtered result. If it
            exists and `force_prepare` is False, it is loaded instead of
            recomputing; otherwise the result is written there.
        force_prepare: Whether to recompute even if `cache_file` exists.

    Returns:
        The filtered DataFrame with stripped text, index reset.
    """
    if not force_prepare and cache_file is not None and cache_file.is_file():
        return pd.read_csv(cache_file)

    df = df.drop(columns=["title", "date", "url"])

    text, summary = df["text"].str.strip(), df["summary"].str.strip()
    text_words = text.str.split().str.len()
    summary_words = summary.str.split().str.len()

    mask = (
        text_words.between(*TEXT_WORDS)
        & summary_words.between(*SUMMARY_WORDS)
        & (_cyrillic_ratio(text) > MIN_CYRILLIC_RATIO)
        & (_cyrillic_ratio(summary) > MIN_CYRILLIC_RATIO)
        & (text.str.lower() != summary.str.lower())
        & ~text.str.contains(r"https?://|www\.", regex=True)
        & ~summary.str.contains(r"https?://|www\.", regex=True)
    )

    clean = df.assign(text=text, summary=summary)[mask]
    clean = clean.drop_duplicates(subset="text").drop_duplicates(subset="summary")
    clean = clean.reset_index(drop=True)

    if cache_file is not None:
        clean.to_csv(cache_file, index=False)

    return clean
