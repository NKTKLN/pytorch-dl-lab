"""Filtering for the IlyaGusev/gazeta summarization corpus."""

from pathlib import Path

import pandas as pd


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

    mask = (
        (text.str.lower() != summary.str.lower())
        & ~text.str.contains(r"https?://|www\.", regex=True)
        & ~summary.str.contains(r"https?://|www\.", regex=True)
    )

    clean = df.assign(text=text, summary=summary)[mask]
    clean = clean.drop_duplicates(subset="text").drop_duplicates(subset="summary")
    clean = clean.reset_index(drop=True)

    if cache_file is not None:
        clean.to_csv(cache_file, index=False)

    return clean


def _normalized_key(series: pd.Series) -> pd.Series:
    """Normalizes text for exact deduplication."""
    return series.str.casefold().str.replace(r"\s+", " ", regex=True).str.strip()


def deduplicate_split(
    df: pd.DataFrame,
    seen_articles: set[str],
    seen_summaries: set[str],
) -> tuple[pd.DataFrame, int]:
    """Removes duplicates within a split and overlaps with earlier splits."""
    article_keys = _normalized_key(df["text"])
    summary_keys = _normalized_key(df["summary"])
    keep = (
        ~article_keys.duplicated()
        & ~summary_keys.duplicated()
        & ~article_keys.isin(seen_articles)
        & ~summary_keys.isin(seen_summaries)
    )

    clean_df = df.loc[keep].reset_index(drop=True)
    seen_articles.update(article_keys.loc[keep])
    seen_summaries.update(summary_keys.loc[keep])
    return clean_df, len(df) - len(clean_df)
