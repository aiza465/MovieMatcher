"""Data loading and cleaning for Movie Matcher.

Dataset: TMDB 5000 Movie Dataset (title, genres, overview), trimmed to the
top 400 movies by vote_count for a fast, recognizable working set. Source:
https://github.com/vamshi121/TMDB-5000-Movie-Dataset (mirror of the public
Kaggle "TMDB 5000 Movie Dataset").

The raw `genres` column ships as a JSON-like string, e.g.
    [{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]
It is pre-parsed into a space-separated genre string ("Action Adventure")
during the trim step (see data/movies.csv) so CountVectorizer can treat it
as a bag-of-genre-words.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ["title", "genres", "overview"]


def load_movies(path: str) -> pd.DataFrame:
    """Load the movies CSV and return a cleaned, de-duplicated DataFrame.

    Cleaning decisions (documented per assignment spec section 3.1):
    - Missing `overview` is filled with "" rather than dropped: a movie
      with genres but no description can still be recommended via the
      genre-based pipeline, so dropping it would lose valid data.
    - Missing `genres` is filled with "" for the same reason (a movie
      with only a description remains useful for the description pipeline).
    - A row missing BOTH genres and overview carries no usable signal for
      either vectorizer, so it is dropped.
    - Titles are normalized into a `title_norm` column (lowercased,
      whitespace-trimmed) used for matching; the original `title` is kept
      for display.
    - Exact duplicate titles (case-insensitive) are deduplicated, keeping
      the FIRST occurrence (the dataset is pre-sorted by vote_count
      descending, so the first occurrence is the more popular/canonical
      entry).
    """
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()

    # Drop rows with no usable signal at all.
    both_missing = df["genres"].isna() & df["overview"].isna()
    df = df[~both_missing]

    # Fill single-field gaps instead of dropping the row.
    df["genres"] = df["genres"].fillna("")
    df["overview"] = df["overview"].fillna("")

    # Normalize titles for case/whitespace-insensitive matching.
    df["title"] = df["title"].astype(str).str.strip()
    df["title_norm"] = df["title"].str.lower().str.strip()

    # Deduplicate exact (normalized) title repeats, keep first occurrence.
    df = df.drop_duplicates(subset="title_norm", keep="first")

    df = df.reset_index(drop=True)
    return df