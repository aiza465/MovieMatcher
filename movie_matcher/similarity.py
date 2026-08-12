"""Vectorization and similarity logic for Movie Matcher.

Kept free of any I/O (no print/input) so it can be unit tested directly.
The CLI layer (cli.py) is the only place that talks to the terminal.
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_genre_vectors(df: pd.DataFrame) -> tuple:
    """Vectorize the `genres` column with CountVectorizer (bag-of-genres).

    CountVectorizer (not TF-IDF) is the right tool here because genre lists
    are short, controlled vocabularies where every word is already
    meaningful ("Action", "Drama", ...) — there's no equivalent of common
    filler words that need down-weighting the way there is in prose.
    A raw count of shared genre-words is a fine similarity signal on its own.

    Returns:
        (vectorizer, genre_matrix) - the fitted CountVectorizer and the
        resulting sparse feature matrix, aligned row-for-row with `df`.
    """
    vectorizer = CountVectorizer(token_pattern=r"[^\s]+")
    genre_matrix = vectorizer.fit_transform(df["genres"])
    return vectorizer, genre_matrix


def top_n_by_genre(df: pd.DataFrame, genre_matrix, movie_index: int, n: int = 3) -> pd.DataFrame:
    """Return the top-n most genre-similar movies to df.iloc[movie_index].

    Excludes the queried movie itself. Ties in similarity score are broken
    alphabetically by title (documented tie-break rule, see README) so
    results are reproducible across runs.
    """
    sims = cosine_similarity(genre_matrix[movie_index], genre_matrix).flatten()

    result = df.copy()
    result["similarity"] = sims
    result = result.drop(index=movie_index)

    result = result.sort_values(
        by=["similarity", "title"], ascending=[False, True]
    )
    return result.head(n)[["title", "genres", "similarity"]].reset_index(drop=True)