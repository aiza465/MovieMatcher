"""Tests for movie_matcher.similarity — pure logic, no CLI/I-O involved.

Run: pytest tests/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from movie_matcher.similarity import (
    build_genre_vectors,
    build_description_vectors,
    find_movie_index,
    top_n_by_genre,
    top_n_by_description,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A small, hand-built dataset so tests don't depend on the real CSV
    or its size — keeps tests fast and independent of dataset changes.
    Includes a deliberate genre tie (rows 1-3 share the exact same genre
    string) to exercise the tie-breaking rule.
    """
    data = {
        "title": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
        "genres": [
            "Action Comedy",
            "Drama Romance",
            "Drama Romance",
            "Drama Romance",
            "Horror Thriller",
        ],
        "overview": [
            "A hero saves the day with jokes and explosions.",
            "Two people fall in love over many years.",
            "A love story spanning decades and continents.",
            "Romance blooms slowly between two strangers.",
            "A killer stalks victims in a dark house.",
        ],
    }
    df = pd.DataFrame(data)
    df["title_norm"] = df["title"].str.lower().str.strip()
    return df


def test_top_n_returns_exactly_three(sample_df):
    _, genre_matrix = build_genre_vectors(sample_df)
    result = top_n_by_genre(sample_df, genre_matrix, movie_index=0, n=3)
    assert len(result) == 3


def test_self_exclusion_genre(sample_df):
    _, genre_matrix = build_genre_vectors(sample_df)
    idx = 1  # "Beta"
    result = top_n_by_genre(sample_df, genre_matrix, movie_index=idx, n=4)
    assert "Beta" not in result["title"].tolist()


def test_self_exclusion_description(sample_df):
    _, desc_matrix = build_description_vectors(sample_df)
    idx = 0  # "Alpha"
    result = top_n_by_description(sample_df, desc_matrix, movie_index=idx, n=4)
    assert "Alpha" not in result["title"].tolist()


def test_not_found_returns_none(sample_df):
    idx, matched = find_movie_index(sample_df, "Completely Unrelated Nonexistent Title XYZ")
    assert idx is None
    assert matched is None


def test_exact_match_found(sample_df):
    idx, matched = find_movie_index(sample_df, "alpha")
    assert idx == 0
    assert matched == "Alpha"


def test_fuzzy_match_finds_close_title(sample_df):
    idx, matched = find_movie_index(sample_df, "Alfa")  # typo for "Alpha"
    assert idx == 0
    assert matched == "Alpha"


def test_tie_breaking_is_alphabetical_and_deterministic(sample_df):
    # Beta, Gamma, Delta all share identical genres -> tie in similarity.
    _, genre_matrix = build_genre_vectors(sample_df)
    idx = sample_df.index[sample_df["title"] == "Epsilon"][0]

    run1 = top_n_by_genre(sample_df, genre_matrix, movie_index=idx, n=4)["title"].tolist()
    run2 = top_n_by_genre(sample_df, genre_matrix, movie_index=idx, n=4)["title"].tolist()

    assert run1 == run2  # deterministic across repeated calls

    # Among the tied "Drama Romance" trio, alphabetical order must hold:
    # Beta < Delta < Gamma
    tied_subset = [t for t in run1 if t in {"Beta", "Gamma", "Delta"}]
    assert tied_subset == sorted(tied_subset)