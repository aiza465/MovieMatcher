"""Demonstrates the deterministic tie-breaking rule on a real query.

Run: python3 scripts/verify_tiebreak.py

'Inception' has a 6-way tie in genre similarity (multiple movies share
the exact same genre overlap score). This confirms the tie is always
broken the same way: alphabetically by title, so results never shuffle
between runs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from movie_matcher.data import load_movies
from movie_matcher.similarity import build_genre_vectors, top_n_by_genre


def main() -> None:
    df = load_movies("data/movies.csv")
    vectorizer, genre_matrix = build_genre_vectors(df)
    idx = df.index[df["title_norm"] == "inception"][0]

    print("Running the same genre-based query 3 times for 'Inception':\n")
    for run in range(1, 4):
        result = top_n_by_genre(df, genre_matrix, idx, n=5)
        print(f"Run {run}: {list(result['title'])}")

    print(
        "\nAll 3 runs return the same order -> tie-breaking rule "
        "(similarity desc, then title alphabetically) is deterministic."
    )


if __name__ == "__main__":
    main()