"""Command-line entry point for Movie Matcher.

All actual logic (loading, cleaning, vectorizing, matching, ranking) lives
in data.py and similarity.py as pure, testable functions. This file's only
job is talking to the terminal: input(), print(), and the main loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from movie_matcher.data import load_movies
from movie_matcher.similarity import (
    build_genre_vectors,
    build_description_vectors,
    find_movie_index,
    top_n_by_genre,
    top_n_by_description,
)

DATA_PATH = "data/movies.csv"


def print_recommendations(title: str, genre_results, desc_results) -> None:
    print(f"\nBecause you liked '{title}':")
    print("\n  Genre-based recommendations:")
    for _, row in genre_results.iterrows():
        print(f"    - {row['title']}  (similarity: {row['similarity']:.3f})")

    print("\n  Description-based recommendations:")
    for _, row in desc_results.iterrows():
        print(f"    - {row['title']}  (similarity: {row['similarity']:.3f})")
    print()


def main() -> None:
    print("Loading movie dataset...")
    df = load_movies(DATA_PATH)
    _, genre_matrix = build_genre_vectors(df)
    _, desc_matrix = build_description_vectors(df)
    print(f"Loaded {len(df)} movies. Type a movie title (or 'quit' to exit).\n")

    while True:
        query = input("Movie title: ").strip()
        if query.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if not query:
            continue

        idx, matched_title = find_movie_index(df, query)
        if idx is None:
            print(f"Sorry, no movie found matching '{query}'. Try another title.\n")
            continue

        if matched_title.lower() != query.lower():
            print(f"Did you mean '{matched_title}'? Showing results for it.")

        genre_results = top_n_by_genre(df, genre_matrix, idx, n=3)
        desc_results = top_n_by_description(df, desc_matrix, idx, n=3)
        print_recommendations(matched_title, genre_results, desc_results)


if __name__ == "__main__":
    main()