"""3-sample-query comparison: genre-based vs description-based recommendations.

Run: python3 scripts/compare_queries.py

Prints top-3 results from both pipelines side by side for 3 movies with
different genre/plot profiles, so the two vectorization strategies can be
compared directly (assignment spec section 3.4).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from movie_matcher.data import load_movies
from movie_matcher.similarity import (
    build_genre_vectors,
    build_description_vectors,
    top_n_by_genre,
    top_n_by_description,
)

SAMPLE_QUERIES = ["Inception", "The Notebook", "Toy Story"]


def main() -> None:
    df = load_movies("data/movies.csv")
    _, genre_matrix = build_genre_vectors(df)
    _, desc_matrix = build_description_vectors(df)

    for title in SAMPLE_QUERIES:
        idx = df.index[df["title_norm"] == title.lower()][0]
        print("=" * 70)
        print(f"QUERY: {title}  |  genres: {df.loc[idx, 'genres']}")

        genre_results = top_n_by_genre(df, genre_matrix, idx, n=3)
        desc_results = top_n_by_description(df, desc_matrix, idx, n=3)

        print("\n-- genre-based (CountVectorizer) --")
        print(genre_results[["title", "similarity"]].to_string(index=False))

        print("\n-- description-based (TF-IDF) --")
        print(desc_results[["title", "similarity"]].to_string(index=False))
        print()


if __name__ == "__main__":
    main()