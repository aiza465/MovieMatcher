# Movie Matcher

A content-based movie recommendation system. Given a movie title, it returns
the 3 most similar movies using two different vectorization strategies —
genre-based and description-based — computed with cosine similarity.

Built for the Khizex AI Engineering Internship, Week 6 build challenge.

## Dataset

TMDB 5000 Movie Dataset, trimmed to the top 400 movies by `vote_count` for a
fast, recognizable working set (title, genres, overview only).
Source: https://github.com/vamshi121/TMDB-5000-Movie-Dataset (a public
mirror of the Kaggle "TMDB 5000 Movie Dataset").

The raw `genres` column ships as a JSON-like string
(`[{"id": 28, "name": "Action"}, ...]`); it was pre-parsed into a
space-separated genre string (`"Action Adventure"`) before saving
`data/movies.csv`.

## Vectorization choices

- **Genres → `CountVectorizer`**: genre lists are a short, controlled
  vocabulary where every token is already meaningful ("Action", "Drama").
  There's no equivalent of filler words that need down-weighting, so a raw
  bag-of-genre-words count is a sufficient similarity signal.
- **Descriptions → `TfidfVectorizer`** (with English stop-word removal):
  plot overviews are free-form prose full of common connective words that
  carry little meaning. TF-IDF down-weights terms that appear across many
  overviews and up-weights rare, distinctive terms, so shared plot-specific
  vocabulary matters more than shared filler words.

The two pipelines are kept separate (not combined into one vector) so they
can be directly compared — see `RESULTS.md`.

## Tie-breaking rule

When two or more movies tie on cosine similarity score, results are sorted
by `(similarity descending, title ascending)` — i.e. ties are broken
alphabetically by title. This is deterministic: the same query always
returns the same order across runs (verified in
`scripts/verify_tiebreak.py`, using a real 6-way tie found in this dataset
for the "Inception" query).

## Fuzzy title matching

If a typed title doesn't exactly match any row (case/whitespace-normalized),
the system falls back to `difflib.get_close_matches()` against every
normalized title, with a similarity cutoff of `0.6`. If a close match is
found, it's confirmed back to the user ("Did you mean 'Inception'?") before
showing recommendations — it never silently answers for a different movie.
If nothing clears the cutoff, a clear "movie not found" message is printed
instead of crashing or guessing.

## Project structure
movie_matcher/
├── data.py # loading + cleaning (pandas, has I/O)
├── similarity.py # vectorization + similarity + matching (pure, no I/O)
└── cli.py # terminal input loop (the only place with input()/print())
scripts/
├── verify_tiebreak.py # demonstrates deterministic tie-breaking
└── compare_queries.py # genre vs description comparison for 3 sample movies
tests/
└── test_similarity.py # pytest suite
data/
└── movies.csv # trimmed TMDB dataset
RESULTS.md # 3-sample-query comparison + written analysis
`similarity.py` has zero I/O (no `print`/`input`) so it can be unit tested
directly without going through the terminal.

## Running it

Install dependencies:
```bash
pip install pandas scikit-learn pytest
```

Run the interactive CLI:
```bash
python3 movie_matcher/cli.py
```
Type a movie title, get genre-based and description-based recommendations.
Type `quit` to exit.

Run the tie-breaking demo:
```bash
python3 scripts/verify_tiebreak.py
```

Run the genre-vs-description comparison:
```bash
python3 scripts/compare_queries.py
```

Run the tests:
```bash
pytest tests/ -v
```
(If pytest crashes with a `langsmith.pytest_plugin` import error, that's an
unrelated broken plugin from another package in the environment — run
`pytest -p "no:langsmith.pytest_plugin" tests/ -v` instead, or
`pip uninstall langsmith`.)

## Results

See [`RESULTS.md`](./RESULTS.md) for the full genre-vs-description comparison
across 3 sample queries, with written analysis of which representation
performed better for each and why.