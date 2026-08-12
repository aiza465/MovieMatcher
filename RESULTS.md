# Results: Genre-based vs Description-based Recommendations

Comparison run via `python3 scripts/compare_queries.py`, full output in that
script's docstring / terminal output above.

## Query 1: Inception (Action, Thriller, Science Fiction, Mystery, Adventure)

- **Genre-based** returned Armageddon, G.I. Joe: Retaliation, Green Lantern —
  all share Inception's genre tags almost exactly (similarity 0.91), but none
  of them share Inception's actual premise (dream infiltration, layered
  reality). This is generic "same genre bucket" matching.
- **Description-based** returned Mission: Impossible - Rogue Nation, The Fifth
  Element, Transformers — lower similarity scores (0.05-0.10, since plots are
  far more unique than genre tags), but these are plausibly closer in tone:
  slick, high-concept action thrillers rather than random Action/Sci-Fi movies.
- **Verdict:** description-based felt more genuinely relevant here, even
  though its scores are much lower in absolute terms.

## Query 2: The Notebook (Romance, Drama)

- **Genre-based** returned A Beautiful Mind, Me Before You, Slumdog Millionaire
  — all tied at similarity 1.0, since Romance+Drama is a small, generic genre
  pool with a lot of movies fitting it loosely.
- **Description-based** returned Mad Max: Fury Road, Forrest Gump, Schindler's
  List — Forrest Gump is a reasonable plot-tone match (life-spanning romantic
  drama), but Mad Max: Fury Road clearly isn't; TF-IDF picked up on shared
  incidental words rather than real thematic overlap.
- **Verdict:** genre-based was actually more consistently on-theme here — this
  is a case where a small, well-defined genre pair (Romance+Drama) is a
  stronger signal than plot text, which can pick up noise on short overviews.

## Query 3: Toy Story (Animation, Comedy, Family)

- **Genre-based** returned Despicable Me 2, Monsters Inc., The Simpsons Movie
  — all tied at 1.0, reasonable family-animated matches but generic.
- **Description-based** returned Toy Story 3, Toy Story 2, then a clear score
  drop to Harry Potter — it found the actual sequels, with a real, meaningful
  gap between similarity scores (0.46, 0.40, then 0.07). This is the clearest
  win for description-based in the whole comparison.
- **Verdict:** description-based was decisively better — it recovered a
  genuine franchise relationship that genre tags alone can't express.

## Overall takeaway

Genre-based matching is fast and often "safe," but it flattens everything in
a genre pair into ties (see Notebook and Toy Story, both hit similarity 1.0
across multiple results) — it can't distinguish *degree* of similarity within
a genre. Description-based (TF-IDF) surfaces more specific, sometimes more
meaningful connections (Toy Story's actual sequels), but is more sensitive to
incidental wording overlap in short plot summaries (Mad Max under The Notebook).
Neither pipeline is strictly better; combining both (e.g. blending or
reranking) would likely outperform either alone.