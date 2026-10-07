# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project is pre-1.0, so a
breaking change bumps the **minor** version.

## [Unreleased]

### Breaking

- **A query with nothing positive to match returns every document instead of throwing
  `empty query`**, in `search()`, `semantic_query()` and `hybrid_query()`. Nothing positive
  means no `text`, `boolean_tree` or `in`, and no `must`/`should` `where`: only
  `range`/`match_all` filters, only `mustnot` `where` groups, or no parameters at all. It used
  to throw (or return zero hits with only `mustnot`); hosts that catch the exception showed an
  empty screen. It now matches every live document, narrowed by the filters and minus the
  `mustnot`s, like an empty text in OpenSearch. `semantic_query()` has nothing to expand there
  and returns the same listing as `search()`.
- **An `in` with no values matches nothing instead of being dropped.** Hosts use `in` for
  visibility; an empty one ("visible in none") used to be ignored, so a text search
  returned every text match regardless of the filter.
- **A `boolean_tree` that only negates at the root** (`NOT x`, an `and` whose children are all
  `not`) returns every document minus the negated ones, like the equivalent `mustnot` `where`,
  instead of zero hits. A `not` nested under an `or` still drops out.
- **Rust API:** `Query` gains a `MatchAll` variant, `QueryError::Empty` is gone (`build_query`
  no longer rejects an empty query), and `IndexReader::is_deleted` is a new required method.

### Fixed

- **`semantic_query()` and `hybrid_query()` apply `range` and `match_all`.** Both were parsed
  and silently ignored, so a date filter did nothing in semantic/hybrid mode and PRF drew its
  feedback from documents outside it.

## [0.3.0] - 2026-08-26

### Breaking

- **`Engine::search()` now returns an object, not an array.** The old shape was a bare JSON
  array of hits; it is now `{"hits": [...], "total": 128, "total_capped": false}`. Callers
  doing `foreach (json_decode($json) as $hit)` must switch to `json_decode($json)->hits`.
  `total` is absent when `track_total_hits` is `false`.
- **`wildcard_min_prefix` defaults to `2`.** A free-text wildcard leaf with a literal prefix
  shorter than two characters no longer expands, so single-character prefix queries stop
  scanning the whole vocabulary. Typeahead surfaces that relied on the old behavior should
  pass `"wildcard_min_prefix": 0`.
- **A `sort` by a field the index does not have throws** instead of returning a
  tie-broken order that looks sorted. Stored-only fields count as present; an index with no
  documents is exempt.
- **An unknown `sort_dir` throws** instead of falling back to `"desc"`, so a typo cannot
  silently reverse the order.

### Added

- `Engine::semantic_query()` — two-pass pseudo-relevance-feedback (PRF) search, tunable via
  a `prf` object (`top_k`, `num_terms`, `feedback_weight`, `fields`, frequency bounds,
  `posting_budget`).
- `Engine::hybrid_query()` — Reciprocal Rank Fusion of the lexical and PRF rankings, tunable
  via a `hybrid` object (`k`, `depth`). Rank-based, so the two score scales never mix.
- `search()` query params: `range[]` (inclusive term-range filter), `match_all[]`
  (field-scoped AND filter), `offset` and `track_total_hits` (pagination), `sort` /
  `sort_dir` (keyword-field ordering, numeric when the value parses as an i64),
  `synonyms` (WordNet EN + MCR ES bundle, cross-lingual, scored below the literal token),
  `exact_match` (phrase matching across every indexed field), `boolean_tree` (nested
  AND/OR/NOT over phrase leaves, replaces the free-text sub-query), and `wildcard_min_prefix`.
- `IndexReader::stored_value()` and `terms_in_range()`; the ZSL reader reads one stored field
  without materializing the rest.

### Performance

- Field sort runs off a bounded top-K heap — no dictionary lookups, flat memory under paging.
- Filter-first candidate restriction in boolean evaluation; range allow-lists intersect in
  place and are sized from summed `doc_freq`.
- Wildcard expansion stops early at 4096 terms; top-K selection in `finalize` replaces a full
  sort.
- The synonym dictionary is only resolved when `synonyms` is on.

### Fixed

- `Must` clause scores are summed in declared order, keeping results bit-identical to the
  pre-filter-first evaluator.
- `terms_with_prefix_limited` returns empty for `limit 0` instead of everything.

## [0.2.1] - 2026-07-18

### Fixed

- An excluding `where`/`in` filter is never bypassed when the text sub-query matches nothing.

## [0.2.0] - 2026-07-17

### Breaking

- **BM25 is the default ranking**, replacing TF-IDF. Pass `"similarity": "tfidf"` to keep the
  legacy scoring shape.

### Added

- Per-search `similarity` selection (`"bm25"` / `"tfidf"`) from PHP; unknown value throws.
- `avg_field_len` on the `IndexReader` trait, the on-disk v2 reader, and aggregated in
  `ZslIndex`.

### Fixed

- Missing `SmallFloat` exponent bias in `decode_norm`.

### Performance

- `avg_field_len` is a sampled estimate bounded at open instead of a full scan.

## [0.1.4] - 2026-07-16

### Added

- More-like-this: `min_should_match` (absolute or percentage), numeric `range_filters` over
  stored fields, term selection by tf*idf with frequency filters and a posting budget, and
  memory-safety defaults for `max_doc_freq`/`posting_budget` inferred from index size.
- `Engine::more_like_this()` exposed in the PHP extension (kept `snake_case`).

## [0.1.3] - 2026-07-13

### Added

- Optional Spanish accent-insensitive matching and per-field score boosting.

## [0.1.2] - 2026-07-12

### Fixed

- Durability on the `optimize()` path: the merged `.cfs` is written atomically (staging file
  plus rename), the index directory is fsynced, and old-segment deletion is deferred so the
  reader grace window is real.

## [0.1.1] - 2026-07-11

### Added

- Bounded-memory streaming k-way merge for `optimize()` and a streaming CFS assembler.
- `WriterOpts.max_segments` (default 256); flushed segments compact when the ceiling is crossed.

### Fixed

- Stale `segments_N` manifests are pruned after each generation flip.
- Silent CFS corruption on a source-length mismatch is now caught.

## [0.1.0] - 2026-07-10

Initial public release: a native Rust engine that reads and writes the Zend Search Lucene
index format byte-for-byte, plus the PHP extension that exposes it.

[0.3.0]: https://github.com/InvGate/php-sdsearch/releases/tag/v0.3.0
[0.2.1]: https://github.com/InvGate/php-sdsearch/releases/tag/v0.2.1
[0.2.0]: https://github.com/InvGate/php-sdsearch/releases/tag/v0.2.0
[0.1.4]: https://github.com/InvGate/php-sdsearch/releases/tag/v0.1.4
[0.1.3]: https://github.com/InvGate/php-sdsearch/releases/tag/v0.1.3
[0.1.2]: https://github.com/InvGate/php-sdsearch/releases/tag/v0.1.2
[0.1.1]: https://github.com/InvGate/php-sdsearch/releases/tag/v0.1.1
[0.1.0]: https://github.com/InvGate/php-sdsearch/releases/tag/v0.1.0
