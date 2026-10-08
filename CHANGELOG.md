# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project is pre-1.0, so a
breaking change bumps the **minor** version.

## [Unreleased]

## [0.4.0] - 2026-10-08

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
- **`accent_insensitive` defaults to `true`** in `search()`, `semantic_query()` and
  `hybrid_query()`. A host that omits it now gets accent-insensitive matching; pass `false`
  for accents as typed.
- **Rust API:** `Query` gains a `MatchAll` variant and `Query::Wildcard` and `Query::Fuzzy` an
  `accent_insensitive` field, `QueryError::Empty` is gone (`build_query` no longer rejects an
  empty query), and `IndexReader::is_deleted` is a new required method.

### Added

- `search()` params `fuzzy_similarity` (default `0.5`, must be in `[0, 1)`) and
  `fuzzy_prefix_len` (default `3`), until now fixed in the extension, so hosts can tune typo
  matching without a release.

### Changed

- **A field-sorted `search()` no longer pays a cold-cache spike on many matches.** Each match's
  sort value used to be read from the `.fdt`, next to that doc's full text, so the first sorted
  query on a cold page cache touched most of the file (1.5 s on a 1 GB index with 42k matches;
  ~5 s reported on a 1.5 GB one). From ~1/128 of the index in matches, the numeric sort field
  is now read once from its terms (a few MB, ~25 ms for 135k docs) into a per-query table:
  that query now takes 0.13 s cold and 0.12 s warm (was 0.19 s), with identical results.
  Fewer matches keep the per-match reads, now in doc-id order so readahead batches them. A
  stored-only, non-numeric or multi-valued sort field always uses the per-match reads. Works
  on existing Zend-written indexes; nothing is persisted.

### Fixed

- **`semantic_query()` and `hybrid_query()` apply `range` and `match_all`.** Both were parsed
  and silently ignored, so a date filter did nothing in semantic/hybrid mode and PRF drew its
  feedback from documents outside it.
- **`accent_insensitive` now covers the free-text prefix leaf.** The `text*` wildcard was
  only lowercased, so `camion` reached `camioneta` through it and `camión` did not. The prefix
  as typed is searched first, then its single-tilde variants; the `wildcard_min_prefix` gate is
  measured on the folded prefix.
- **`accent_insensitive` now covers the per-word typo (fuzzy) match**, so both spellings
  return the same totals and ranking. It counted an accent as a different letter: `camion` was
  a typo away from `camino` and `camión` was not, and in a `camion` query a doc with `camión`
  scored as a typo (boost 1/3) of the word it is. The distance is now measured between the
  folded forms, and the exact prefix also reaches its accent variants (`ultimo` finds the
  typo `últmo`). The accented spelling now reaches the same typos as the plain one, so its
  total can grow.
- **Accent variants stop at 64-character tokens.** A longer whitespace-free run (a pasted
  path, `a,a,a,…`) expanded to one variant per vowel, O(n²) memory that could abort the PHP
  worker; past the cap only its typed and folded forms are searched.
- **`accent_insensitive` finds a token typed exactly as indexed with two or more tildes**
  (`información-gestión`, a compound the analyzer keeps whole). Only single-tilde variants
  were searched, so the exact term and phrase leaves matched nothing and the doc lost that
  part of its score.
- **A typo variant no longer weighs as much as the exact word.** The fuzzy leaf computed each
  term's similarity and dropped it, so every variant scored at full weight; with length
  normalization, a request containing the exact word could rank below fuzzy-only ones. Each
  variant now scores times `(sim - min) / (1 - min)`, the boost Zend's `Fuzzy::rewrite`
  applies (1 for the exact word).

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

[0.4.0]: https://github.com/InvGate/php-sdsearch/releases/tag/v0.4.0
[0.3.0]: https://github.com/InvGate/php-sdsearch/releases/tag/v0.3.0
[0.2.1]: https://github.com/InvGate/php-sdsearch/releases/tag/v0.2.1
[0.2.0]: https://github.com/InvGate/php-sdsearch/releases/tag/v0.2.0
[0.1.4]: https://github.com/InvGate/php-sdsearch/releases/tag/v0.1.4
[0.1.3]: https://github.com/InvGate/php-sdsearch/releases/tag/v0.1.3
[0.1.2]: https://github.com/InvGate/php-sdsearch/releases/tag/v0.1.2
[0.1.1]: https://github.com/InvGate/php-sdsearch/releases/tag/v0.1.1
[0.1.0]: https://github.com/InvGate/php-sdsearch/releases/tag/v0.1.0
