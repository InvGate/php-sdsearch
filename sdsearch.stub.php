<?php

/**
 * IDE / static-analysis stub for the `sdsearch` PHP extension.
 *
 * `sdsearch` is a compiled extension (Rust, via ext-php-rs); there is no PHP source for
 * its classes. This file declares the extension's public surface with PHPDoc so that IDEs
 * (PhpStorm) and static analyzers (PHPStan/Psalm) understand it. It is NEVER loaded or
 * executed at runtime — the real symbols come from the compiled `sdsearch.so` / `sdsearch.dll`.
 *
 * The boundary is panic-safe: any internal Rust error or panic surfaces as a catchable
 * \Exception, never a crashed PHP worker.
 *
 * @see docs/API.md   for usage examples
 * @see docs/FORMAT.md for the on-disk index format
 */

namespace {
    /**
     * Returns the native engine version (the crate version, e.g. "0.1.0").
     *
     * Doubles as an end-to-end smoke test that the extension loaded correctly.
     */
    function sdsearch_version(): string {}
}

namespace SdSearch {
    /**
     * Read-only search over a ZSL index directory.
     */
    class Engine
    {
        public function __construct() {}

        /**
         * Searches a ZSL index and returns the hits as a JSON string.
         *
         * The `$paramsJson` is a JSON object with this shape (all keys optional):
         * ```json
         * {
         *   "text": "free text query",
         *   "where": [ { "field": "status", "values": ["open"], "occur": "must" } ],
         *   "in":    [ { "field": "category_key", "values": ["10", "11"] } ],
         *   "min_score": 0.0,
         *   "limit": 20,
         *   "accent_insensitive": false,
         *   "field_weights": { "title": 3.0, "description": 1.0 },
         *   "similarity": "bm25"
         * }
         * ```
         * - `where[].occur` is one of `"must"`, `"mustnot"`, `"should"` (default `should`).
         * - `in[]` matches a field against any of the literal values (already-suffixed key fields).
         * - `accent_insensitive` (optional, default `false`): when `true`, text matching is
         *   Spanish accent-insensitive (`avion` also matches `avión` and vice-versa).
         * - `field_weights` (optional, default `{}`): per-field score multipliers; a field not
         *   listed weighs `1.0`. Empty = every field weighed equally (current behavior).
         * - `similarity` (optional, default `"bm25"`): scoring algorithm, `"bm25"` or `"tfidf"`;
         *   an unknown value throws. As of 0.2.0 BM25 is the default ranking; pass
         *   `"similarity": "tfidf"` to select the legacy TF-IDF scoring shape instead of BM25.
         *
         * The return value is a JSON array of hits:
         * ```json
         * [ { "id": 42, "score": 1.7, "fields": { "title": "…", "status": "open" } } ]
         * ```
         * `id` is the global internal document id; `fields` are the stored fields.
         *
         * @param string $indexDir   Path to the ZSL index directory.
         * @param string $paramsJson JSON-encoded query parameters (see above).
         * @return string JSON-encoded array of hits (see above).
         * @throws \Exception on malformed params JSON, a missing/unreadable index, or an
         *                    internal engine error.
         */
        public function search(string $indexDir, string $paramsJson): string {}

        /**
         * More Like This: returns documents similar to a reference document as a JSON hit array.
         *
         * `$paramsJson` is a JSON object:
         * ```json
         * {
         *   "id_field": "id",
         *   "id_value": "12345",
         *   "fields": ["title", "description"],
         *   "source_fields": ["id"],
         *   "term_filters": [ { "field": "status_key", "value": "open" } ],
         *   "range_filters": [ { "field": "created_at_key", "from": 1700000000, "to": 1800000000 } ],
         *   "min_should_match": "30%",
         *   "min_term_freq": 2,
         *   "max_query_terms": 25,
         *   "min_doc_freq": 5,
         *   "max_doc_freq": null,     // omit for a safety default from index size
         *   "posting_budget": null,   // omit for a safety default from index size
         *   "timeout_ms": 0,
         *   "field_weights": { "title": 3.0 },
         *   "size": 10,
         *   "min_score": 0.0
         * }
         * ```
         * - `id_field`/`id_value` identify the reference document. `id_field` is a LOGICAL
         *   name: the engine appends `_key` (so `"id"` resolves against the indexed `id_key`).
         * - `fields` are the stored text fields to mine candidate terms from. A field that is
         *   not stored as text (typo, keyword-only, indexed-but-not-stored) is silently ignored;
         *   if none of the requested fields is found, the result is `[]` (same as "no match").
         * - `term_filters[].field` is used VERBATIM — unlike `id_field`, no `_key` is appended.
         *   Pass the already-suffixed indexed name (e.g. `"status_key"`); a wrong name matches
         *   nothing and silently empties the result set.
         * - `range_filters[]` is `{ field, from?, to? }`: a hit's stored `field` must parse as a
         *   number within `[from, to]` (inclusive; either bound may be omitted for a half-open
         *   range). `field` is verbatim like `term_filters`; a missing/non-numeric value on a doc
         *   excludes it. Suits epoch-int fields such as `created_at_key`.
         * - `min_should_match` (optional): a hit must match at least this many of the selected
         *   terms. An integer (`2`) is an absolute count; a string `"N%"` (e.g. `"30%"`) is a
         *   percentage of the selected terms, floored like OpenSearch (`3` terms × `30%` → `0`).
         *   `0`/`1` (or a percentage that floors to them) = off. Full OpenSearch grammar (negatives,
         *   `"2<75%"` combinations) is NOT supported. CAVEATS: the number of selected terms is NOT
         *   visible to the caller — it is data-dependent (a short source doc, `max_query_terms`, or
         *   `posting_budget` can trim it), so an absolute count above it returns `[]` even when
         *   similar docs exist (percentage scales with it, so it is safer). And under an early
         *   `timeout_ms`, only the terms processed before the deadline count, so a high threshold
         *   can empty the result set.
         * - `source_fields` (optional) projects the returned `fields` to just these keys; empty = all.
         * - `max_doc_freq` and `posting_budget` are tri-state: **omit** (or `null`) → the engine
         *   infers a safety default from the index size (max_doc_freq ≈ half the docs;
         *   posting_budget ≈ the doc count) so a single request can't load memory proportional
         *   to the whole collection; `0` → explicitly unbounded/off; a positive number → explicit cap.
         * - `posting_budget` caps Σ doc-frequency over selected terms (deterministic cost guard);
         *   `timeout_ms` (`0` = off) is a best-effort wall-clock guard (approximate scores if it fires).
         *
         * Returns a JSON array of hits `[ { "id": 42, "score": 1.0, "fields": {…} } ]`;
         * an unknown reference id returns `[]`.
         *
         * @param string $indexDir   Path to the ZSL index directory.
         * @param string $paramsJson JSON-encoded MLT parameters (see above).
         * @return string JSON-encoded array of hits.
         * @throws \Exception on malformed params JSON, a missing/unreadable index, or an
         *                    internal engine error.
         */
        public function more_like_this(string $indexDir, string $paramsJson): string {}

        /**
         * Semantic search via two-pass pseudo-relevance feedback (PRF): runs `$paramsJson`
         * as a normal {@see Engine::search()} query, harvests feedback terms from the
         * top hits, then re-runs an augmented query for the final result.
         *
         * `$paramsJson` accepts the SAME object as {@see Engine::search()}, plus an optional
         * `"prf"` object (all keys optional, defaults shown):
         * ```json
         * {
         *   "text": "free text query",
         *   "prf": {
         *     "top_k": 5,
         *     "num_terms": 10,
         *     "feedback_weight": 0.3,
         *     "fields": [],
         *     "min_term_freq": 1,
         *     "min_doc_freq": 1,
         *     "max_doc_freq": null,
         *     "posting_budget": null
         *   }
         * }
         * ```
         * - `prf.top_k`: number of top pass-1 hits treated as pseudo-relevant.
         * - `prf.num_terms`: max feedback terms added to the augmented query.
         * - `prf.feedback_weight`: score multiplier for the feedback-term subtree.
         * - `prf.fields` (optional, default `[]`): source fields to harvest terms from;
         *   empty = all indexed fields.
         *
         * Returns the same JSON hit array shape as {@see Engine::search()}. The result is a
         * RERANK of the augmented query, not strictly a superset of {@see Engine::search()}:
         * with a nonzero `min_score` or a limit that binds, `semantic_query` may omit hits
         * that a plain `search()` call would return.
         *
         * @param string $indexDir   Path to the ZSL index directory.
         * @param string $paramsJson JSON-encoded query parameters + optional `prf` object (see above).
         * @return string JSON-encoded array of hits (see {@see Engine::search()}).
         * @throws \Exception on malformed params JSON, a missing/unreadable index, or an
         *                    internal engine error.
         */
        public function semantic_query(string $indexDir, string $paramsJson): string {}
    }

    /**
     * Streaming writer over a ZSL index directory.
     *
     * Lifecycle: {@see Writer::open()} (or {@see Writer::try_open()}) takes an exclusive
     * write-lock; buffer changes with {@see Writer::add_document()} /
     * {@see Writer::delete_document()}; then {@see Writer::commit()} or
     * {@see Writer::optimize()} flushes them and **consumes** the writer (a second call
     * throws). A `Writer` is single-batch and not reentrant: open, mutate, commit, discard.
     *
     * Document ids returned by {@see Writer::find_doc_id()} / {@see Writer::find_doc_ids()}
     * are GLOBAL internal ids in the same space {@see Writer::delete_document()} expects.
     */
    class Writer
    {
        public function __construct() {}

        /**
         * Opens the index for writing, taking the exclusive write-lock and a cached reader
         * over the current (pre-commit) snapshot.
         *
         * @param string $indexDir Path to an existing ZSL index directory.
         * @throws \Exception if the index is already locked by another writer, or cannot be opened.
         */
        public function open(string $indexDir): void {}

        /**
         * Like {@see Writer::open()}, but returns `false` (instead of throwing) when the
         * index is already locked by another writer, leaving no writer open. Lets a feed
         * distinguish "index busy" without matching on an exception message.
         *
         * @param string $indexDir Path to an existing ZSL index directory.
         * @return bool `true` if the lock was taken and the writer is open; `false` if busy.
         * @throws \Exception on any error other than "already locked".
         */
        public function try_open(string $indexDir): bool {}

        /**
         * Resolves `<idField>_key:value` against the writer's base snapshot to a global
         * internal document id.
         *
         * @param string $idField Logical id field name; the `_key` suffix is added internally.
         * @param string $value   The id value to look up.
         * @return int The global internal document id, or `-1` if there is no match.
         * @throws \Exception if the writer is not open, or on an internal error.
         */
        public function find_doc_id(string $idField, string $value): int {}

        /**
         * Resolves a LITERAL `<field>:value` term query to ALL live matching global document
         * ids. Unlike {@see Writer::find_doc_id()}, `$field` is used verbatim (already
         * carrying its `_key` suffix as indexed). Use for multi-doc removal/dedup (e.g. a
         * multi-language category whose id maps to several docs, or `language_key:<lang>`).
         *
         * @param string $field Literal indexed field name.
         * @param string $value The value to match.
         * @return int[] Global internal document ids (empty array if none match).
         * @throws \Exception if the writer is not open, or on an internal error.
         */
        public function find_doc_ids(string $field, string $value): array {}

        /**
         * Marks a document deleted by its global internal id. A negative or out-of-range id
         * is a silent no-op. Takes effect on the next
         * {@see Writer::commit()}/{@see Writer::optimize()}.
         *
         * @param int $docId Global internal document id (from find_doc_id / find_doc_ids).
         * @throws \Exception if the writer is not open, or on an internal error.
         */
        public function delete_document(int $docId): void {}

        /**
         * Buffers a document to be added on the next commit.
         *
         * `$docJson` is a JSON object:
         * ```json
         * { "fields": [ { "name": "title", "value": "Hello", "kind": "text" } ] }
         * ```
         * `kind` is one of:
         * - `"text"`      — tokenized + indexed + stored (full-text searchable).
         * - `"keyword"`   — indexed as a single token + stored (exact-match / key fields).
         * - `"unindexed"` — stored only, not searchable.
         *
         * @param string $docJson JSON-encoded document (see above).
         * @throws \Exception on malformed JSON, an unknown `kind`, a closed writer, or an
         *                    internal error.
         */
        public function add_document(string $docJson): void {}

        /**
         * Commits buffered additions and pending deletes, then **consumes** the writer
         * (releases the lock and closes the index).
         *
         * @return int The document count the index will have after the commit.
         * @throws \Exception if the writer is not open (e.g. already committed), or on error.
         */
        public function commit(): int {}

        /**
         * Commits, then merges all live segments (plus pending deletes) into a single
         * compacted segment. **Consumes** the writer, like {@see Writer::commit()}.
         *
         * The merge is streaming and bounded-memory: peak heap is a per-term working set
         * plus small per-document bookkeeping, independent of the index's total text volume
         * (so optimizing a large index does not risk an OOM). While running it writes
         * temporary files (`<segment>.fdt.tmp`/`.frq.tmp`/`.prx.tmp`) into the index dir.
         * Prefer "open once → add_document* → optimize() once" per feed; the segment count
         * only shrinks on optimize (no automatic merge policy).
         *
         * @return int The document count the index will have after the optimize.
         * @throws \Exception if the writer is not open, or on error.
         */
        public function optimize(): int {}

        /**
         * Total documents the index will see after commit (live base + buffered − deletes).
         * Requires an open writer.
         *
         * @return int
         * @throws \Exception if the writer is not open, or on an internal error.
         */
        public function document_count(): int {}
    }
}
