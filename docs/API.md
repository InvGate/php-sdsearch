# PHP API

The `sdsearch` extension exposes three symbols to PHP: the function `sdsearch_version()`
and the classes `SdSearch\Engine` (search) and `SdSearch\Writer` (indexing). All data
crosses the boundary as JSON strings.

The full signatures with PHPDoc live in [`sdsearch.stub.php`](../sdsearch.stub.php) at the
repo root — point your IDE / PHPStan at it for autocompletion and type-checking (it is a
stub, never loaded at runtime). This page is the narrative guide with runnable examples.

> **Error handling.** Every method throws a catchable `\Exception` on any failure
> (malformed JSON, missing index, lock contention, internal error). The FFI boundary is
> panic-safe: an internal Rust panic becomes an `\Exception`, never a crashed PHP worker.
> Wrap calls in `try/catch`.

## Loading the extension

```ini
; php.ini
extension=sdsearch.so     ; Linux
; extension=sdsearch.dll  ; Windows
```

```php
echo sdsearch_version(); // "0.3.0" — also a smoke test that the extension loaded
```

## Method reference

### `SdSearch\Engine`

| Method | Purpose | Throws |
|---|---|---|
| `__construct()` | Create an engine. | — |
| `search(string $indexDir, string $paramsJson): string` | Run a query, return `{hits, total, total_capped}` as JSON. | bad params JSON, unknown `similarity`/`sort_dir`, missing index, engine error |
| `more_like_this(string $indexDir, string $paramsJson): string` | Find documents similar to a reference document, return hits as JSON. | bad params JSON, missing index, engine error |

### `SdSearch\Writer`

| Method | Purpose | Throws |
|---|---|---|
| `__construct()` | Create a writer (not yet open). | — |
| `open(string $indexDir): void` | Take the write-lock + open. | index locked, open error |
| `try_open(string $indexDir): bool` | Like `open` but returns `false` if busy. | any error except "locked" |
| `find_doc_id(string $idField, string $value): int` | `<idField>_key:value` → global doc id, or `-1`. | writer not open |
| `find_doc_ids(string $field, string $value): int[]` | Literal `<field>:value` → all matching doc ids. | writer not open |
| `delete_document(int $docId): void` | Mark a doc deleted (neg/out-of-range = no-op). | writer not open |
| `add_document(string $docJson): void` | Buffer a doc for the next commit. | bad JSON, unknown kind, writer closed |
| `commit(): int` | Flush adds+deletes, **consume** the writer. | writer not open |
| `optimize(): int` | Commit then merge into one segment, **consume**. | writer not open |
| `document_count(): int` | Live base + buffered − deletes. | writer not open |

`commit()` and `optimize()` consume the writer: after either, the object is closed and any
further method throws "writer not open". Create a fresh `Writer` for the next batch.

**When to call `optimize()`.** Each `commit()` adds one or more segments; the segment count
only shrinks when you `optimize()` (there is no automatic merge policy). A read (search) opens
every live segment, so an index that is committed many times without optimizing gets slower
and heavier to open. The recommended pattern for a bulk feed is **open once → `add_document`
per doc → `optimize()` once at the end** (rather than open/commit per document), which keeps
the index compacted to a single segment.

**`optimize()` resource profile.** The merge is streaming and bounded-memory: peak heap is a
per-term working set plus small per-document bookkeeping, independent of the corpus text
volume (on a ~135k-doc index, ~0.1 GB peak heap). While it runs it writes temporary files
(`<segment>.fdt.tmp` / `.frq.tmp` / `.prx.tmp`) into the index directory sized in aggregate
close to the final segment, so provision disk accordingly. Stale generation manifests
(`segments_N`) are pruned automatically after each commit/optimize.

## Indexing (write path)

```php
use SdSearch\Writer;

$indexDir = '/var/lib/app/search-index';

$w = new Writer();
$w->open($indexDir);              // throws if another writer holds the lock

// Add a document. Field kinds: "text" (tokenized+stored), "keyword" (exact+stored),
// "unindexed" (stored only).
$doc = [
    'fields' => [
        ['name' => 'id_key', 'value' => '42',            'kind' => 'keyword'],
        ['name' => 'title',  'value' => 'How to reset a password', 'kind' => 'text'],
        ['name' => 'body',   'value' => 'Open settings, ...',      'kind' => 'text'],
        ['name' => 'status', 'value' => 'published',      'kind' => 'keyword'],
    ],
];
$w->add_document(json_encode($doc));

$w->commit();                     // flush; writer is now closed
```

### Updating an existing document

Deletes are by internal doc id, so resolve first, then delete, then add the new version in
the same batch:

```php
$w = new Writer();
$w->open($indexDir);

$docId = $w->find_doc_id('id', '42');   // resolves id_key:42 → global id, or -1
if ($docId !== -1) {
    $w->delete_document($docId);
}
$w->add_document(json_encode($updatedDoc));

$w->optimize();                          // commit + compact into a single segment
```

### Non-blocking open for a background feed

```php
$w = new Writer();
if (!$w->try_open($indexDir)) {
    // another worker is writing — skip this cycle instead of throwing/blocking
    return;
}
// ... add/delete ...
$w->commit();
```

## Searching (read path)

```php
use SdSearch\Engine;

$engine = new Engine();

$params = [
    'text'      => 'reset password',
    'where'     => [
        ['field' => 'status', 'values' => ['published'], 'occur' => 'must'],
    ],
    'in'        => [
        ['field' => 'category_key', 'values' => ['10', '11']],
    ],
    'range'     => [
        ['field' => 'created_at_key', 'from' => '1700000000', 'to' => '1800000000'],
    ],
    'match_all' => [
        ['field' => 'title', 'text' => 'impresora oficina'],
    ],
    'min_score' => 0.0,
    'limit'     => 20,
    'offset'    => 0,
    'sort'      => 'created_at_key',
    'sort_dir'  => 'desc',
];

$json = $engine->search($indexDir, json_encode($params));
$res  = json_decode($json, true);

foreach ($res['hits'] as $hit) {
    // $hit = ['id' => int, 'score' => float, 'fields' => ['name' => 'value', ...]]
    printf("#%d  score=%.3f  %s\n", $hit['id'], $hit['score'], $hit['fields']['title'] ?? '');
}
printf("%d%s matches\n", $res['total'], $res['total_capped'] ? '+' : '');
```

> **Breaking change in 0.3.0.** `search()` used to return a bare JSON array of hits. It now
> returns an object, `{hits, total, total_capped}` — the hits moved under `hits`. Code doing
> `foreach (json_decode($json, true) as $hit)` iterates the three envelope keys instead of
> failing, so it must be updated rather than left to error. `more_like_this()` is unchanged
> and still returns a bare array.

### Query parameters

| Key | Type | Meaning |
|---|---|---|
| `text` | string | Free-text query over tokenized fields. With no `text`, `boolean_tree`, `in`, nor a `must`/`should` `where`, `search()` matches EVERY document (constant score), narrowed by `range`/`match_all` and minus any `mustnot` `where` — like an empty text in OpenSearch. `semantic_query()`/`hybrid_query()` still THROW `empty query` there (they do not apply `range`/`match_all`). |
| `where` | array | Each `{field, values[], occur}`; `occur` ∈ `must` \| `mustnot` \| `should` (default `should`). |
| `in` | array | Each `{field, values[]}`; matches the (literal, key-suffixed) field against any value. Groups whose values are ALL empty match NOTHING (an empty allow-list, not "no filter"). |
| `range` | array | Optional (default `[]`). Each `{field, from?, to?}`; keeps docs whose `field` term is in the inclusive `[from, to]` range (either bound optional). `field` verbatim. Multiple entries are ANDed. Docs missing the field are excluded. **Bounds are compared as bytes, not numerically** — see the note below. |
| `match_all` | array | Optional (default `[]`). Each `{field, text}`; keeps docs whose `field` contains ALL the analyzed words of `text` (AND). A non-scoring filter, ANDed with `range` and with the other `match_all` entries. Matching is on the engine's analyzed tokens: `"impresora"` does not match `"impresoras"`. |
| `min_score` | float | Drop hits below this score. |
| `limit` | int | Maximum hits to return (`0` = unlimited). |
| `offset` | int | Optional (default `0`). Leading hits to skip, for pagination. |
| `track_total_hits` | int\|bool | Optional (default `1001`). Integer `n` caps the reported `total` at `n`; `true` = exact count; `false` = omit `total` from the response. |
| `sort` | string | Optional. Keyword field to order by, used VERBATIM (pass the `_key` name). Omitted or `"_score"` = relevance order. A field the index does not have → error. See the ordering rules below. |
| `sort_dir` | string | Optional (default `"desc"`). `"asc"` or `"desc"`; any other value → error. Only read when `sort` is set. |
| `accent_insensitive` | bool | Optional (default `false`). When `true`, text matching is Spanish accent-insensitive (`avion` also matches `avión` and vice-versa), including the free-text prefix leaf (`camión` also reaches `camioneta`). |
| `field_weights` | object | Optional (default `{}`). Per-field score multipliers (`{"title": 3.0}`); a field not listed weighs `1.0`. |
| `similarity` | string | Optional scoring algorithm: `"bm25"` (default) or `"tfidf"`. Unknown value → error. As of 0.2.0 BM25 is the default ranking; pass `"similarity": "tfidf"` to select the legacy TF-IDF scoring shape instead of BM25. |
| `wildcard_min_prefix` | int | Optional (default `2`). Minimum literal-prefix length before the first `*`/`?` in the free-text wildcard leaf, so a short single-word query does not scan the whole vocabulary. Pass `0`/`1` for typeahead surfaces. Changed in 0.3.0: previously always `0`. |
| `synonyms` | bool | Optional (default `false`). When `true`, each query token is also matched against its bundled synonyms and cross-lingual (ES↔EN) translations, each added as a down-weighted `should` clause. See "Bundled data / attribution" below. |
| `exact_match` | bool | Optional (default `false`). When `true`, `text` is matched as an exact PHRASE — the words must appear adjacent and in order inside a single field — instead of the default fuzzy/prefix/OR bag. The phrase must occur in at least ONE indexed field; it never spans two fields. No stemming and no slop: `"impresora rota"` does not match `"rota la impresora"` nor `"impresoras rotas"`. Ignored when `boolean_tree` is set. An empty `text` makes it a no-op. |
| `boolean_tree` | object | Optional. A nested AND/OR/NOT expression whose leaves are phrases. When present it REPLACES the free-text sub-query — `text` no longer participates in matching (callers still use it for highlighting). `where` / `in` / `range` / `match_all` / `sort` / paging still apply on top. See "Boolean tree queries" below. |

### Boolean tree queries

`boolean_tree` node shapes — `and`/`or` carry `children`, `not` carries a single `child`
(singular), `term` carries a `phrase`:

```json
{ "type": "and", "children": [
    { "type": "term", "phrase": "impresora rota" },
    { "type": "not", "child": { "type": "term", "phrase": "garantia" } }
] }
```

```json
{ "type": "or", "children": [
    { "type": "term", "phrase": "vpn" },
    { "type": "term", "phrase": "acceso remoto" }
] }
```

A negated branch (`not`) contributes NO documents of its own — it only subtracts from
whatever it's ANDed against. So `or(not(x), y)` returns whatever `y` matches, NOT zero: the
negated branch drops out silently rather than failing the whole `or`. Only a tree where NO
branch contributes any positive match at all (`NOT foo` at the root, an `and` whose children
are all `not`, `children: []`) returns ZERO hits — a negated branch has nothing to subtract
from. Nesting deeper than 32 levels THROWS.

`accent_insensitive` applies INSIDE a phrase (`impresion rota` also matches `impresión
rota`). `synonyms` does NOT: a phrase is literal.

`exact_match` is pure sugar for `boolean_tree: {"type":"term","phrase":<text>}`; the
precedence is `boolean_tree` > `exact_match` > free text.

`boolean_tree`/`exact_match` are NOT accepted by `Engine::semantic_query()` or
`Engine::hybrid_query()`: pseudo-relevance feedback relaxes the base query to a should
clause, so the tree would stop being a hard filter — passing either THROWS. Use `search()`
instead. (`where` and `in` have the same underlying softening, but that is pre-existing
`search()` behavior and out of scope here — only these two new knobs are rejected.)

### Response shape

```json
{ "hits": [ { "id": 42, "score": 1.7, "fields": { "title": "…" } } ],
  "total": 128, "total_capped": false }
```

`id` is the global internal document id and `fields` are the document's stored fields.
`total` is the match count, capped per `track_total_hits` and absent entirely when that is
`false`; `total_capped` is `true` when the real count exceeded the cap (render it as
`"1000+"`).

### Sort ordering

Ordering is resolved at READ time, **per value, not per field**: a value that parses as a
64-bit integer sorts numerically, anything else sorts by byte order. Consequences:

- variable-width numeric ids order correctly (`"3" < "20" < "100"`) with **no zero-padding in
  the feed and no reindex** — this works on indexes the legacy PHP engine wrote;
- ISO-8601 timestamps come out chronological through the byte-order fallback;
- docs with no value for the field sort LAST in both directions;
- ties break by score desc, then by document id asc;
- a doc with several values for the field is placed once, under its first value in write order.

Sorting by a field the index does not have is an **error**, not an unsorted result. Without that
check a misspelled name makes every document tie on a missing value, and the hits come back in
the tiebreak order — indistinguishable from working sorting. Stored-only fields count as
present, since that is what a sort field usually is. An index with no documents is exempt, so a
sorted query over an empty index still returns an empty result rather than failing.

Cost is roughly `0.3 µs` per matched doc and flat in index size: the sort walks the matched
docs and reads each one's stored value, performing zero term-dictionary lookups. Memory is
bounded by `offset + limit` — a sum, not a product — so paging to offset 10000 retains 10020
entries, not 200000. Requesting `limit = 0` (unlimited) makes that bound vacuous and retains
one entry per match.

> **`range` bounds are byte comparisons, `sort` is numeric-aware.** The two do *not* agree on
> the same field. Sorting `id_key` with un-padded values orders correctly, but a `range` filter
> over that same field does not (`"9" > "100"` as bytes). For range-filtered numeric or date
> fields, feed a fixed-width form (epoch seconds, zero-padded ids). Neither side reports an
> error when this is wrong, so it is worth checking at feed time.

## More Like This (read path)

Given a reference document already in the index, `more_like_this` finds similar documents:
it reads the reference's stored text for the requested `fields`, picks the most distinctive
terms (high term-frequency in the doc but rare across the collection, by `tf*idf`), and runs
a boolean query for them — excluding the reference document itself.

```php
use SdSearch\Engine;

$engine = new Engine();

$params = [
    'id_field'    => 'id',          // logical id field; the engine resolves it as `id_key`
    'id_value'    => '42',          // the reference document's id value
    'fields'      => ['title', 'body'],   // stored TEXT fields to mine candidate terms from
    'source_fields' => ['id_key', 'title'], // stored fields to return per hit ([] = all)
    'term_filters'  => [            // each hit must also match these (fields used verbatim)
        ['field' => 'status_key', 'value' => 'published'],
    ],
    'range_filters' => [            // numeric range over a stored field (inclusive, half-open ok)
        ['field' => 'created_at_key', 'from' => 1_700_000_000, 'to' => 1_800_000_000],
    ],
    'min_should_match' => '30%',    // >= N terms (int) or a percentage of selected terms; 0/1 = off
    'min_term_freq'   => 2,
    'max_query_terms' => 25,
    'min_doc_freq'    => 5,
    // max_doc_freq / posting_budget: OMIT for a safety default inferred from the index size;
    // 0 = explicitly unbounded/off; a positive number = explicit cap.
    'timeout_ms'      => 0,         // 0 = off; best-effort wall-clock guard
    'field_weights'   => ['title' => 3.0],
    'size'            => 10,
    'min_score'       => 0.0,
];

$json = $engine->more_like_this($indexDir, json_encode($params));
$hits = json_decode($json, true);   // same shape as search(); [] if the reference id is unknown
```

### More Like This parameters

| Key | Type | Meaning |
|---|---|---|
| `id_field` | string | Logical id field of the reference doc; the engine resolves it as `<id_field>_key`. |
| `id_value` | string | The reference document's id value. Unknown → `[]`. |
| `fields` | array | Stored text fields to extract candidate terms from. A non-stored/unknown field is silently skipped. |
| `source_fields` | array | Optional projection of returned stored fields (`[]` = all). |
| `term_filters` | array | Each `{field, value}`; a hit must match all. `field` is used **verbatim** (no `_key` appended, unlike `id_field`). |
| `range_filters` | array | Each `{field, from?, to?}`; a hit's stored `field` must parse as a number within `[from, to]` (inclusive; either bound optional). Missing/non-numeric field → excluded. `field` verbatim. |
| `min_should_match` | int \| string | A hit must match at least this many of the selected terms. An integer (`2`) is an absolute count; a string `"N%"` (e.g. `"30%"`) is a percentage of the selected terms, floored (`3` terms × `30%` → `0`). `0`/`1` = off; a value above the selected-term count matches nothing. |
| `min_term_freq` | int | Ignore reference terms occurring fewer than this many times (default 2). |
| `max_query_terms` | int | Keep at most this many candidate terms (default 25; `0` = no cap). |
| `min_doc_freq` | int | Ignore terms rarer than this across the collection (default 5). |
| `max_doc_freq` | int | Ignore terms more common than this. **Omit** → safety default of ~half the collection size (skips non-discriminative, memory-heavy terms); `0` = unbounded. |
| `posting_budget` | int | Cap on Σ doc-frequency of selected terms — a deterministic cost guard. **Omit** → safety default of ~the collection size; `0` = off. |
| `timeout_ms` | int | Best-effort wall-clock guard; approximate scores if it fires (`0` = off). |
| `field_weights` | object | Per-field score multipliers, as in `search`. |
| `size` | int | Maximum hits to return (`0` = unlimited). |
| `min_score` | float | Drop hits below this score (scores are normalized so the top hit is `1.0`). |

## Bundled data / attribution

The `synonyms` search param (see "Query parameters" above) expands each query token
against a bundled cross-lingual dictionary derived from the Open Multilingual Wordnet
(OMW): Princeton WordNet 3.0 (English) plus the Multilingual Central Repository (MCR)
Spanish wordnet, linked through the Collaborative Interlingual Index. See
[`NOTICE`](../NOTICE) at the repo root for full licensing/attribution — notably the MCR
data is CC BY 3.0.

The compiled blob lives at `sdsearch-core/src/synonyms.bin` and is embedded at build
time; nothing reads it unless a query sets `"synonyms": true`. To regenerate it from
scratch:

1. Extract a cross-lingual pairs TSV with the `wn`-based Python extractor (one line per
   key: `key<TAB>val1<TAB>val2...`), using `omw-en:1.4` + `omw-es:1.4`.
2. Run `cargo run -p sdsearch-core --example gen_synonyms -- omw_pairs.tsv` — this
   overwrites `sdsearch-core/src/synonyms.bin` in place.

## Wrapping it safely

```php
try {
    $json = (new Engine())->search($indexDir, json_encode($params));
    $hits = json_decode($json, true, flags: JSON_THROW_ON_ERROR);
} catch (\Exception $e) {
    // missing index, malformed params, or internal engine error — never a crashed worker
    error_log('sdsearch: ' . $e->getMessage());
    $hits = [];
}
```
