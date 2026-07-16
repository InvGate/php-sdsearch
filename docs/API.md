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
echo sdsearch_version(); // "0.1.0" — also a smoke test that the extension loaded
```

## Method reference

### `SdSearch\Engine`

| Method | Purpose | Throws |
|---|---|---|
| `__construct()` | Create an engine. | — |
| `search(string $indexDir, string $paramsJson): string` | Run a query, return hits as JSON. | bad params JSON, missing index, engine error |
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
    'min_score' => 0.0,
    'limit'     => 20,
];

$json = $engine->search($indexDir, json_encode($params));
$hits = json_decode($json, true);

foreach ($hits as $hit) {
    // $hit = ['id' => int, 'score' => float, 'fields' => ['name' => 'value', ...]]
    printf("#%d  score=%.3f  %s\n", $hit['id'], $hit['score'], $hit['fields']['title'] ?? '');
}
```

### Query parameters

| Key | Type | Meaning |
|---|---|---|
| `text` | string | Free-text query over tokenized fields. |
| `where` | array | Each `{field, values[], occur}`; `occur` ∈ `must` \| `mustnot` \| `should` (default `should`). |
| `in` | array | Each `{field, values[]}`; matches the (literal, key-suffixed) field against any value. |
| `min_score` | float | Drop hits below this score. |
| `limit` | int | Maximum hits to return (`0` = unlimited). |
| `accent_insensitive` | bool | Optional (default `false`). When `true`, text matching is Spanish accent-insensitive (`avion` also matches `avión` and vice-versa). |
| `field_weights` | object | Optional (default `{}`). Per-field score multipliers (`{"title": 3.0}`); a field not listed weighs `1.0`. |

Each hit is `{ "id": int, "score": float, "fields": { name: value, ... } }`, where `id` is
the global internal document id and `fields` are the document's stored fields.

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
