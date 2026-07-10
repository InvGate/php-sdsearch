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
| `limit` | int | Maximum hits to return. |

Each hit is `{ "id": int, "score": float, "fields": { name: value, ... } }`, where `id` is
the global internal document id and `fields` are the document's stored fields.

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
