# Architecture

`sdsearch` is a native Rust reimplementation of the on-disk index format used by
[Zend Search Lucene](https://framework.zend.com/manual/1.12/en/zend.search.lucene.html)
(ZSL), a pure-PHP Lucene port. The goal is byte-for-byte format compatibility: an index
directory produced by ZSL must be fully readable (and mergeable) by `sdsearch`, and an
index directory produced by `sdsearch` must be fully readable (and mergeable) by ZSL —
in both directions, without any migration step. That constraint drives almost every
design decision below: the engine never invents its own format, and every writer module
is built as the deliberate inverse of a reader module so the two stay honest against
each other.

The crate is split in two:

- **`sdsearch-core`** — the engine itself: the ZSL format reader/writer and the query
  engine (term/boolean/wildcard/fuzzy/phrase). Pure Rust, no PHP dependency, and kept
  buildable/testable on both Linux and Windows.
- **`sdsearch-php`** — a thin [`ext-php-rs`](https://github.com/davidcole1340/ext-php-rs)
  binding that exposes `sdsearch-core` to PHP as a compiled extension. It only does JSON
  marshalling and panic-to-exception translation; all engine logic lives in
  `sdsearch-core`.

## Format modules (`sdsearch-core/src/zsl/`)

The reader is organized as one module per piece of the ZSL on-disk format, each a
self-contained parser for that file type:

| Module | Format piece | Responsibility |
|---|---|---|
| `bytes` | primitives | Big-endian int/long, VInt (LEB128-style varint), and ZSL's *modified* UTF-8 string encoding (length-prefixed by character count, not byte count). |
| `cfs` | `.cfs` | Compound file: a directory of `(offset, name)` entries mapped over one physical file via `mmap`, giving virtual sub-files for everything else below. |
| `fields` | `.fnm` | Field info table: field number → name + flags (indexed, tokenized, stores norms, ...). |
| `stored` | `.fdt` / `.fdx` | Stored field values (the fields returned to the caller, as opposed to the inverted index used for search). |
| `terms` | `.tis` / `.tii` | The term dictionary: terms sorted by `fieldName\0text`, with a shared-prefix encoding within a field and a sparse in-memory index (`.tii`) for seeking into the full dictionary (`.tis`). |
| `postings` | `.frq` / `.prx` | Per-term document frequency and positions: delta-encoded doc IDs and term frequencies (`.frq`), delta-encoded token positions (`.prx`). |
| `norms` | `.nrm` | Per-document, per-field length norms, encoded with Lucene's `SmallFloat` byte encoding. |
| `deletes` | `.del` | A dense bitset of deleted document IDs, one bit per document in the segment. |
| `segments` | `segments.gen` / `segments_N` | The top-level generation pointer and the list of live segments (with per-segment delete-generation), i.e. what makes a directory a coherent index at a point in time. |

A single segment is assembled by `zsl/segment.rs`, which composes the pieces above into
one `IndexReader` implementation. Real ZSL indexes are usually **multi-segment**
(incremental writes accumulate segments until an `optimize()`/merge collapses them);
`zsl/index.rs` aggregates N segments into one logical index, assigning each segment a
global document-ID base (the running sum of prior segments' document counts, deletions
included — the same numbering scheme Lucene's multi-segment readers use) and routing
every per-document lookup to the owning segment.

The writer (`sdsearch-core/src/zsl/writer/`) mirrors this module-for-module: `fnm.rs`,
`stored.rs`, `terms.rs`, `postings.rs`, `norms.rs`, `deletes.rs`, `cfs.rs`, `segments.rs`
each serialize the format their reader counterpart parses. Building each writer module as
the deliberate inverse of a trusted reader module gives a free correctness check: a
round-trip test (write, then read back with the existing reader) exercises every module
without needing an external oracle. `merge.rs` and `durability.rs` are the two pieces
without a direct reader counterpart — see below.

## Reader design

`IndexReader` (`sdsearch-core/src/index.rs`) is the trait the rest of the engine (the
query layer in `sdsearch-core/src/search.rs` and `query.rs`) is written against:
document/term frequency stats, postings, per-document field length, stored fields, and
term-prefix enumeration. `ZslSegment` and `ZslIndex` are both implementations of it —
single-segment and multi-segment respectively — so the entire query engine (boolean
composition, `build_query`, term/wildcard/fuzzy/phrase scoring) is written once and reused
unchanged against real ZSL data. That separation is also what made the read-heavy
performance work possible: the query engine's assumptions about the trait's cost model
(binary-searchable term lookups, cached global counts, lazy stored-field hydration only
for the final top-N hits) are enforced once at the trait boundary rather than duplicated
per query type.

## Writer design

`IndexWriter` (`sdsearch-core/src/zsl/writer/index_writer.rs`) is a stateful, streaming
writer with the lifecycle:

```
IndexWriter::open(dir, opts)   // takes an exclusive write-lock, snapshots the current generation
  .add_document(doc)*          // buffers in memory; auto-flushes to a new segment at a configurable cap
  .delete_document(doc_id)*    // marks a document (from the snapshot taken at open) for deletion
  .commit()                    // flushes remaining buffer, writes one new generation, releases the lock
  // or
  .optimize()                  // commits, then merges all live segments (+ pending deletes) into one
```

Key properties:

- **Bounded memory.** Documents are buffered up to a configurable cap and flushed to a
  new segment file before the cap is reached, so indexing a large batch does not require
  holding the whole inverted index in RAM.
- **Segments are invisible until commit.** A flushed segment file exists on disk but is
  not referenced by any generation record until `commit()` writes the new
  `segments_N` listing it. If the process dies mid-batch, the on-disk generation still
  points at the old, fully-valid segment list; the orphaned segment file is simply
  ignored by any reader (both engines) and is safe to garbage-collect later.
- **Merge copies, never re-tokenizes.** `merge.rs` implements the ZSL merge semantics:
  union field numbers by first-sighting order across the input segments, then for every
  live (non-deleted) document, copy — not recompute — its stored fields, its per-field
  norm bytes, and its postings/positions, renumbering document IDs densely as it goes.
  Recomputing anything from the original text (re-tokenizing, re-scoring norms) would
  reintroduce drift the merge is specifically designed to avoid; the merged segment must
  be indistinguishable, byte-for-byte in content, from what the legacy engine's own
  merge would have produced.
- **Ordered, crash-safe durability.** `durability.rs` writes new segment/generation data
  and only then atomically flips the pointer that makes it visible (`segments.gen`),
  fsyncing before the flip on the merge path; unreferenced leftovers from an interrupted
  write are cleaned up only *after* the flip, never before. See the ordering invariant
  below.
- **Cross-platform locking.** The write-lock uses `std::fs::File::{try_lock, lock,
  unlock}` — the same underlying `flock`/`LockFileEx` primitives the legacy PHP engine's
  own file lock uses — so a native writer and a legacy-engine writer correctly exclude
  each other, on both Linux and Windows, without any platform-specific code.

## Byte-fidelity details worth knowing

These are the format quirks that, if missed, silently produce an index the legacy engine
can still open but that diverges from what it would have written itself (or vice versa) —
each one cost real debugging time to pin down and is easy to get subtly wrong again:

- **Modified UTF-8, including 4-byte sequences.** ZSL strings are length-prefixed by
  *character* count and use a Java-style modified UTF-8 encoding (a NUL byte is encoded as
  the two-byte sequence `C0 80` rather than a raw `00`). Critically, 4-byte UTF-8
  sequences (emoji, CJK extension characters) must be decoded correctly — a
  straightforward two/three-byte-only decoder silently drops or corrupts these
  characters instead of erroring, so the gap only surfaces on real-world text that
  happens to contain them.
- **Term sort order is `fieldName\0text`, not by field number.** The term dictionary is
  sorted by the concatenation of field name and term text (a plain string sort), which is
  independent of the field's numeric ID in that segment. Sorting by field number instead
  produces a dictionary that looks plausible but is wrong wherever field-number
  assignment order differs from field-name alphabetical order.
- **The `.tii` (term index) synthetic entry.** The sparse term index file used to seek
  into the main term dictionary starts with a synthetic sentinel entry (a specific
  out-of-band field number, encoded as a *raw* integer rather than the usual delta
  encoding) before the real, regularly-sampled entries begin. Omitting or
  mis-encoding it desyncs every subsequent seek offset.
- **`SmallFloat` norm encoding is exact, not the reader's approximation.** Per-document
  field-length norms are stored as a single byte using Lucene's `SmallFloat` encoding.
  The reader's decode routine is an approximation good enough for scoring, but is *not*
  invertible — a writer must implement the real encode function (mapping a float to the
  nearest representable `SmallFloat` byte), not attempt to reverse the reader's
  approximation, or written norms will silently diverge from what the legacy engine
  would have produced for the same field lengths.
- **Write the compound file before flipping the generation pointer.** A new segment's
  `.cfs` (and any new `.del`/merged files) must be fully written and durable *before* the
  generation pointer (`segments.gen`) is updated to reference it. The ordering invariant
  end-to-end is: write new data → fsync (on the merge/optimize path) → atomically flip the
  pointer → clean up now-unreferenced old data. If the process is killed at any point
  before the flip, the old generation is still complete and valid; if killed after, the
  new generation is complete and valid, and it's only harmless orphaned files that lag
  behind. The pointer flip itself uses a temp-file-plus-rename so a crash mid-flip can
  never leave a torn, inconsistent pointer value.
