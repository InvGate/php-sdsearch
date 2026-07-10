# sdsearch

[![build](https://github.com/InvGate/php-sdsearch/actions/workflows/build.yml/badge.svg)](https://github.com/InvGate/php-sdsearch/actions/workflows/build.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A native Rust engine that reads **and** writes the [Zend Search Lucene](https://framework.zend.com/manual/1.12/en/zend.search.lucene.html)
(ZSL) index format byte-for-byte, plus a PHP extension binding
([`ext-php-rs`](https://github.com/davidcole1340/ext-php-rs)) that exposes it to PHP applications.

Zend Search Lucene is the pure-PHP Lucene port shipped with Zend Framework 1. It is
correct but slow — both indexing and querying run entirely in PHP. `sdsearch` is a
drop-in engine for the *same on-disk index format*: it can open an index that
`Zend_Search_Lucene` created, and `Zend_Search_Lucene` can open (and merge) an index
that `sdsearch` wrote. Nothing about the format changes, so an application can switch
between the two engines — in either direction — without reindexing.

## Status

The engine is feature-complete for the read and write paths described below and is
exercised by an extensive test suite (`sdsearch-core`) plus a set of differential/golden
harnesses that check byte-for-byte parity against a real Zend Search Lucene install (see
[Testing against the ZSL oracle](#testing-against-the-zsl-oracle-optional)). It is not a
general-purpose Lucene implementation — it supports the format and query surface needed
to replace `Zend_Search_Lucene` as an index reader/writer, not the full historical Lucene
feature set.

## What it does

- **Reader** (`sdsearch-core::zsl`) — opens a ZSL index directory (single- or
  multi-segment, including deleted docs), and answers term, boolean, wildcard, fuzzy, and
  phrase queries through the same query engine used by the rest of `sdsearch-core`.
- **Writer** (`sdsearch-core::zsl::writer`) — a streaming `IndexWriter` that adds
  documents, deletes documents, commits, and merges/optimizes segments, producing index
  files a stock Zend Search Lucene install can open, merge, and continue writing to.
- **PHP binding** (`sdsearch-php`) — a compiled extension (`sdsearch`) exposing
  `SdSearch\Engine` (search) and `SdSearch\Writer` (index/commit/optimize) to PHP, with a
  panic-safe FFI boundary (a Rust panic becomes a catchable `PhpException`, never a
  crashed worker).

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the module layout, the reader/writer design,
and the byte-fidelity details that make the format compatibility work.

## Building

Requires a stable Rust toolchain (see `rust-toolchain.toml`) and, for the PHP extension,
a PHP development install plus `clang`/`libclang` (needed by `ext-php-rs`'s bindgen step).

```bash
# core engine only (no PHP required)
cargo build -p sdsearch-core --release

# PHP extension (produces target/release/libsdsearch.so on Linux, sdsearch.dll on Windows)
cargo build -p sdsearch-php --release
```

## Testing

```bash
cargo test -p sdsearch-core
cargo clippy --all-targets -- -D warnings
```

The core test suite runs entirely against fixtures committed under
`sdsearch-core/tests/fixtures/` — no external Zend Search Lucene install is required for
CI or a normal `cargo test` run.

### Testing against the ZSL oracle (optional)

A separate set of PHP tools under `tools/` cross-checks the engine against a real
Zend Search Lucene install as an independent oracle: golden interchange tests (ZSL
re-reads and merges what the native writer produced, and vice versa), a differential
suite (index the same corpus with both engines, compare term dictionaries and query
results), and perf harnesses. These are opt-in and local-only — they need a checkout of
Zend Search Lucene's `library/Zend/Search/Lucene` tree on disk.

Point them at it with `ZEND_LUCENE_PATH`:

```bash
cargo build -p sdsearch-core --release --example append_writer --example stream_writer
ZEND_LUCENE_PATH=/path/to/zend-framework-1/library php tools/golden_writer.php
ZEND_LUCENE_PATH=/path/to/zend-framework-1/library php tools/diff_writer.php
```

If `ZEND_LUCENE_PATH` is unset, these tools print a message and exit `0` (no-op) so they
never fail CI or a casual run.

## Performance

Order-of-magnitude numbers from local benchmarking against the legacy PHP engine:

- **Query latency:** on a single-segment index of ~135k documents, the native reader
  answered representative queries **~47–95× faster** than `Zend_Search_Lucene`
  (single-digit-to-low-double-digit ms vs hundreds-to-thousands of ms), with lower peak
  memory (mmap-backed, resident cost scales with the term dictionary, not the corpus).
- **Indexing throughput:** the native streaming writer indexed the same document batches
  **~95–150× faster** than ZSL's PHP writer, with peak RSS bounded by a configurable
  buffered-docs cap rather than growing with the batch size.

These are single-machine, order-of-magnitude measurements, not precise benchmarks — treat
them as a shape-of-the-win indicator, not a guarantee.

## Continuous integration

`.github/workflows/build.yml` builds and tests on both **Linux** and **Windows** on every
push/PR: `cargo test -p sdsearch-core`, a release build of the PHP extension, and a smoke
test that loads it into PHP and calls into it. No ZSL install is needed in CI — it only
exercises the fixture-backed test suite and the extension build/load.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
