# Contributing

## Prerequisites

- A stable Rust toolchain (see `rust-toolchain.toml`; `rustfmt` and `clippy` components
  are included).
- For `sdsearch-php` only: a PHP development install, plus `clang`/`libclang` (needed by
  `ext-php-rs`'s `bindgen` step to generate PHP header bindings).

## Building

```bash
# core engine only, no PHP required
cargo build -p sdsearch-core --release

# PHP extension (target/release/libsdsearch.so on Linux, sdsearch.dll on Windows)
cargo build -p sdsearch-php --release
```

## Testing

```bash
cargo test -p sdsearch-core
cargo clippy --all-targets -- -D warnings
```

The core test suite is entirely self-contained: it runs against fixture index
directories committed under `sdsearch-core/tests/fixtures/`, so no external Zend Search
Lucene install is required to build, test, or contribute.

If you touch `sdsearch-php`, also build and smoke-test the extension:

```bash
cargo build -p sdsearch-php --release
php -d extension=$(pwd)/target/release/libsdsearch.so tests/smoke.php
```

## Testing against the ZSL oracle (optional, local-only)

Format-level changes (anything under `sdsearch-core/src/zsl/`) should ideally also be
checked against a real Zend Search Lucene install, which acts as an independent oracle
for byte-for-byte compatibility. This is optional and not required for CI, but strongly
recommended before changing reader/writer format code.

Set `ZEND_LUCENE_PATH` to a local checkout containing the `Zend/Search/Lucene` library
tree, then run the tools under `tools/`:

```bash
cargo build -p sdsearch-core --release --example append_writer --example stream_writer

# golden interchange: ZSL re-reads and merges what the native writer produced
ZEND_LUCENE_PATH=/path/to/zend-framework-1/library php tools/golden_writer.php

# differential: index the same corpus with both engines, compare term dicts + results
ZEND_LUCENE_PATH=/path/to/zend-framework-1/library php tools/diff_writer.php
```

If `ZEND_LUCENE_PATH` is unset, these tools print a message and exit `0` (no-op) —
they never fail a run or CI job that doesn't set it.

## Windows compatibility (hard constraint)

`sdsearch-core` **must** compile and pass its test suite on Windows — CI runs a
dedicated Windows job, and losing Windows support is a regression, not an acceptable
trade-off.

Concretely, that means:

- **No `libc`, no `std::os::unix`, no `#[cfg(unix)]`** in `sdsearch-core`. Anything
  platform-specific needs a cross-platform `std` equivalent or it doesn't belong in the
  core crate.
- **File locking uses `std::fs::File::{lock, try_lock, unlock}`** (stable since Rust
  1.89), not a raw `flock(2)` call or a Unix-only crate. These map to `flock`/
  `LockFileEx` under the hood — the same primitives Zend Search Lucene's own file lock
  uses on each platform — which is also what keeps native↔legacy-engine interop
  correct on both Linux and Windows.
- Durability primitives (`File::sync_all`, `std::fs::rename` for atomic replace) are
  likewise `std`-only and cross-platform; keep new durability code on that same
  foundation.

The PHP binding crate (`sdsearch-php`) has looser constraints — Windows support there is
tracked separately and may lag the core — but do not add anything to `sdsearch-core`
that only works on one platform.

## Pull requests

- Keep `cargo test -p sdsearch-core` and `cargo clippy --all-targets -- -D warnings`
  clean.
- If you change on-disk format handling, explain the byte-fidelity reasoning in the PR
  description (see `ARCHITECTURE.md` for the class of gotcha this project cares about),
  and run the ZSL oracle tools locally if you have access to a Zend Search Lucene
  install.
- Prefer small, focused changes — the format modules are deliberately one-file-per-piece
  so a change to, say, term dictionary encoding shouldn't need to touch stored-field or
  postings code.
