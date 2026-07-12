# sdsearch benchmarks

A reproducible benchmark suite that measures the sdsearch engine on three workloads across
index sizes of **1k / 10k / 50k / 100k / 500k** documents, and compares **our PHP extension
against Zend Search Lucene** on the same corpus and the same queries.

```bash
# full run (needs the Zend oracle for the comparison — see Prerequisites)
ZEND_LUCENE_PATH=/path/to/zend-framework-1/library ./benches/run.sh

# fast smoke run (1k + 10k only)
./benches/run.sh --quick
```

Output: `benches/results.jsonl` (one JSON line per measurement) and `benches/RESULTS.md`
(rendered tables with cross-engine speedups). `results.jsonl` is machine-specific and
git-ignored; regenerate it locally.

## What is measured

Three workloads, each as a function of index size N:

| Workload  | What it does |
|-----------|--------------|
| `rebuild` | Build a **production-shaped** index from scratch: add N docs on top of the committed KB base, then `optimize()` to a single segment (as the host does per batch). |
| `churn`   | On an optimized N-doc index, delete the first 1% of docs (by global id) and add 1% fresh docs, then commit (incremental update). |
| `search`  | Query three term classes at two realistic paging depths (top-20, top-100) over the **optimized** index and report p50/p95 latency; the true hit count per class is measured separately. |

Every index the benchmark searches or churns is `optimize()`d to a single segment first — the shape
a production deployment uses — so query cost reflects the real deployment, not a many-segment
work-in-progress index.

Three engines:

- **`native`** — the Rust core directly (`sdsearch-core/examples/bench_engine.rs`). This is the
  only engine that can report the **exact Rust heap** high-water mark of an operation.
- **`sdsearch`** — our PHP extension (`SdSearch\Writer` / `SdSearch\Engine`), i.e. exactly what a
  PHP application uses.
- **`zend`** — `Zend_Search_Lucene`, the pure-PHP engine sdsearch replaces.

## Memory: three different numbers, on purpose

- `heap_peak_kb` (**native only**): peak heap bytes attributed to the measured op alone (a
  tracking global allocator, reset to steady-state right before the op). PHP's
  `memory_get_peak_usage()` **cannot** see this, because the Rust heap lives outside the Zend
  memory manager — that is why the native bench exists.
- `rss_peak_kb` / `rss_peak_mb`: process peak resident memory (`VmHWM` from
  `/proc/self/status`). It includes both engines' real footprint (the Rust heap shows up here),
  so **RSS is the number to compare across engines**.
- `php_peak_mb` (PHP engines): `memory_get_peak_usage(true)`. Meaningful for `zend`, but it
  **understates `sdsearch`** (Rust heap invisible), so it is kept in `results.jsonl` for
  reference and left out of the report tables.

Each measurement runs in a **fresh process** so peak-RSS is not cross-contaminated. For `churn`
and `search`, the PHP engines run against a **pre-built index** (an unmeasured `build` step), so
the build's memory does not leak into the churn/search RSS. The native bench builds in-process
but isolates the op's heap with the allocator reset, so its `heap_peak_kb` stays clean (its
`rss_peak_kb` for churn/search may still include the build and is context-only).

**Native runs in two passes.** The tracking allocator's atomic bookkeeping on every alloc/free
taxes the hot path, so a heap-tracked run's wall time is not comparable to the extension's (which
uses the plain system allocator). The runner therefore measures each native `rebuild`/`churn`
twice: a **time pass** (`BENCH_TRACK_HEAP=0`, no atomics → honest `ms`, the engine floor without
the FFI/JSON boundary) and a **heap pass** (`BENCH_TRACK_HEAP=1` → accurate `heap`, `ms` ignored).
The report takes `ms`/`rss` from the time pass and `heap` from the heap pass. Absolute times are
also sensitive to machine load (a busy laptop / thermal throttling adds noise, especially at
small N where a run is only milliseconds); the **relative** ordering across engines is the stable
signal — run on an idle, AC-powered machine for clean absolute numbers.

## Corpus and queries (deterministic)

The corpus is fully deterministic — no RNG — so runs are reproducible and the two engines index
byte-identical documents. Doc `i` has a `title` (~5 tokens), a `body` (~40 tokens drawn from a
fixed 25-word pool), and an `id` keyword. The generator is defined **identically** in
`sdsearch-core/examples/bench_engine.rs` (`gen_docs`) and `tools/bench_compare.php` (`gen_one`);
keep them in sync or the engines stop comparing the same work.

Three tokens are planted to fix the search result classes at exact, size-independent doc
frequencies (distinct 3-char prefixes so the fuzzy query path, `prefix_len = 3`, can never
cross-match one class to another):

| Class  | Token         | Appears in | Query result |
|--------|---------------|------------|--------------|
| `many` | `widetoken`   | every doc  | ~N hits |
| `few`  | `sparsetoken` | 5 docs     | 5 hits |
| `none` | `absenttoken` | never      | 0 hits |

Searches use the same **fuzzy + wildcard + parser** boolean on all three engines (mirroring the
engine's `text_subquery` and the host app's query builder), so hit counts match — the report's
`hits` column doubles as a correctness gate: it must be identical across engines. Each query
**reopens the index**, exactly as `SdSearch\Engine::search` and a fresh `Zend_Search_Lucene::open`
do per request; at larger N the index-open cost dominates the low-hit classes. The native column
uses the same `search_index` path, so it measures the engine's real per-request cost minus the
FFI/JSON boundary. Zend's `find()` has no top-K limit (it always scores and returns everything),
so its top-20 and top-100 latencies coincide.

## Prerequisites

- A Rust toolchain (builds `examples/bench_engine` and the PHP extension).
- PHP 8.x CLI with the `sdsearch` extension available. If it is not already enabled in `php.ini`,
  the runner loads `target/release/libsdsearch.so` with `-d extension=…` automatically.
- **`ZEND_LUCENE_PATH`** pointing at a Zend Search Lucene `library/` root, for the `zend`
  comparison. Without it, `zend` runs are recorded as `skipped` and the suite still reports
  `native` + `sdsearch`.

`./benches/run.sh` builds everything in release by default (pass `--no-build` to skip).

## Zend at large sizes

Zend indexes in pure PHP and is orders of magnitude slower, so large rebuilds are guarded:

- `zend` is **skipped for sizes above `BENCH_ZEND_MAX_DOCS` (default 100000)**. Set
  `BENCH_ZEND_MAX_DOCS=500000` to attempt the 500k point.
- Every `zend` run has a wall-clock budget `BENCH_ZEND_TIMEOUT` (default 900s); over-budget runs
  are recorded as `skipped`, never fatal.
- For sizes at/above `BENCH_ZEND_ULIMIT_FROM` (default 500000) an address-space cap
  `BENCH_ZEND_MEM_MB` (MB, `0` = off) is applied; an over-cap run fails its malloc and is skipped.

`native` and `sdsearch` always run the full 1k–500k range.

## Flags

```
--quick                 sizes = 1000 10000 only
--sizes "1000 50000"    explicit size list
--workloads "rebuild"   subset of {rebuild churn search}
--engines "native zend" subset of {native sdsearch zend}
--iters N               sampled search iterations (default 50)
--no-build              skip the release build
--append                append to results.jsonl instead of truncating
```

## Files

- `benches/run.sh` — orchestrator (matrix, guards, collection, report).
- `tools/bench_compare.php` — one measurement (engine × workload × size) → one JSON line.
- `sdsearch-core/examples/bench_engine.rs` — the native bench (exact heap).
- `tools/bench_report.php` — renders `RESULTS.md` from `results.jsonl`.
