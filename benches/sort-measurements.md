# Field sort: cost and memory ceiling

Measured with `bench_search_perf`'s `sort criteria` matrix, which sorts over an already-open
index (so the `.tis` load — 98% of a cold per-request search — does not swamp the signal) using
the query `widetoken`, present in **every** generated doc. So the matched set is the whole corpus:
this is the worst case for a design that walks matches instead of the term dictionary.

    toolchain: rustc 1.97.1     command: bench_search_perf <N> <iters>
    N = 50k/500k at 10 iters, 200k at 30 iters

Criteria are chosen to hit each `SortKey` branch and each cardinality regime:

| criterion | branch | cardinality |
|---|---|---|
| `created_at_key` (epoch) | `Num` | near-unique — the production shape |
| `cat_key` (`i % 50`) | `Num` | 50 distinct — massive ties |
| `id` (`"REC-{i}"`, does not parse as i64) | `Text` | unique — the branch that allocates |
| `absent_key` | `Missing` | none |
| no sort | — | control |

## Memory: sort retains nothing at bounded paging

`hiwater` is retained + transient growth (the honest "how much RAM does this query need"), and it
has **zero run-to-run variance** — so these are exact numbers, not estimates within a noise band.

| N (matched) | paging | hiwater, no sort | hiwater, sorted | delta |
|---:|---|---:|---:|---:|
| 50k | top20 / top100 / deep 10k+20 | 4 896 KiB | 4 896 KiB | **+0** |
| 200k | top20 / top100 / deep 10k+20 | 19 584 KiB | 19 584 KiB | **+0** |
| 500k | top20 / top100 / deep 10k+20 | 78 336 KiB | 78 336 KiB | **+0** |
| 50k | unlimited | 41 489 KiB | 43 268 KiB | +1 779 KiB |
| 200k | unlimited | 166 087 KiB | 173 202 KiB | +7 115 KiB |
| 500k | unlimited | 415 380 KiB | 428 047 KiB | +12 667 KiB |

**The delta is identical (zero) for all four criteria**, including the `Text` one that allocates a
`String` per retained entry, and it stays zero as N grows 10×. That is the design check, not just
an implementation check: if the heap were not bounding, the delta would track N.

Zero rather than "small" because the heap keeps `offset + limit` entries — 20, 100, or 10 020 —
next to a matched-set map that is already megabytes. Even the deep-paging row's ~10 020 entries
never lift the high-water mark above the peak that scoring reached earlier in the same call.

The growth from 4 896 → 78 336 KiB across N is **pre-existing** and unrelated to sort: it is the
`HashMap` of matched docs that `eval` builds, and the no-sort control pays exactly the same.

### The unlimited case

`limit == usize::MAX` (the runner's `limit: 0`) makes the bound vacuous — the heap retains every
match. It is also a request for every document, so N whole-document hydrations dominate: 415 MB at
500k, ~830 B per doc, of which sort adds **12.7 MB (3%)**, about 26 B per matched doc.

No mitigation was applied. The number is small relative to a cost the caller already asked for,
and the alternatives (`Box<str>`, clamping K', a quickselect fallback) would each add a code path
to save 3% of a case nobody in the app issues. Revisit only if unbounded browse becomes real.

## Time: ~0.3–0.4 µs per matched doc

Cost of resolving one stored value, over and above the no-sort control at the same paging:

| N | `created_at_key` | `cat_key` | `id` (Text) | `absent_key` |
|---:|---:|---:|---:|---:|
| 50k, top20 | 0.21 µs/doc | 0.22 µs/doc | 0.22 µs/doc | −0.01 µs/doc |
| 200k, top20 | 0.32 µs/doc | 0.38 µs/doc | 0.30 µs/doc | −0.02 µs/doc |
| 500k, top20 | 0.37 µs/doc | 0.32 µs/doc | 0.38 µs/doc | 0.02 µs/doc |

Read these as ~0.3 µs with real spread, not as three significant figures: an earlier 50k run (same
binary, same arguments, 10 iters) put `cat_key` at 0.40 µs/doc against the 0.22 above. Latency
noise on this machine is ±24%, and these cells are differences between two noisy numbers, so their
spread is worse. The shape is what holds: **flat in N, same order for every criterion, free when
the field is absent.** Unlike the memory columns, no single cell here should be quoted alone.

For scale: a matched doc already costs ~0.65 µs at 200k
(128.9 ms / 200k for the no-sort control), so sort adds roughly 50% to the per-match cost **in the
worst case where every document matches**. The app's global search is normally narrowed by text
first, where N is orders of magnitude smaller and this is invisible.

`absent_key` is free — `ZslSegment::stored_value` resolves the field name against the segment's
`.fnm` first and returns `None` before touching the `.fdt`, so sorting by a field the index does
not have costs nothing rather than costing a full scan per doc.

The `Text` branch only separates from `Num` under `unlimited`, where it retains 500k `String`s:
1 383 ms against 1 282 ms for `created_at_key`. At bounded paging the two are indistinguishable.

## What was NOT built, and why the numbers say so

The deferred plan's ordered term walk would cost 1.6 s at 200k and 4.2 s at 500k for a near-unique
field, paid up front to enumerate the vocabulary regardless of how few docs match. Resolving per
matched doc costs 0.37 µs × N instead — and crucially scales with the MATCH count, so a query
returning twelve tickets pays for twelve docs. The two only converge when nearly every doc matches,
and even at 500k-matches-out-of-500k the walk is still ~7× more expensive.

A doc-values sidecar (`doc_id → i64`, mmapped) would cut the 0.37 µs to a near-zero array index,
but it needs building, maintaining, and staleness detection against writes from the legacy PHP
engine. Nothing measured here justifies that yet.

## No-regression gate (path that does not use sort)

Against `baseline-pre-sort.md`, captured on a clean tree at `b4bb7dd`:

1. **Exact result parity.** The `PARITY <label>: [(id, score)]` dumps for all 9 pre-existing query
   classes are **byte-for-byte identical**. Not "equivalent" — identical, including float scores.
2. **Exact memory parity.** `hiwater`, `transient` and `hits` match on **9/9** rows. These columns
   have zero run-to-run variance, so this is a real gate rather than a statistical one.
3. **Latency.** Every row came out 1–12% *faster*, which is machine state, not an improvement —
   the noise floor between separate runs on this machine is **24.4%**, so nothing below that can be
   claimed in either direction. What this row rules out is a gross regression, and there is none.
4. **Tests.** 293 before → 310 after (17 new), with no pre-existing test modified.

The non-sort path reaches `finalize_paged` through the same call it always did, now under a `match`
arm. What it additionally executes is one branch on an `Option` and a 32-byte-larger `QueryParams`,
**once per query** — below any achievable measurement resolution, which is why no benchmark here
claims to have measured it. Points 1 and 2 are the evidence; they are exact.

## Raw matrices

### 50k
```
sort by                    paging            p50 ms   hiwater KiB   transient KiB      hits
(none: relevance)          top20             21.342          4896            4880        20
(none: relevance)          top100            21.141          4896            4815       100
(none: relevance)          deep 10k+20       21.904          4896            4880        20
(none: relevance)          unlimited         67.338         41489             781     50000
created_at_key Num ~uniq   top20             31.684          4896            4880        20
created_at_key Num ~uniq   top100            32.979          4896            4815       100
created_at_key Num ~uniq   deep 10k+20       35.421          4896            4880        20
created_at_key Num ~uniq   unlimited         84.438         43268            2560     50000
cat_key Num 50 distinct    top20             32.312          4896            4880        20
cat_key Num 50 distinct    top100            32.558          4896            4815       100
cat_key Num 50 distinct    deep 10k+20       36.671          4896            4880        20
cat_key Num 50 distinct    unlimited         90.551         43268            2560     50000
id Text unique (allocs)    top20             32.228          4896            4880        20
id Text unique (allocs)    top100            32.477          4896            4815       100
id Text unique (allocs)    deep 10k+20       38.624          4896            4880        20
id Text unique (allocs)    unlimited         94.048         43268            2560     50000
absent_key Missing         top20             20.644          4896            4880        20
absent_key Missing         top100            21.973          4896            4815       100
absent_key Missing         deep 10k+20       23.706          4896            4880        20
absent_key Missing         unlimited         68.245         43268            2560     50000
```

### 200k
```
sort by                    paging            p50 ms   hiwater KiB   transient KiB      hits
(none: relevance)          top20            128.910         19584           19568        20
(none: relevance)          top100           123.792         19584           19503       100
(none: relevance)          deep 10k+20      127.000         19584           19568        20
(none: relevance)          unlimited        339.555        166087            3125    200000
created_at_key Num ~uniq   top20            192.467         19584           19568        20
created_at_key Num ~uniq   top100           191.352         19584           19503       100
created_at_key Num ~uniq   deep 10k+20      206.912         19584           19568        20
created_at_key Num ~uniq   unlimited        419.427        173202           10240    200000
cat_key Num 50 distinct    top20            204.070         19584           19568        20
cat_key Num 50 distinct    top100           183.572         19584           19503       100
cat_key Num 50 distinct    deep 10k+20      194.118         19584           19568        20
cat_key Num 50 distinct    unlimited        429.393        173202           10240    200000
id Text unique (allocs)    top20            188.365         19584           19568        20
id Text unique (allocs)    top100           188.169         19584           19503       100
id Text unique (allocs)    deep 10k+20      202.396         19584           19568        20
id Text unique (allocs)    unlimited        526.026        173202           10240    200000
absent_key Missing         top20            124.808         19584           19568        20
absent_key Missing         top100           124.151         19584           19503       100
absent_key Missing         deep 10k+20      127.332         19584           19568        20
absent_key Missing         unlimited        355.026        173202           10240    200000
```

### 500k
```
sort by                    paging            p50 ms   hiwater KiB   transient KiB      hits
(none: relevance)          top20            429.531         78336           78320        20
(none: relevance)          top100           432.931         78336           78255       100
(none: relevance)          deep 10k+20      426.257         78336           78320        20
(none: relevance)          unlimited        985.431        415380            7812    500000
created_at_key Num ~uniq   top20            616.957         78336           78320        20
created_at_key Num ~uniq   top100           600.296         78336           78255       100
created_at_key Num ~uniq   deep 10k+20      610.396         78336           78320        20
created_at_key Num ~uniq   unlimited       1281.806        428047           20480    500000
cat_key Num 50 distinct    top20            589.673         78336           78320        20
cat_key Num 50 distinct    top100           595.234         78336           78255       100
cat_key Num 50 distinct    deep 10k+20      603.879         78336           78320        20
cat_key Num 50 distinct    unlimited       1269.899        428047           20480    500000
id Text unique (allocs)    top20            619.460         78336           78320        20
id Text unique (allocs)    top100           608.667         78336           78255       100
id Text unique (allocs)    deep 10k+20      627.576         78336           78320        20
id Text unique (allocs)    unlimited       1382.644        428047           20480    500000
absent_key Missing         top20            438.332         78336           78320        20
absent_key Missing         top100           434.500         78336           78255       100
absent_key Missing         deep 10k+20      437.436         78336           78320        20
absent_key Missing         unlimited       1029.756        428047           20480    500000
```
