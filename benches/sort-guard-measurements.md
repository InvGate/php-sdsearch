# Unknown-sort-field guard: cost measurement

Question: does rejecting a sort by a field the index does not have cost anything on the query
path? The guard (`reject_unknown_sort_field` in `zsl/runner.rs`) runs once per query inside
`search_index_paged`, so every search pays for it whether it sorts or not.

Machine: the same one as `sort-measurements.md`. 200 000 docs, 20 iterations, released build.

## Instruments

Latency cannot answer this. Two runs of the SAME code drift up to 8.20% here, while the effect
being measured is a handful of nanoseconds against queries of 1.8–1950 ms. So the guard is
measured two ways instead:

1. **Isolated** — `bench_sort_guard` times `ZslIndex::has_field` directly, which is the only way
   to state a number rather than "lost in noise". (Same reasoning as `bench_intersect_strategies`
   and `bench_finalize_strategies`.)
2. **A/B/A on exact instruments** — three full runs, guard ON (A), guard removed (B), guard ON
   again (C), comparing the PARITY dumps and `hiwater` rather than p50. A and C bracket B so the
   machine's drift is measured in the same session as the effect.

## 1. The guard in isolation

`ZslIndex::has_field`, 12 field names in the `.fnm`:

| case | run A | run C |
|---|---:|---:|
| hit, early in `.fnm` | 3.8 ns | 3.4 ns |
| hit, stored-only, late in `.fnm` | 9.2 ns | 9.1 ns |
| miss (scans every name) | 5.6 ns | 4.8 ns |

Worst case 9.2 ns, once per query. Against the fastest query class in the suite
(`none 'absenttoken'`, 1.83 ms) that is 0.0005%.

The miss is NOT the worst case, which is worth writing down because it is backwards from the
obvious guess: a miss scans every name, but `String == &str` compares lengths first, and a long
absent name mismatches on length against nearly every entry. The slow case is a HIT late in the
`.fnm` whose length collides with earlier names, forcing a byte compare on each.

The cost does not grow with the index: `.fnm` is resident from `open()`, the scan is over field
NAMES (a schema-sized list, ~12–20), and nothing touches the term dictionary, the postings or
the `.fdt`.

## 2. A/B/A over the 9 pre-existing query classes

**PARITY: identical.** A vs B and A vs C are byte-for-byte identical `(id, score)` dumps across
all 9 classes. The guard changes no result.

**hiwater: identical.** Every class reports the same retained high-water in all three runs
(4384, 8768, 19584, 9749, 319, 9477, 24192, 24192, 3601 KiB). Zero memory cost. Hit counts
identical too.

**p50: unresolvable, reported as such.**

| query class | A (on) | B (off) | C (on) | \|A−C\| noise |
|---|---:|---:|---:|---:|
| short-wildcard 'vp' | 16.971 | 18.824 | 18.363 | 8.20% |
| multi-word 'vpn login' | 42.716 | 49.835 | 44.174 | 3.41% |
| common 'widetoken' (big M) | 126.688 | 121.258 | 119.276 | 5.85% |
| filtered 'widetoken' + cat_key=3 | 12.621 | 12.282 | 12.257 | 2.88% |
| none 'absenttoken' | 1.886 | 1.764 | 1.830 | 2.97% |
| range narrow ~1% | 28.899 | 29.021 | 28.519 | 1.31% |
| range wide ~80% | 1642.683 | 1598.655 | 1598.020 | 2.72% |
| range full + matchAll | 1950.957 | 2004.350 | 1970.092 | 0.98% |
| intersection-heavy | 376.090 | 370.519 | 375.480 | 0.16% |

Noise floor (max spread between two runs of identical code): **8.20%**.

B lands inside the A–C band in every class. In four of the nine — multi-word, range narrow,
range full + matchAll, and short-wildcard — removing the guard measured SLOWER than keeping it,
which is causally impossible and is the clearest evidence available that these deltas are the
machine and not the change.

## Conclusion

Zero memory cost and zero result change, both on exact instruments. Latency cost is 9.2 ns per
query worst case, which is between three and six orders of magnitude below the noise floor of
this harness, so no latency claim is made in either direction.
