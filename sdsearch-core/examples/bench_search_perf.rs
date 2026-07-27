//! DIAGNOSTIC (not part of the committed bench suite): measures per-query wall-clock (warm
//! median) and peak heap allocation for representative query classes over an optimized ZSL
//! index, and dumps (id, score) per query so a baseline-vs-change diff can prove identity
//! (#3/#5) or characterize divergence (#2). A counting global allocator makes the memory
//! measurement portable (no /proc, Windows-clean).
//!
//! Usage: cargo run -p sdsearch-core --release --example bench_search_perf -- [N] [iters]
//!   N default 200000, iters default 30 (enough samples for a meaningful p95).

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use sdsearch_core::query::{
    InGroup, MatchAllFilter, QueryParams, RangeFilter, build_query, intersect_allow,
    match_all_allow_list, range_allow_list, search_with_weights_paged,
};
use sdsearch_core::score::Similarity;
use sdsearch_core::zsl::index::ZslIndex;
use sdsearch_core::zsl::runner::search_index_paged;
use sdsearch_core::zsl::writer::{FieldKind, IndexWriter, WriterDoc, WriterField, WriterOpts};

// ---- counting allocator: current + peak bytes ----
struct Counting;
static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            let cur = CURRENT.fetch_add(l.size(), Ordering::Relaxed) + l.size();
            PEAK.fetch_max(cur, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        CURRENT.fetch_sub(l.size(), Ordering::Relaxed);
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

static CURRENT_AT_RESET: AtomicUsize = AtomicUsize::new(0);

fn reset_peak() {
    let cur = CURRENT.load(Ordering::Relaxed);
    PEAK.store(cur, Ordering::Relaxed);
    CURRENT_AT_RESET.store(cur, Ordering::Relaxed);
}
/// Transient bytes: allocated-then-freed since the last reset (the peak minus whatever is
/// live right now). Right metric for rehash waste; wrong metric for "how much RAM does this
/// query need", since it excludes everything still retained at measurement time.
fn peak_since_reset() -> usize {
    PEAK.load(Ordering::Relaxed)
        .saturating_sub(CURRENT.load(Ordering::Relaxed))
}
/// True high-water growth: the peak minus whatever was ALREADY live at the last reset. This is
/// retained + transient — the real answer to "how much RAM does this query need".
fn hiwater_since_reset() -> usize {
    PEAK.load(Ordering::Relaxed)
        .saturating_sub(CURRENT_AT_RESET.load(Ordering::Relaxed))
}

const POOL: &[&str] = &[
    "printer", "network", "vpn", "login", "email", "server", "crash", "slow", "reset", "password",
    "access", "error", "update", "install", "config", "backup", "restore", "timeout", "license",
    "upgrade", "firewall", "router", "disk", "memory", "cpu",
];

fn gen_one(i: usize) -> WriterDoc {
    let np = POOL.len();
    let title = format!(
        "ticket {} {} {}",
        POOL[i % np],
        POOL[(i * 3) % np],
        POOL[(i * 7) % np]
    );
    let mut body: String = (0..40)
        .map(|j| POOL[(i * 7 + j * 5) % np])
        .collect::<Vec<_>>()
        .join(" ");
    body.push_str(" widetoken"); // common token in EVERY doc: gives the filtered/common cases hits
    WriterDoc {
        fields: vec![
            WriterField {
                name: "title".into(),
                value: title,
                kind: FieldKind::Text,
                stored: true,
            },
            WriterField {
                name: "body".into(),
                value: body,
                kind: FieldKind::Text,
                stored: true,
            },
            WriterField {
                name: "cat_key".into(),
                value: format!("{}", i % 50),
                kind: FieldKind::Keyword,
                stored: true,
            },
            WriterField {
                name: "created_at_key".into(),
                // fixed-width 10 digits for i up to ~2.9e9, so byte order == numeric order
                value: format!("{}", 1_700_000_000u64 + i as u64),
                kind: FieldKind::Keyword,
                stored: true,
            },
            WriterField {
                name: "id".into(),
                value: format!("REC-{i}"),
                kind: FieldKind::Keyword,
                stored: true,
            },
        ],
    }
}

fn copy_kb_base() -> PathBuf {
    let src = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/zsl_index_kb"
    ));
    let dst = std::env::temp_dir().join(format!("sdsearch_benchperf_{}", std::process::id()));
    if dst.is_dir() {
        std::fs::remove_dir_all(&dst).ok();
    }
    std::fs::create_dir_all(&dst).unwrap();
    for e in std::fs::read_dir(&src).unwrap() {
        let p = e.unwrap().path();
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        if name.contains("lock") || name.ends_with(".sti") {
            continue;
        }
        std::fs::copy(&p, dst.join(&name)).unwrap();
    }
    dst
}

fn params(text: &str, in_groups: Vec<InGroup>) -> QueryParams {
    QueryParams {
        text: text.to_string(),
        where_groups: vec![],
        in_groups,
        fuzzy_similarity: 0.5,
        fuzzy_prefix_len: 3,
        wildcard_min_prefix: 2,
        accent_insensitive: false,
        field_weights: HashMap::new(),
        similarity: Similarity::Bm25,
        range_filters: vec![],
        match_all: vec![],
    }
}

fn with_range(mut p: QueryParams, field: &str, lo: Option<&str>, hi: Option<&str>) -> QueryParams {
    p.range_filters.push(RangeFilter {
        field: field.to_string(),
        lower: lo.map(str::to_string),
        upper: hi.map(str::to_string),
    });
    p
}

fn with_match_all(mut p: QueryParams, field: &str, text: &str) -> QueryParams {
    p.match_all.push(MatchAllFilter {
        field: field.to_string(),
        text: text.to_string(),
    });
    p
}

/// warm (p50, p95) in ms over `iters` timed runs after 3 warm-up runs.
fn percentiles_ms(iters: usize, mut f: impl FnMut()) -> (f64, f64) {
    for _ in 0..3 {
        f();
    }
    let mut s: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        s.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| s[(((s.len() as f64) * p) as usize).min(s.len() - 1)];
    (pct(0.50), pct(0.95))
}

/// #5 isolation: does the hand-rolled `select_nth` top-k actually beat the stdlib's adaptive
/// `sort_by`? Times both strategies over synthetic scored data for several (M, limit) pairs.
/// The `clone` column is the per-iter copy floor (equal for both strategies); the algorithm
/// cost is (strategy − clone). If select_nth is not consistently below full-sort for realistic
/// (M, limit), #5 is not worth keeping.
fn bench_finalize_strategies(iters: usize) {
    let cmp = |a: &(usize, f32), b: &(usize, f32)| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    };
    println!("\n---- finalize: full-sort vs select_nth top-k (p50 ms, incl. clone floor) ----");
    println!(
        "{:<18} {:>10} {:>10} {:>12}",
        "M / limit", "clone", "full-sort", "select_nth"
    );
    for &m in &[1_000usize, 50_000, 200_000, 1_000_000] {
        // deterministic pseudo-scores (Knuth multiplicative hash) — no rng needed.
        let base: Vec<(usize, f32)> = (0..m)
            .map(|i| (i, (i.wrapping_mul(2_654_435_761) % 100_003) as f32))
            .collect();
        for &limit in &[20usize, 1_000] {
            if limit >= m {
                continue;
            }
            let clone_p50 = percentiles_ms(iters, || {
                let v = base.clone();
                std::hint::black_box(&v);
            })
            .0;
            let full = percentiles_ms(iters, || {
                let mut v = base.clone();
                v.sort_by(cmp);
                v.truncate(limit);
                std::hint::black_box(&v);
            })
            .0;
            let sel = percentiles_ms(iters, || {
                let mut v = base.clone();
                v.select_nth_unstable_by(limit, cmp);
                v.truncate(limit);
                v.sort_by(cmp);
                std::hint::black_box(&v);
            })
            .0;
            println!(
                "{:<18} {clone_p50:>10.3} {full:>10.3} {sel:>12.3}",
                format!("{m} / {limit}")
            );
        }
    }
}

/// mirrors `query.rs`'s private `intersect_in_place` helper (kept local here since that fn is
/// not `pub`): retains on the smaller operand and probes the larger, so the cost is O(min(|a|,
/// |b|)) probes with no third-set allocation and no rehashing of survivors.
fn intersect_in_place_local(a: HashSet<usize>, b: HashSet<usize>) -> HashSet<usize> {
    let (mut keep, probe) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    keep.retain(|d| probe.contains(d));
    keep
}

/// #2 isolation: the macro bench (query classes above) couldn't resolve whether
/// `intersect_in_place` actually beats the naive `prev.intersection(&other).copied().collect()`
/// it replaced — both range/matchAll query classes with a single filter never call it at all,
/// and run-to-run noise on unrelated rows (±20-30%) dwarfed the deltas on the rows that do. This
/// synthetic head-to-head isolates just the two intersection strategies over `HashSet<usize>`
/// pairs shaped like what the engine actually intersects: same-size pairs, and the asymmetric
/// (small, large) / (large, small) pairs the helper is specifically designed for. Order matters:
/// every real call site intersects as `prev.intersection(&x)`, i.e. always iterates the FIRST
/// operand — cheap when `prev` is the smaller set, expensive when it is the larger one.
/// `intersect_in_place` picks the smaller operand regardless of position, so it should differ
/// from naive only on the asymmetric rows. Both strategies consume their inputs, so a fresh
/// clone of both sets is built every iteration; the `clone` column is that per-iter floor
/// (identical cost for both strategies), so the real algorithm cost is (strategy − clone).
/// Overlap is deterministic (`b` starts at `a.len() / 2`) — no randomness — so result sizes are
/// stable across runs.
fn bench_intersect_strategies(iters: usize) {
    println!(
        "\n---- intersect: prev.intersection(..).collect() vs intersect_in_place (p50 ms, incl. clone floor) ----"
    );
    println!(
        "{:<18} {:>10} {:>12} {:>20}",
        "|a| / |b|", "clone", "naive", "intersect_in_place"
    );
    for &(na, nb) in &[
        (1_000usize, 1_000usize),
        (100_000, 100_000),
        (1_000, 100_000),
        (100_000, 1_000),
    ] {
        // deterministic overlap: b starts at a.len()/2, so results are stable across runs.
        let a: HashSet<usize> = (0..na).collect();
        let b: HashSet<usize> = (na / 2..na / 2 + nb).collect();

        let clone_p50 = percentiles_ms(iters, || {
            let pa = a.clone();
            let pb = b.clone();
            std::hint::black_box(&pa);
            std::hint::black_box(&pb);
        })
        .0;
        let naive_p50 = percentiles_ms(iters, || {
            let pa = a.clone();
            let pb = b.clone();
            let out: HashSet<usize> = pa.intersection(&pb).copied().collect();
            std::hint::black_box(&out);
        })
        .0;
        let new_p50 = percentiles_ms(iters, || {
            let pa = a.clone();
            let pb = b.clone();
            let out = intersect_in_place_local(pa, pb);
            std::hint::black_box(&out);
        })
        .0;
        println!(
            "{:<18} {clone_p50:>10.3} {naive_p50:>12.3} {new_p50:>20.3}",
            format!("{na} / {nb}")
        );
    }
}

/// #3 isolation: does sizing the `HashSet<usize>` up front (`HashSet::with_capacity(n)`) actually
/// avoid rehash-cycle cost versus growing it unsized (`HashSet::new()`) through the same insert
/// loop? Mirrors exactly what `range_allow_list` does — N distinct ids inserted one at a time —
/// so this isolates just the allocation strategy, not the surrounding term/posting walk. Reports
/// both wall-clock (p50 ms) and peak bytes via the counting allocator: the peak column is the
/// more interesting one, since avoiding rehashing removes the transient where hashbrown holds
/// both the old and new tables live at once. Ids are deterministic (`0..n`), no randomness, so
/// results are stable across runs.
fn bench_reserve_strategies(iters: usize) {
    println!(
        "\n---- reserve: HashSet::new() vs HashSet::with_capacity(n) (p50 ms + peak bytes) ----"
    );
    println!(
        "{:<10} {:>12} {:>12} {:>14} {:>14}",
        "n", "new p50", "with_cap p50", "new peak B", "with_cap peak B"
    );
    for &n in &[1_000usize, 50_000, 200_000] {
        let new_p50 = percentiles_ms(iters, || {
            let mut docs: HashSet<usize> = HashSet::new();
            for id in 0..n {
                docs.insert(id);
            }
            std::hint::black_box(&docs);
        })
        .0;
        reset_peak();
        let mut docs: HashSet<usize> = HashSet::new();
        for id in 0..n {
            docs.insert(id);
        }
        std::hint::black_box(&docs);
        let new_peak = peak_since_reset();
        drop(docs);

        let cap_p50 = percentiles_ms(iters, || {
            let mut docs: HashSet<usize> = HashSet::with_capacity(n);
            for id in 0..n {
                docs.insert(id);
            }
            std::hint::black_box(&docs);
        })
        .0;
        reset_peak();
        let mut docs: HashSet<usize> = HashSet::with_capacity(n);
        for id in 0..n {
            docs.insert(id);
        }
        std::hint::black_box(&docs);
        let cap_peak = peak_since_reset();
        drop(docs);

        println!("{n:<10} {new_p50:>12.3} {cap_p50:>12.3} {new_peak:>14} {cap_peak:>14}");
    }
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200_000);
    let iters: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);

    eprintln!("building + optimizing a {n}-doc index (unmeasured setup)…");
    let dir = copy_kb_base();
    {
        let opts = WriterOpts {
            max_buffered_docs: 1000,
            ..WriterOpts::default()
        };
        let mut w = IndexWriter::open(&dir, opts).unwrap();
        for i in 0..n {
            w.add_document(gen_one(i)).unwrap();
        }
        w.optimize().unwrap();
    }

    // range bounds derived from gen_one's created_at_key = 1_700_000_000 + i
    let base_ts = 1_700_000_000u64;
    let lo_narrow = format!("{}", base_ts + (n as u64) / 2);
    let hi_narrow = format!("{}", base_ts + (n as u64) / 2 + (n as u64) / 100);
    let lo_wide = format!("{base_ts}");
    let hi_wide = format!("{}", base_ts + (n as u64) * 4 / 5);
    // two overlapping ~10%-wide windows: [40%,50%) and [45%,55%) — a 5% overlap. Each range's
    // term walk only touches ~10% of the dictionary (cheap), but created_at_key is 1 doc/term,
    // so each resulting set is ~10% of the corpus — large enough that intersecting the two
    // (inside range_allow_list) and then intersecting against the matchAll set (via
    // intersect_allow) is real hash-set work, not a no-op on a handful of ids.
    let lo_ov1 = format!("{}", base_ts + (n as u64) * 40 / 100);
    let hi_ov1 = format!("{}", base_ts + (n as u64) * 50 / 100);
    let lo_ov2 = format!("{}", base_ts + (n as u64) * 45 / 100);
    let hi_ov2 = format!("{}", base_ts + (n as u64) * 55 / 100);

    // query classes: (label, params)
    let cases: Vec<(&str, QueryParams)> = vec![
        ("short-wildcard 'vp'", params("vp", vec![])),
        ("multi-word 'vpn login'", params("vpn login", vec![])),
        // CONTROL for this plan: same text, no filters. Must not regress.
        ("common 'widetoken' (big M)", params("widetoken", vec![])),
        (
            "filtered 'widetoken' + cat_key=3",
            params(
                "widetoken",
                vec![InGroup {
                    field: "cat_key".into(),
                    values: vec!["3".into()],
                }],
            ),
        ),
        ("none 'absenttoken'", params("absenttoken", vec![])),
        (
            "range narrow ~1%",
            with_range(
                params("widetoken", vec![]),
                "created_at_key",
                Some(&lo_narrow),
                Some(&hi_narrow),
            ),
        ),
        (
            "range wide ~80%",
            with_range(
                params("widetoken", vec![]),
                "created_at_key",
                Some(&lo_wide),
                Some(&hi_wide),
            ),
        ),
        (
            "range full + matchAll",
            with_match_all(
                with_range(params("widetoken", vec![]), "created_at_key", None, None),
                "body",
                "widetoken",
            ),
        ),
        (
            "intersection-heavy",
            with_match_all(
                with_match_all(
                    with_range(
                        with_range(
                            params("vpn login", vec![]),
                            "created_at_key",
                            Some(&lo_ov1),
                            Some(&hi_ov1),
                        ),
                        "created_at_key",
                        Some(&lo_ov2),
                        Some(&hi_ov2),
                    ),
                    "body",
                    "vpn slow",
                ),
                "body",
                "update timeout",
            ),
        ),
    ];

    println!("\n==== bench_search_perf: {n} docs, iters={iters} ====\n");
    println!(
        "{:<28} {:>10} {:>10} {:>14} {:>14} {:>8}",
        "query", "p50 ms", "p95 ms", "hiwater KiB", "transient KiB", "hits"
    );
    for (label, p) in &cases {
        let q = build_query(p).unwrap();
        let idx = ZslIndex::open(&dir).unwrap();
        // measure peak allocation for one query over an already-open index
        // peak must cover allow-list construction, not just scoring — `search()` bypasses restrict
        reset_peak();
        let restrict = intersect_allow(
            range_allow_list(&idx, &p.range_filters),
            match_all_allow_list(&idx, &p.match_all),
        );
        let outcome = search_with_weights_paged(
            &idx,
            &q,
            &p.field_weights,
            p.similarity,
            0.0,
            0,
            20,
            None,
            restrict.as_ref(),
        );
        let hiwater = hiwater_since_reset();
        let transient = peak_since_reset();
        let hits = outcome.hits;
        // warm p50/p95 latency of the full per-request path (open + query), like production
        let (p50, p95) = percentiles_ms(iters, || {
            let h = search_index_paged(&dir, p, 0.0, 0, 20, None).unwrap();
            std::hint::black_box(&h);
        });
        println!(
            "{label:<28} {p50:>10.3} {p95:>10.3} {:>14} {:>14} {:>8}",
            hiwater / 1024,
            transient / 1024,
            hits.len()
        );
        // parity dump: (id, score) for baseline-vs-change diffing
        let mut dump: Vec<(usize, f32)> = hits.iter().map(|h| (h.id, h.score)).collect();
        dump.sort_by_key(|a| a.0);
        eprintln!("PARITY {label}: {dump:?}");
    }

    bench_finalize_strategies(iters);
    bench_intersect_strategies(iters);
    bench_reserve_strategies(iters);

    std::fs::remove_dir_all(&dir).ok();
}
