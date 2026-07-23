//! DIAGNOSTIC (not part of the committed bench suite): measures per-query wall-clock (warm
//! median) and peak heap allocation for representative query classes over an optimized ZSL
//! index, and dumps (id, score) per query so a baseline-vs-change diff can prove identity
//! (#3/#5) or characterize divergence (#2). A counting global allocator makes the memory
//! measurement portable (no /proc, Windows-clean).
//!
//! Usage: cargo run -p sdsearch-core --release --example bench_search_perf -- [N] [iters]
//!   N default 200000, iters default 30 (enough samples for a meaningful p95).

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use sdsearch_core::query::{InGroup, QueryParams, build_query, search};
use sdsearch_core::score::Similarity;
use sdsearch_core::zsl::index::ZslIndex;
use sdsearch_core::zsl::runner::search_index;
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

fn reset_peak() {
    PEAK.store(CURRENT.load(Ordering::Relaxed), Ordering::Relaxed);
}
fn peak_since_reset() -> usize {
    PEAK.load(Ordering::Relaxed)
        .saturating_sub(CURRENT.load(Ordering::Relaxed))
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

    // query classes: (label, params)
    let cases: Vec<(&str, QueryParams)> = vec![
        ("short-wildcard 'vp'", params("vp", vec![])),
        ("multi-word 'vpn login'", params("vpn login", vec![])),
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
    ];

    println!("\n==== bench_search_perf: {n} docs, iters={iters} ====\n");
    println!(
        "{:<28} {:>10} {:>10} {:>14} {:>8}",
        "query", "p50 ms", "p95 ms", "peak KiB", "hits"
    );
    for (label, p) in &cases {
        let q = build_query(p).unwrap();
        let idx = ZslIndex::open(&dir).unwrap();
        // measure peak allocation for one query over an already-open index
        reset_peak();
        let hits = search(&idx, &q, 0.0, 20);
        let peak = peak_since_reset();
        // warm p50/p95 latency of the full per-request path (open + query), like production
        let (p50, p95) = percentiles_ms(iters, || {
            let h = search_index(&dir, p, 0.0, 20).unwrap();
            std::hint::black_box(&h);
        });
        println!(
            "{label:<28} {p50:>10.3} {p95:>10.3} {:>14} {:>8}",
            peak / 1024,
            hits.len()
        );
        // parity dump: (id, score) for baseline-vs-change diffing
        let mut dump: Vec<(usize, f32)> = hits.iter().map(|h| (h.id, h.score)).collect();
        dump.sort_by_key(|a| a.0);
        eprintln!("PARITY {label}: {dump:?}");
    }

    bench_finalize_strategies(iters);

    std::fs::remove_dir_all(&dir).ok();
}
