//! NATIVE engine benchmark: characterizes the sdsearch Rust engine on three workloads as a
//! function of index size N, reporting wall time AND the two memory numbers the PHP harness
//! cannot obtain:
//!   - `heap_peak_kb`: peak HEAP bytes attributed to the measured op alone (tracking global
//!     allocator; the counter is reset to steady-state right before the op). This is the
//!     number PHP's `memory_get_peak_usage()` cannot see, because the Rust heap lives outside
//!     the Zend memory manager.
//!   - `rss_peak_kb`: process peak RSS (VmHWM from /proc/self/status). Includes the mmap of the
//!     source segments; this is the number comparable to the PHP harness's RSS reading.
//!
//! This example does NOT measure Zend — it characterizes our engine only. The cross-engine
//! comparison (sdsearch extension vs Zend) lives in `tools/bench_compare.php`.
//!
//! Every workload starts from a COPY of the committed KB fixture (same base as
//! `optimize_bench.rs` / `perf_writer.php`), so the three engines share an identical starting
//! point. The ~20 base docs are negligible noise at N >= 1000. The measured op adds/searches
//! `N` synthetic docs with planted terms (see `gen_one`) so the three search classes have an
//! exact, size-independent doc_freq.
//!
//! Usage:
//!   cargo run -p sdsearch-core --release --example bench_engine -- rebuild <N> [cap]
//!   cargo run -p sdsearch-core --release --example bench_engine -- churn   <N> [cap]
//!   cargo run -p sdsearch-core --release --example bench_engine -- search  <N> [iters]
//!
//! Prints ONE JSON line per run, e.g.:
//!   {"engine":"native","workload":"rebuild","n":1000,"ms":12.34,"heap_peak_kb":..,
//!    "rss_peak_kb":..,"doc_count":1020}
//!   {"engine":"native","workload":"search","n":1000,"iters":50,"heap_peak_kb":..,"rss_peak_kb":..,
//!    "many":{"hits":1000,"top20":{"p50_ms":..,"p95_ms":..},"top100":{"p50_ms":..,"p95_ms":..}},
//!    "few":{...}, "none":{...}}
//! Search latency is measured at two realistic paging depths (top-20 and top-100); `hits` is the
//! TRUE match count for the class (an unlimited call, unmeasured).

use sdsearch_core::index::IndexReader;
use sdsearch_core::query::QueryParams;
use sdsearch_core::zsl::index::ZslIndex;
use sdsearch_core::zsl::runner::search_index;
use sdsearch_core::zsl::writer::{FieldKind, IndexWriter, WriterDoc, WriterField, WriterOpts};
use std::alloc::{GlobalAlloc, Layout, System};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

// ---- tracking allocator: measures live + peak heap bytes ----
//
// The atomic bookkeeping on every alloc/free is not free — it taxes the hot path, so a
// heap-tracked run's WALL TIME is NOT comparable to the extension's (which uses the plain
// system allocator). We therefore run natives in two passes: a TIME pass with TRACK=false (no
// atomics → honest ms, the "engine floor" without the FFI/JSON boundary) and a HEAP pass with
// TRACK=true (accurate heap, ms ignored). `BENCH_TRACK_HEAP=0` selects the time pass.
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static TRACK: AtomicBool = AtomicBool::new(true);

struct TrackingAlloc;

unsafe impl GlobalAlloc for TrackingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe {
            let p = System.alloc(layout);
            if !p.is_null() && TRACK.load(Ordering::Relaxed) {
                let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
                PEAK.fetch_max(live, Ordering::Relaxed);
            }
            p
        }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe {
            System.dealloc(ptr, layout);
            if TRACK.load(Ordering::Relaxed) {
                LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
            }
        }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        unsafe {
            let p = System.realloc(ptr, layout, new_size);
            if !p.is_null() && TRACK.load(Ordering::Relaxed) {
                let old = layout.size();
                if new_size >= old {
                    let live = LIVE.fetch_add(new_size - old, Ordering::Relaxed) + (new_size - old);
                    PEAK.fetch_max(live, Ordering::Relaxed);
                } else {
                    LIVE.fetch_sub(old - new_size, Ordering::Relaxed);
                }
            }
            p
        }
    }
}

/// whether this process is the HEAP pass (TRACK on) or the TIME pass (TRACK off). Read once from
/// `BENCH_TRACK_HEAP` (default: on). Startup allocations before this are minimal and precede any
/// `reset_peak`, so LIVE accounting stays consistent for the whole measured region.
fn init_track() -> bool {
    let on = std::env::var("BENCH_TRACK_HEAP").map_or(true, |v| v != "0");
    TRACK.store(on, Ordering::Relaxed);
    on
}

#[global_allocator]
static GLOBAL: TrackingAlloc = TrackingAlloc;

/// Anchors PEAK to the current live bytes, so the next PEAK read reflects only what is
/// allocated from here on (i.e. the measured op's own heap high-water mark above steady state).
fn reset_peak() {
    let live = LIVE.load(Ordering::Relaxed);
    PEAK.store(live, Ordering::Relaxed);
}

fn heap_peak_kb() -> u64 {
    PEAK.load(Ordering::Relaxed) as u64 / 1024
}

/// "heap" when the tracking allocator is on (heap number is accurate, ms is taxed), "time"
/// otherwise (ms is honest, heap is meaningless). The report keys native rows on this.
fn pass_label() -> &'static str {
    if TRACK.load(Ordering::Relaxed) {
        "heap"
    } else {
        "time"
    }
}

/// process peak RSS (VmHWM from /proc/self/status), in KB; 0 if unavailable.
fn rss_peak_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kb| kb.parse().ok())
        })
        .unwrap_or(0)
}

// ---- planted-term corpus (MUST stay in sync with tools/bench_compare.php) ----
//
// Every added doc's `body` carries COMMON in every doc (doc_freq == N → "many"); RARE is
// planted in ~RARE_K docs, spread out (doc_freq == RARE_K → "few"); MISSING is never emitted
// (doc_freq == 0 → "none"). The three query tokens are chosen so search cost is exact and
// size-independent across engines.
// Distinct 3-char prefixes (wid/spa/abs) so the PHP fuzzy path (prefix_len 3) can never
// cross-match one class token to another — the three classes stay exact on both engines.
const COMMON: &str = "widetoken";
const RARE: &str = "sparsetoken";
const MISSING: &str = "absenttoken";
const RARE_K: usize = 5;

const POOL: &[&str] = &[
    "printer", "network", "vpn", "login", "email", "server", "crash", "slow", "reset", "password",
    "access", "error", "update", "install", "config", "backup", "restore", "timeout", "license",
    "upgrade", "firewall", "router", "disk", "memory", "cpu",
];

/// True when doc `i` of an N-doc batch should carry the RARE token, spread over the batch so
/// its doc_freq is `RARE_K` (approximately, when N is not a multiple of RARE_K).
fn is_rare(i: usize, n: usize) -> bool {
    let step = (n / RARE_K).max(1);
    i.is_multiple_of(step) && i / step < RARE_K
}

/// ONE deterministic doc numbered `i`. `id_prefix` keeps churn-added docs in a distinct id
/// namespace from the rebuild docs. Body: 40 POOL tokens + COMMON (+ RARE for planted docs) +
/// `ref{i}`. Generated one at a time (never a whole-corpus Vec) so the measured heap reflects
/// the ENGINE, not the corpus. Matches `gen_one` in tools/bench_compare.php token-for-token.
fn gen_one(i: usize, n: usize, id_prefix: &str) -> WriterDoc {
    let np = POOL.len();
    // title/body draw ONLY from the fixed POOL (bounded vocabulary, like real text) — no unique
    // per-doc token, so the term dictionary stays realistic. The only unique-per-doc term is the
    // `id` keyword (as a real index of N records has N ids).
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
    body.push(' ');
    body.push_str(COMMON);
    if is_rare(i, n) {
        body.push(' ');
        body.push_str(RARE);
    }
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
                name: "id".into(),
                value: format!("{id_prefix}-{i}"),
                kind: FieldKind::Keyword,
                stored: true,
            },
        ],
    }
}

/// copies the committed KB fixture to a fresh temp dir (skips locks and `.sti`), returns it.
fn copy_kb_base(tag: &str) -> PathBuf {
    let src = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/zsl_index_kb"
    ));
    let dst = std::env::temp_dir().join(format!("sdsearch_bench_{}_{}", std::process::id(), tag));
    if dst.is_dir() {
        std::fs::remove_dir_all(&dst).ok();
    }
    std::fs::create_dir_all(&dst).expect("create temp dir");
    for entry in std::fs::read_dir(&src).expect("read KB fixture") {
        let p = entry.unwrap().path();
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        if name.contains("lock") || name.ends_with(".sti") {
            continue;
        }
        std::fs::copy(&p, dst.join(&name)).expect("copy fixture file");
    }
    dst
}

/// builds an N-doc index on top of a fresh KB base copy and OPTIMIZES it to a single segment
/// (production shape — the host runs `optimize()` per batch). Unmeasured setup for churn/search.
fn build_index(n: usize, cap: usize, tag: &str) -> PathBuf {
    let dir = copy_kb_base(tag);
    let opts = WriterOpts {
        max_buffered_docs: cap,
        ..WriterOpts::default()
    };
    let mut w = IndexWriter::open(&dir, opts).expect("open (build) failed");
    for i in 0..n {
        w.add_document(gen_one(i, n, "REC"))
            .expect("add_document failed");
    }
    w.optimize().expect("optimize failed");
    dir
}

/// `rebuild <N> [cap]`: measures producing a PRODUCTION-shaped index from scratch — copy the KB
/// base, add N docs, then `optimize()` (merge to a single segment, as the host does per batch).
/// The copy is unmeasured (filesystem, not heap); the writer open/add/optimize is the measured op.
fn run_rebuild(n: usize, cap: usize) {
    let dir = copy_kb_base("rebuild");
    reset_peak();
    let t0 = std::time::Instant::now();
    let opts = WriterOpts {
        max_buffered_docs: cap,
        ..WriterOpts::default()
    };
    let mut w = IndexWriter::open(&dir, opts).expect("open (build) failed");
    for i in 0..n {
        w.add_document(gen_one(i, n, "REC"))
            .expect("add_document failed");
    }
    w.optimize().expect("optimize failed");
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    // total LIVE docs (base + added) via a post-commit reader — CommitReport.doc_count is the
    // per-session count, not the index total (see index_writer.rs commit_inner).
    let doc_count = ZslIndex::open(&dir).map_or(0, |r| r.num_docs());
    println!(
        "{{\"engine\":\"native\",\"pass\":\"{}\",\"workload\":\"rebuild\",\"n\":{},\"cap\":{},\"ms\":{:.3},\"heap_peak_kb\":{},\"rss_peak_kb\":{},\"doc_count\":{}}}",
        pass_label(),
        n,
        cap,
        ms,
        heap_peak_kb(),
        rss_peak_kb(),
        doc_count
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// `churn <N> [cap]`: on a committed N-doc index, deletes the first 1% of docs (by global id)
/// and adds 1% fresh docs, then commits — the measured op. The initial build is unmeasured.
fn run_churn(n: usize, cap: usize) {
    let dir = build_index(n, cap, "churn");
    let one_pct = (n / 100).max(1);
    let doc_count_before = ZslIndex::open(&dir).expect("open reader").num_docs();

    reset_peak();
    let t0 = std::time::Instant::now();
    let opts = WriterOpts {
        max_buffered_docs: cap,
        ..WriterOpts::default()
    };
    let mut w = IndexWriter::open(&dir, opts).expect("open (churn) failed");
    for gid in 0..one_pct {
        w.delete_document(gid);
    }
    for i in 0..one_pct {
        w.add_document(gen_one(i, n, "CHURN"))
            .expect("add_document failed");
    }
    w.commit().expect("commit failed");
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let doc_count = ZslIndex::open(&dir).map_or(0, |r| r.num_docs());
    println!(
        "{{\"engine\":\"native\",\"pass\":\"{}\",\"workload\":\"churn\",\"n\":{},\"cap\":{},\"pct1\":{},\"ms\":{:.3},\"heap_peak_kb\":{},\"rss_peak_kb\":{},\"doc_count_before\":{},\"doc_count\":{}}}",
        pass_label(),
        n,
        cap,
        one_pct,
        ms,
        heap_peak_kb(),
        rss_peak_kb(),
        doc_count_before,
        doc_count
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// the free-text query params for `token`, IDENTICAL to what the PHP extension builds
/// (fuzzy 0.5 / prefix 3, wildcard min-prefix 0) — so the native and sdsearch search columns
/// measure the same query, minus the FFI/JSON boundary.
fn params_for(token: &str) -> QueryParams {
    QueryParams {
        text: token.to_string(),
        where_groups: vec![],
        in_groups: vec![],
        fuzzy_similarity: 0.5,
        fuzzy_prefix_len: 3,
        wildcard_min_prefix: 0,
        accent_insensitive: false,
        field_weights: std::collections::HashMap::new(),
        similarity: sdsearch_core::score::Similarity::Bm25,
    }
}

/// times `iters` `search_index` runs at `limit`, discards a warm-up, returns (p50, p95) in ms.
/// Reopens the index per call, exactly as SdSearch\Engine::search does, so the two are comparable.
fn time_search(dir: &Path, params: &QueryParams, limit: usize, iters: usize) -> (f64, f64) {
    let _ = search_index(dir, params, 0.0, limit); // warm-up (not sampled)
    let mut samples: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = std::time::Instant::now();
        let r = search_index(dir, params, 0.0, limit).expect("search failed");
        samples.push(t0.elapsed().as_secs_f64() * 1000.0);
        std::hint::black_box(r);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = |q: f64| samples[((samples.len() - 1) as f64 * q).round() as usize];
    (p(0.50), p(0.95))
}

/// `search <N> [iters]`: builds an N-doc index (unmeasured), then times the three query classes
/// at two realistic paging depths (top-20 and top-100). Reports the TRUE hit count per class
/// (one unlimited call, unmeasured) alongside the paged latencies.
fn run_search(n: usize, iters: usize) {
    let dir = build_index(n, 1000, "search");

    reset_peak();
    let mut parts: Vec<String> = Vec::new();
    for (label, token) in [("many", COMMON), ("few", RARE), ("none", MISSING)] {
        let params = params_for(token);
        let hits = search_index(&dir, &params, 0.0, 0).expect("count").len(); // true freq (unlimited)
        let (t20_50, t20_95) = time_search(&dir, &params, 20, iters);
        let (t100_50, t100_95) = time_search(&dir, &params, 100, iters);
        parts.push(format!(
            "\"{label}\":{{\"hits\":{hits},\"top20\":{{\"p50_ms\":{t20_50:.4},\"p95_ms\":{t20_95:.4}}},\
\"top100\":{{\"p50_ms\":{t100_50:.4},\"p95_ms\":{t100_95:.4}}}}}"
        ));
    }
    println!(
        "{{\"engine\":\"native\",\"pass\":\"{}\",\"workload\":\"search\",\"n\":{},\"iters\":{},\"heap_peak_kb\":{},\"rss_peak_kb\":{},{}}}",
        pass_label(),
        n,
        iters,
        heap_peak_kb(),
        rss_peak_kb(),
        parts.join(",")
    );
    std::fs::remove_dir_all(&dir).ok();
}

fn main() {
    init_track(); // TIME pass (BENCH_TRACK_HEAP=0) vs HEAP pass (default), before any measured op.
    let args: Vec<String> = std::env::args().collect();
    let workload = args.get(1).map_or("rebuild", String::as_str);
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1000);
    match workload {
        "rebuild" => run_rebuild(n, args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1000)),
        "churn" => run_churn(n, args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1000)),
        "search" => run_search(n, args.get(3).and_then(|s| s.parse().ok()).unwrap_or(50)),
        other => {
            eprintln!("unknown workload: {other} (expected rebuild|churn|search)");
            std::process::exit(2);
        }
    }
}
