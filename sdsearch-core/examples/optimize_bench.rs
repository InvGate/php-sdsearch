//! OPTIMIZE baseline: measures the RAM and time cost of `IndexWriter::optimize()` (the
//! full-index merge) as a function of index size, to evaluate a future streaming merge.
//!
//! Two memory numbers are reported, because they answer different questions:
//!   - `heap_peak_kb`: peak HEAP bytes attributed to the optimize alone (a tracking global
//!     allocator; the counter is reset to steady-state right before optimize). This is the
//!     number a streaming merge is meant to bound — it ignores the mmap of the source segments.
//!   - `vmhwm_kb`: process peak RSS (VmHWM). Includes the mmap of the source segments, which a
//!     streaming merge does NOT reduce. Reported for context, not as the target metric.
//!
//! Flow per run: copy the KB fixture as base → stream N synthetic docs with `cap` (→ a
//! multi-segment index) → reset the heap counter → optimize() → report.
//!
//! Usage:
//!   cargo run -p sdsearch-core --release --example optimize_bench -- <N> [cap]
//!     N   number of docs to add on top of the base (default 2000)
//!     cap max_buffered_docs while building (default 1000 → ceil(N/cap) flushed segments)
//!
//! Prints ONE JSON line: {"n":..,"cap":..,"segments_before":..,"doc_count":..,
//!                        "optimize_ms":..,"heap_peak_kb":..,"vmhwm_kb":..}

use sdsearch_core::zsl::writer::{FieldKind, IndexWriter, WriterDoc, WriterField, WriterOpts};
use std::alloc::{GlobalAlloc, Layout, System};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

// ---- tracking allocator: measures live + peak heap bytes ----
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

struct TrackingAlloc;

unsafe impl GlobalAlloc for TrackingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc(layout);
        if !p.is_null() {
            let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = System.realloc(ptr, layout, new_size);
        if !p.is_null() {
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

#[global_allocator]
static GLOBAL: TrackingAlloc = TrackingAlloc;

/// Anchors PEAK to the current live bytes, so the next PEAK read reflects only what is
/// allocated from here on (i.e. the optimize's own heap high-water mark above steady state).
fn reset_peak() -> usize {
    let live = LIVE.load(Ordering::Relaxed);
    PEAK.store(live, Ordering::Relaxed);
    live
}

/// process peak RSS (VmHWM from /proc/self/status), in KB; 0 if unavailable.
fn vmhwm_kb() -> u64 {
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

/// copies the committed KB fixture to a fresh temp dir (skips locks and `.sti`), returns it.
fn copy_kb_base(n: usize, cap: usize) -> PathBuf {
    let src = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/zsl_index_kb"
    ));
    let dst = std::env::temp_dir().join(format!("sdsearch_optbench_{}_{}_{}", std::process::id(), n, cap));
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

/// N deterministic docs: title ~5 tokens, body ~40 tokens, id keyword (== perf_writer.php).
fn gen_docs(n: usize) -> Vec<WriterDoc> {
    const POOL: &[&str] = &[
        "printer", "network", "vpn", "login", "email", "server", "crash", "slow", "reset",
        "password", "access", "error", "update", "install", "config", "backup", "restore",
        "timeout", "license", "upgrade", "firewall", "router", "disk", "memory", "cpu",
    ];
    let np = POOL.len();
    (0..n)
        .map(|i| {
            let title = format!("ticket {i} {} {} issue{i}", POOL[i % np], POOL[(i * 3) % np]);
            let body: String = (0..40)
                .map(|j| POOL[(i * 7 + j * 5) % np])
                .collect::<Vec<_>>()
                .join(" ");
            let body = format!("{body} ref{i}");
            WriterDoc {
                fields: vec![
                    WriterField { name: "title".into(), value: title, kind: FieldKind::Text, stored: true },
                    WriterField { name: "body".into(), value: body, kind: FieldKind::Text, stored: true },
                    WriterField { name: "id".into(), value: format!("REC-{i}"), kind: FieldKind::Keyword, stored: true },
                ],
            }
        })
        .collect()
}

/// copies an arbitrary index dir to `dst` (skips lock files and `.sti`). Overwrites `dst`.
fn copy_index(src: &Path, dst: &Path) {
    if dst.is_dir() {
        std::fs::remove_dir_all(dst).ok();
    }
    std::fs::create_dir_all(dst).expect("create scratch dir");
    for entry in std::fs::read_dir(src).expect("read source index") {
        let p = entry.unwrap().path();
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        if name.contains("lock") || name.ends_with(".sti") {
            continue;
        }
        std::fs::copy(&p, dst.join(&name)).expect("copy index file");
    }
}

/// `optimize_bench existing <index_dir> <n_extra> <scratch_dir>`: copies a REAL index to a
/// disk-backed scratch, adds `n_extra` tiny docs (cap=1 → one segment each) to force a
/// multi-segment optimize, then measures optimize() over the whole corpus.
fn run_existing(args: &[String]) {
    let src = Path::new(args.get(2).expect("usage: optimize_bench existing <dir> <n_extra> <scratch>"));
    let n_extra: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2);
    let scratch = Path::new(args.get(4).expect("scratch dir required (use a disk-backed path)"));

    copy_index(src, scratch);

    {
        let opts = WriterOpts { max_buffered_docs: 1, ..WriterOpts::default() };
        let mut w = IndexWriter::open(scratch, opts).expect("open (build) failed");
        for d in gen_docs(n_extra) {
            w.add_document(d).expect("add_document failed");
        }
        w.commit().expect("commit failed");
    }

    let segments_before = sdsearch_core::zsl::segments::read_segment_infos(scratch)
        .expect("read segment infos")
        .len();

    reset_peak();
    let t0 = std::time::Instant::now();
    let w = IndexWriter::open(scratch, WriterOpts::default()).expect("open (optimize) failed");
    let report = w.optimize().expect("optimize failed");
    let optimize_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let heap_peak_kb = PEAK.load(Ordering::Relaxed) as u64 / 1024;

    println!(
        "{{\"source\":\"{}\",\"n_extra\":{},\"segments_before\":{},\"doc_count\":{},\"optimize_ms\":{:.2},\"heap_peak_kb\":{},\"vmhwm_kb\":{}}}",
        src.display(), n_extra, segments_before, report.doc_count, optimize_ms, heap_peak_kb, vmhwm_kb()
    );

    std::fs::remove_dir_all(scratch).ok();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("existing") {
        run_existing(&args);
        return;
    }
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let cap: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1000);

    let dir = copy_kb_base(n, cap);

    // ---- build: stream N docs into a multi-segment index (bounded-memory add path) ----
    {
        let opts = WriterOpts { max_buffered_docs: cap, ..WriterOpts::default() };
        let mut w = IndexWriter::open(&dir, opts).expect("open (build) failed");
        for d in gen_docs(n) {
            w.add_document(d).expect("add_document failed");
        }
        w.commit().expect("commit failed");
    }

    let segments_before = sdsearch_core::zsl::segments::read_segment_infos(&dir)
        .expect("read segment infos")
        .len();

    // ---- measure: optimize() only ----
    reset_peak();
    let t0 = std::time::Instant::now();
    let w = IndexWriter::open(&dir, WriterOpts::default()).expect("open (optimize) failed");
    let report = w.optimize().expect("optimize failed");
    let optimize_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let heap_peak_kb = PEAK.load(Ordering::Relaxed) as u64 / 1024;

    println!(
        "{{\"n\":{},\"cap\":{},\"segments_before\":{},\"doc_count\":{},\"optimize_ms\":{:.2},\"heap_peak_kb\":{},\"vmhwm_kb\":{}}}",
        n, cap, segments_before, report.doc_count, optimize_ms, heap_peak_kb, vmhwm_kb()
    );

    std::fs::remove_dir_all(&dir).ok();
}
