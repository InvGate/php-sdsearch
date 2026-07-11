//! CI guard for the bounded-memory streaming k-way segment merge.
//!
//! The whole point of [`merge::merge_segments_streaming`] is that its peak HEAP does NOT grow
//! with a near-stopword term's `doc_freq` (nor with total text volume): it streams postings
//! per-doc through temp files, whereas the batch oracle [`merge::merge_segments`] materializes
//! the ENTIRE merged segment in RAM (a `term_map` holding every term's postings/positions for
//! every doc, plus the full stored column, plus the full `.cfs` byte buffer).
//!
//! Until now the only thing measuring this property was the manual `examples/optimize_bench.rs`
//! binary; there was no automated test. This file is that test.
//!
//! Design: it drives BOTH public merge entry points over the SAME multi-segment input and
//! compares their peak heap using a tracking global allocator. The assertion is DIFFERENTIAL
//! (streaming peak << batch peak over identical input), which gives it teeth by construction and
//! keeps it robust across debug/release (release strips debug_asserts and shifts alloc patterns,
//! but the batch path still builds the whole segment in RAM while streaming does not).
//!
//! NOTE: a `#[global_allocator]` in an integration-test file only affects THIS test binary — each
//! `tests/*.rs` file is compiled as its own crate — so it does not pollute other tests.

use sdsearch_core::zsl::segments::read_segment_infos;
use sdsearch_core::zsl::writer::merge;
use sdsearch_core::zsl::writer::{FieldKind, IndexWriter, WriterDoc, WriterField, WriterOpts};
use std::alloc::{GlobalAlloc, Layout, System};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

// ---- tracking allocator: measures live + peak heap bytes (same pattern as optimize_bench) ----
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

/// Anchors PEAK to the current live bytes, so the next PEAK read reflects only the heap
/// high-water mark ABOVE this steady state (i.e. the merge's own allocations).
fn reset_peak() {
    let live = LIVE.load(Ordering::Relaxed);
    PEAK.store(live, Ordering::Relaxed);
}

/// Copies the committed KB fixture to a fresh temp dir (skips lock files and `.sti`), returns it.
/// `IndexWriter::open` requires an existing base index (it reads the generation), so we start
/// from the real KB fixture — exactly as `examples/optimize_bench.rs` and the merge unit tests do.
fn copy_kb_base() -> PathBuf {
    let src = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/zsl_index_kb"
    ));
    let dst = std::env::temp_dir().join(format!(
        "sdsearch_merge_mem_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
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

/// N docs that all share ONE "stopword" token (so its `doc_freq == N` in both `title` and
/// `body` — the near-stopword shape the batch path would hold fully resident) PLUS a per-doc
/// unique token, so the postings genuinely differ doc to doc.
fn gen_hot_docs(n: usize) -> Vec<WriterDoc> {
    const POOL: &[&str] = &[
        "printer", "network", "vpn", "login", "email", "server", "crash", "slow", "reset",
        "password", "access", "error", "update", "install", "config", "backup", "restore",
        "timeout", "license", "upgrade", "firewall", "router", "disk", "memory", "cpu",
    ];
    let np = POOL.len();
    (0..n)
        .map(|i| {
            let title = format!("stopword ticket{i} {}", POOL[i % np]);
            let filler: String = (0..30)
                .map(|j| POOL[(i * 7 + j * 5) % np])
                .collect::<Vec<_>>()
                .join(" ");
            let body = format!("stopword {filler} ref{i}");
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
                        value: format!("REC-{i}"),
                        kind: FieldKind::Keyword,
                        stored: true,
                    },
                ],
            }
        })
        .collect()
}

#[test]
fn streaming_merge_peak_heap_is_far_below_batch() {
    // ---- build a multi-segment index with a large-doc_freq term ----
    // N deliberately in the hundreds+ so "materialize the whole term / whole segment" (batch)
    // vs "stream one posting" (streaming) is a clear heap difference, not allocator noise.
    const N: usize = 1500;
    let dir = copy_kb_base();

    {
        // small buffer => ceil(N / cap) flushed segments => a genuine k-way merge.
        let opts = WriterOpts {
            max_buffered_docs: 50,
            ..WriterOpts::default()
        };
        let mut w = IndexWriter::open(&dir, opts).expect("open (build) failed");
        for d in gen_hot_docs(N) {
            w.add_document(d).expect("add_document failed");
        }
        w.commit().expect("commit failed");
    }

    let infos = read_segment_infos(&dir).expect("read segment infos");
    assert!(
        infos.len() >= 3,
        "expected a multi-segment index for a meaningful k-way merge, got {}",
        infos.len()
    );
    let refs: Vec<(String, i64)> = infos.iter().map(|s| (s.name.clone(), s.del_gen)).collect();

    // ---- measure STREAMING peak heap over the input ----
    // (source segments are mmap'd, not heap-allocated, so they don't count toward PEAK.)
    reset_peak();
    let streaming_docs =
        merge::merge_segments_streaming(&dir, "_stream", &refs).expect("streaming merge failed");
    let streaming_peak = PEAK.load(Ordering::Relaxed);
    // the streaming merge writes its output to disk; drop it so it can't skew the batch run.
    std::fs::remove_file(dir.join("_stream.cfs")).ok();

    // ---- measure BATCH peak heap over the SAME input ----
    reset_peak();
    let batch = merge::merge_segments(&dir, "_batch", &refs).expect("batch merge failed");
    let batch_peak = PEAK.load(Ordering::Relaxed);
    let batch_docs = batch.doc_count;
    // hold cfs_bytes only until the peak is read, then release.
    drop(batch);

    // sanity: both paths merged the same corpus (KB fixture's 20 docs + N added).
    assert_eq!(streaming_docs, batch_docs, "doc counts must agree");
    assert_eq!(streaming_docs, N + 20, "unexpected merged doc count");

    // ---- differential assertion ----
    // The batch path holds the entire merged segment resident (term_map with every posting,
    // the full stored column, and the full .cfs byte buffer), so its peak scales with total
    // content. The streaming path keeps only small per-field/index blocks resident. Measured
    // (see report) the batch peak is an order of magnitude larger; we require a conservative 3x
    // so the guard is robust across debug/release yet still fails loudly if the streaming merge
    // ever regresses to materializing the term (which would make the two peaks converge).
    eprintln!(
        "streaming_peak = {} bytes ({} KB), batch_peak = {} bytes ({} KB), ratio = {:.2}x",
        streaming_peak,
        streaming_peak / 1024,
        batch_peak,
        batch_peak / 1024,
        batch_peak as f64 / streaming_peak.max(1) as f64,
    );
    assert!(
        streaming_peak.saturating_mul(3) < batch_peak,
        "streaming merge peak heap ({streaming_peak} B) is not far below the batch peak \
         ({batch_peak} B): the bounded-memory guarantee appears to have regressed",
    );

    std::fs::remove_dir_all(&dir).ok();
}
