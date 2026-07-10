//! Streaming append: open → add_document* → commit via IndexWriter, with
//! configurable `max_buffered_docs` (flush to MULTIPLE segments in one commit). Used by
//! golden_writer.php (multi-segment interchange) and perf_writer.php (bounded memory).
//!
//! Usage:
//!   cargo run -p sdsearch-core --release --example stream_writer -- <index_dir> <docs.json> [cap]
//!
//! docs.json: { "docs": [ { "fields": [
//!   {"name":"title","value":"...","kind":"text","stored":true}, ... ] } ] }
//!   kind ∈ {"text","keyword","unindexed"} (default stored=true); cap default 1000.
//!
//! delete subcommand: deletes doc(s) from the base snapshot by global_doc_id (the same
//! id `Zend_Search_Lucene::delete()` uses — 0-based position over Σ maxDoc of the base
//! segments, IN segments_N order; NOT a stored-field value).
//!
//!   cargo run -p sdsearch-core --release --example stream_writer -- delete <index_dir> <gid1,gid2,...>
//!
//! optimize subcommand: opens the IndexWriter and runs optimize() (merge everything into
//! 1 compacted segment, collapsing pending deletes).
//!
//!   cargo run -p sdsearch-core --release --example stream_writer -- optimize <index_dir>
//!
//! hold-lock subcommand: opens the IndexWriter (takes the write-lock), prints
//! LOCK_ACQUIRED, holds it for hold_ms and exits 0; if the lock is already held it prints
//! LOCK_WOULDBLOCK and exits 3. Used by the inter-process test zsl_interprocess_lock.rs.
//!
//!   cargo run -p sdsearch-core --release --example stream_writer -- hold-lock <index_dir> <hold_ms>

use sdsearch_core::zsl::writer::{FieldKind, IndexWriter, WriterDoc, WriterField, WriterOpts};
use std::path::Path;

#[derive(serde::Deserialize)]
struct Spec {
    docs: Vec<DocSpec>,
}
#[derive(serde::Deserialize)]
struct DocSpec {
    fields: Vec<FieldSpec>,
}
#[derive(serde::Deserialize)]
struct FieldSpec {
    name: String,
    value: String,
    kind: String,
    #[serde(default = "default_true")]
    stored: bool,
}
fn default_true() -> bool {
    true
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("delete") {
        run_delete(&args);
        return;
    }
    if args.get(1).map(String::as_str) == Some("optimize") {
        run_optimize(&args);
        return;
    }
    if args.get(1).map(String::as_str) == Some("hold-lock") {
        run_hold_lock(&args);
        return;
    }
    run_stream(&args);
}

/// `stream_writer delete <index_dir> <gid1,gid2,...>`: opens the `IndexWriter`, marks each gid
/// as deleted (`delete_document`) and commits. Prints `{"deleted":N,"generation":G}`.
fn run_delete(args: &[String]) {
    let index_dir = args.get(2).expect("usage: stream_writer delete <index_dir> <gid1,gid2,...>");
    let gids_raw = args.get(3).expect("usage: stream_writer delete <index_dir> <gid1,gid2,...>");
    let gids: Vec<usize> = gids_raw
        .split(',')
        .map(|s| s.trim().parse().expect("invalid gid (expected an integer)"))
        .collect();

    let mut w = IndexWriter::open(Path::new(index_dir), WriterOpts::default()).expect("open failed");
    for gid in &gids {
        w.delete_document(*gid);
    }
    let report = w.commit().expect("commit failed");

    println!(
        "{{\"deleted\":{},\"generation\":{}}}",
        gids.len(),
        report.generation
    );
}

/// `stream_writer optimize <index_dir>`: opens the IndexWriter and runs optimize() (merge
/// everything into 1 compacted segment). Prints `{"optimized":true,"generation":G,"doc_count":D}`.
fn run_optimize(args: &[String]) {
    let index_dir = args.get(2).expect("usage: stream_writer optimize <index_dir>");
    let w = IndexWriter::open(Path::new(index_dir), WriterOpts::default()).expect("open failed");
    let report = w.optimize().expect("optimize failed");
    println!(
        "{{\"optimized\":true,\"generation\":{},\"doc_count\":{}}}",
        report.generation, report.doc_count
    );
}

/// `stream_writer hold-lock <index_dir> <hold_ms>`: opens the IndexWriter (takes the write-lock),
/// prints LOCK_ACQUIRED, holds the lock for hold_ms ms and exits 0. If the lock is held, prints
/// LOCK_WOULDBLOCK and exits 3. Used by the inter-process test zsl_interprocess_lock.rs.
fn run_hold_lock(args: &[String]) {
    use std::io::Write;
    let index_dir = args.get(2).expect("usage: stream_writer hold-lock <index_dir> <hold_ms>");
    let hold_ms: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    match IndexWriter::open(Path::new(index_dir), WriterOpts::default()) {
        Ok(w) => {
            println!("LOCK_ACQUIRED");
            std::io::stdout().flush().ok();
            std::thread::sleep(std::time::Duration::from_millis(hold_ms));
            drop(w); // release the lock explicitly at the end of the hold
        }
        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
            println!("LOCK_WOULDBLOCK");
            std::process::exit(3);
        }
        Err(e) => {
            eprintln!("open failed: {e}");
            std::process::exit(2);
        }
    }
}

fn run_stream(args: &[String]) {
    let index_dir = args.get(1).expect("usage: stream_writer <index_dir> <docs.json> [cap]");
    let docs_path = args.get(2).expect("usage: stream_writer <index_dir> <docs.json> [cap]");
    let cap: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1000);

    let raw = std::fs::read_to_string(docs_path).expect("could not read docs.json");
    let spec: Spec = serde_json::from_str(&raw).expect("invalid docs.json");

    let opts = WriterOpts { max_buffered_docs: cap, ..WriterOpts::default() };
    let t0 = std::time::Instant::now();
    let mut w = IndexWriter::open(Path::new(index_dir), opts).expect("open failed");
    for d in spec.docs {
        let fields = d
            .fields
            .into_iter()
            .map(|f| {
                let kind = match f.kind.as_str() {
                    "text" => FieldKind::Text,
                    "keyword" => FieldKind::Keyword,
                    "unindexed" => FieldKind::UnIndexed,
                    other => panic!("unknown kind: {other}"),
                };
                WriterField { name: f.name, value: f.value, kind, stored: f.stored }
            })
            .collect();
        w.add_document(WriterDoc { fields }).expect("add_document failed");
    }
    let report = w.commit().expect("commit failed");
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    println!(
        "{{\"segments\":{},\"doc_count\":{},\"generation\":{},\"elapsed_ms\":{:.3},\"peak_rss_kb\":{}}}",
        report.segments.len(),
        report.doc_count,
        report.generation,
        elapsed_ms,
        peak_rss_kb()
    );
}

/// process peak RSS (VmHWM from /proc/self/status), in KB; 0 if unavailable.
fn peak_rss_kb() -> u64 {
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
