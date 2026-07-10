//! Append a batch of docs (JSON) to an existing ZSL index via the native writer.
//! Used by the golden interchange gate (`tools/golden_writer.php`) and the perf harness.
//!
//! Usage:
//!   cargo run -p sdsearch-core --release --example append_writer -- <index_dir> <docs.json>
//!
//! docs.json:
//!   { "docs": [ { "fields": [
//!       {"name":"title","value":"...","kind":"text","stored":true}, ... ] } ] }
//!   kind ∈ {"text","keyword","unindexed"} (default stored=true)

use sdsearch_core::zsl::writer::{append_documents, FieldKind, WriterDoc, WriterField, WriterOpts};
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
    let index_dir = args
        .get(1)
        .expect("usage: append_writer <index_dir> <docs.json>");
    let docs_path = args
        .get(2)
        .expect("usage: append_writer <index_dir> <docs.json>");

    let raw = std::fs::read_to_string(docs_path).expect("could not read docs.json");
    let spec: Spec = serde_json::from_str(&raw).expect("invalid docs.json");

    let docs: Vec<WriterDoc> = spec
        .docs
        .into_iter()
        .map(|d| WriterDoc {
            fields: d
                .fields
                .into_iter()
                .map(|f| {
                    let kind = match f.kind.as_str() {
                        "text" => FieldKind::Text,
                        "keyword" => FieldKind::Keyword,
                        "unindexed" => FieldKind::UnIndexed,
                        other => panic!("unknown kind: {other}"),
                    };
                    WriterField {
                        name: f.name,
                        value: f.value,
                        kind,
                        stored: f.stored,
                    }
                })
                .collect(),
        })
        .collect();

    let t0 = std::time::Instant::now();
    let report = append_documents(Path::new(index_dir), &docs, &WriterOpts::default())
        .expect("append_documents failed");
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // JSON output for the harness to parse (includes timing + process peak RSS)
    println!(
        "{{\"segment\":\"{}\",\"doc_count\":{},\"generation\":{},\"elapsed_ms\":{:.3},\"peak_rss_kb\":{}}}",
        report.segment_name, report.doc_count, report.generation, elapsed_ms, peak_rss_kb()
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
