//! perf of the native reader over a large user-provided ZSL index.
//! Usage: SDSEARCH_PERF_INDEX=/path/to/index cargo run -p sdsearch-core --release --example perf_native
//! If the env var is unset, exits 0 without running (for CI/dev without the index).
use sdsearch_core::query::{QueryParams, build_query, search};
use sdsearch_core::zsl::index::ZslIndex;
use std::time::Instant;

fn params(text: &str) -> QueryParams {
    QueryParams {
        text: text.into(),
        where_groups: vec![],
        in_groups: vec![],
        fuzzy_similarity: 0.5,
        fuzzy_prefix_len: 3,
        wildcard_min_prefix: 0,
    }
}

fn main() {
    let Ok(dir) = std::env::var("SDSEARCH_PERF_INDEX") else {
        eprintln!("SDSEARCH_PERF_INDEX not set — native perf skipped");
        return;
    };
    let index = ZslIndex::open(std::path::Path::new(&dir)).expect("could not open the index");

    // Edit this list so it reflects real terms from the provided index.
    // WARNING: keep this list IDENTICAL to $queries in tools/perf_zsl.php, and in
    // lowercase/space-separated — this side does lowercase + split_whitespace,
    // the PHP side does not; if they diverge, the latency numbers compare different queries.
    let queries = ["felipe", "roa", "juan rodriguez", "andrés barrios"];
    let iters = 20usize;

    println!("query,p50_ms,p95_ms,hits");
    for text in queries {
        let q = build_query(&params(text)).unwrap();
        let mut samples: Vec<f64> = Vec::with_capacity(iters);
        let mut hits = 0usize;
        for _ in 0..iters {
            let t = Instant::now();
            let r = search(&index, &q, 0.0, 100);
            samples.push(t.elapsed().as_secs_f64() * 1000.0);
            hits = r.len();
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p = |q: f64| samples[((samples.len() as f64 - 1.0) * q).round() as usize];
        println!("{text:?},{:.3},{:.3},{hits}", p(0.50), p(0.95));
    }
}
