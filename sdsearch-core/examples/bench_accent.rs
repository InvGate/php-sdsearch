//! accent_insensitive perf bench: the SAME free-text query run with the flag OFF vs ON, over an
//! N-doc Spanish index, to confirm the accent expansion adds negligible latency.
//!
//! The corpus plants both accented and plain Spanish forms (`como`/`cómo`, `dron`, `un`, `volar`,
//! plus accented filler like `avión`, `gestión`) so the query "como volar un dron?" actually hits
//! and its existing accent variants get read — i.e. we measure the real added work, not a no-op.
//!
//! Plain system allocator (no tracking) → honest wall time. Uses the real runner path
//! (`search_index`), reopening the index per call exactly like `SdSearch\Engine::search`.
//!
//! Usage:
//!   cargo run -p sdsearch-core --release --example bench_accent -- [N] [iters]

use sdsearch_core::query::QueryParams;
use sdsearch_core::zsl::runner::search_index;
use sdsearch_core::zsl::writer::{FieldKind, IndexWriter, WriterDoc, WriterField, WriterOpts};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Spanish vocabulary mixing accented and plain forms so accent variants exist in the dictionary.
/// The query tokens (como/volar/un/dron) are all present; `cómo` is present too, so the flag ON
/// path reads one extra (long) posting list per hit of that variant — a fair worst case.
const POOL: &[&str] = &[
    "como",
    "cómo",
    "volar",
    "dron",
    "un",
    "una",
    "avión",
    "avion",
    "gestión",
    "acción",
    "reunión",
    "camión",
    "función",
    "información",
    "usuario",
    "contraseña",
    "año",
    "niño",
    "configuración",
    "instalación",
    "conexión",
    "servidor",
    "impresora",
    "correo",
    "acceso",
    "error",
    "actualización",
    "licencia",
    "teléfono",
    "público",
    "rápido",
    "árbol",
    "página",
    "código",
    "número",
    "región",
    "versión",
    "sesión",
    "botón",
    "red",
];

/// one deterministic doc numbered `i`: a 40-token body drawn cyclically from POOL, plus a
/// per-doc keyword id. Bounded vocabulary => realistic term dictionary and doc frequencies.
fn gen_one(i: usize) -> WriterDoc {
    let np = POOL.len();
    let body: String = (0..40)
        .map(|j| POOL[(i * 7 + j * 5) % np])
        .collect::<Vec<_>>()
        .join(" ");
    let title = format!(
        "{} {} {}",
        POOL[i % np],
        POOL[(i * 3) % np],
        POOL[(i * 7) % np]
    );
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
}

/// copies the committed KB fixture to a fresh temp dir (skips locks and `.sti`) as the writer base.
fn copy_kb_base(tag: &str) -> PathBuf {
    let src = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/zsl_index_kb"
    ));
    let dst =
        std::env::temp_dir().join(format!("sdsearch_accbench_{}_{}", std::process::id(), tag));
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

/// builds an N-doc index on a fresh KB base and optimizes it to a single segment (production shape).
fn build_index(n: usize) -> PathBuf {
    let dir = copy_kb_base("build");
    let opts = WriterOpts {
        max_buffered_docs: 1000,
        ..WriterOpts::default()
    };
    let mut w = IndexWriter::open(&dir, opts).expect("open writer");
    for i in 0..n {
        w.add_document(gen_one(i)).expect("add_document");
    }
    w.optimize().expect("optimize");
    dir
}

fn params(text: &str, accent_insensitive: bool) -> QueryParams {
    QueryParams {
        text: text.to_string(),
        where_groups: vec![],
        in_groups: vec![],
        fuzzy_similarity: 0.5,
        fuzzy_prefix_len: 3,
        wildcard_min_prefix: 0,
        accent_insensitive,
        synonyms: false,
        field_weights: HashMap::new(),
        similarity: sdsearch_core::score::Similarity::Bm25,
        range_filters: vec![],
        match_all: vec![],
        sort: None,
        boolean_tree: None,
    }
}

/// times `iters` search runs at `limit`, discards a warm-up, returns (p50, p95) ms.
fn time_search(dir: &Path, p: &QueryParams, limit: usize, iters: usize) -> (f64, f64) {
    let _ = search_index(dir, p, 0.0, limit); // warm-up (not sampled)
    let mut samples: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = std::time::Instant::now();
        let r = search_index(dir, p, 0.0, limit).expect("search failed");
        samples.push(t0.elapsed().as_secs_f64() * 1000.0);
        std::hint::black_box(r);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |q: f64| samples[((samples.len() - 1) as f64 * q).round() as usize];
    (pct(0.50), pct(0.95))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // `--dir <path> [iters]` measures against an existing (e.g. real Zend) index, read-only,
    // and does NOT delete it. Otherwise builds a synthetic N-doc Spanish index.
    if args.get(1).map(String::as_str) == Some("--dir") {
        let dir = PathBuf::from(args.get(2).expect("--dir needs a path"));
        let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(200);
        run_against(&dir, iters);
        return;
    }

    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(50_000);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);
    let dir = build_index(n);
    run_against(&dir, iters);
    std::fs::remove_dir_all(&dir).ok();
}

fn run_against(dir: &Path, iters: usize) {
    let n = sdsearch_core::zsl::index::ZslIndex::open(dir).map_or(0, |r| {
        use sdsearch_core::index::IndexReader;
        r.num_docs()
    });

    // Two queries:
    //  - "target": the user's query; accent variants EXIST (cómo), so ON matches more docs =>
    //    the latency delta includes the extra RECALL (the feature), not just overhead.
    //  - "no_extra_recall": tokens whose accent variants do NOT exist in the index, so hits are
    //    identical ON vs OFF => isolates the PURE overhead of the flag (generation + doc_freq).
    let queries = [
        ("target", "como volar un dron?"),
        ("no_extra_recall", "servidor impresora usuario error"),
    ];

    for (label, query) in queries {
        let hits_off = search_index(dir, &params(query, false), 0.0, 0)
            .expect("count off")
            .len();
        let hits_on = search_index(dir, &params(query, true), 0.0, 0)
            .expect("count on")
            .len();
        for limit in [20usize, 100] {
            let (off50, off95) = time_search(dir, &params(query, false), limit, iters);
            let (on50, on95) = time_search(dir, &params(query, true), limit, iters);
            println!(
                "{{\"case\":{label:?},\"n\":{n},\"iters\":{iters},\"query\":{query:?},\"top\":{limit},\
\"hits_off\":{hits_off},\"hits_on\":{hits_on},\
\"off\":{{\"p50_ms\":{off50:.4},\"p95_ms\":{off95:.4}}},\
\"on\":{{\"p50_ms\":{on50:.4},\"p95_ms\":{on95:.4}}},\
\"ratio_p50\":{:.3},\"ratio_p95\":{:.3}}}",
                on50 / off50,
                on95 / off95
            );
        }
    }
}
