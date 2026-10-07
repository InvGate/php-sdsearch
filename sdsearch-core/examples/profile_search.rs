//! DIAGNOSTIC (not part of the committed bench suite): isolates where the per-query time goes
//! for a low-hit ("none") search on a large OPTIMIZED single-segment index, to explain why
//! sdsearch's per-request search trails Zend on few/none at large N.
//!
//! It measures four phases independently, each as a warm median over many iterations (OS page
//! cache primed, so this is CPU + page-cache cost, not cold disk I/O):
//!   dict_read          — TermDict::read(.tis): parse the WHOLE term dictionary into memory.
//!   open_full          — ZslIndex::open(dir): the full per-request open (dict + norms + meta).
//!   query_only         — query::search over an ALREADY-OPEN index (no reopen).
//!   search_index_full  — runner::search_index(dir,…): the real per-call path (open + query).
//! Plus metadata: term_count, .tis vs .tii byte sizes (what an eager vs a .tii-lazy reader
//! would load), indexed-field count.
//!
//! Usage: cargo run -p sdsearch-core --release --example profile_search -- [N] [iters]
//!   N default 500000, iters default 25.

use sdsearch_core::query::{QueryParams, build_query, search};
use sdsearch_core::zsl::cfs::CompoundFile;
use sdsearch_core::zsl::fields::read_field_infos;
use sdsearch_core::zsl::index::ZslIndex;
use sdsearch_core::zsl::runner::search_index;
use sdsearch_core::zsl::segments::read_segment_infos;
use sdsearch_core::zsl::terms::TermDict;
use sdsearch_core::zsl::writer::{FieldKind, IndexWriter, WriterDoc, WriterField, WriterOpts};
use std::path::PathBuf;
use std::time::Instant;

const COMMON: &str = "widetoken";
const MISSING: &str = "absenttoken"; // never emitted → 0 hits (the "none" class)
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
    body.push(' ');
    body.push_str(COMMON);
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

fn copy_kb_base() -> PathBuf {
    let src = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/zsl_index_kb"
    ));
    let dst = std::env::temp_dir().join(format!("sdsearch_profile_{}", std::process::id()));
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

/// warm median (ms) of `f` over `iters` timed runs after 3 warm-up runs.
fn median_ms(iters: usize, mut f: impl FnMut()) -> (f64, f64, f64) {
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
    (s[s.len() / 2], s[0], s[s.len() - 1])
}

fn sub_ending<'a>(cfs: &'a CompoundFile, ext: &str) -> Option<&'a [u8]> {
    let name = cfs.names().into_iter().find(|n| n.ends_with(ext))?;
    cfs.sub(&name)
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(500_000);
    let iters: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(25);

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

    // ---- metadata: single optimized segment, its .tis/.tii/.nrm sizes and term_count ----
    let infos = read_segment_infos(&dir).unwrap();
    let seg = &infos[0];
    let cfs = CompoundFile::open(&dir.join(format!("{}.cfs", seg.name))).unwrap();
    let tis = sub_ending(&cfs, ".tis").expect(".tis").to_vec();
    let tii = sub_ending(&cfs, ".tii")
        .map(<[u8]>::to_vec)
        .unwrap_or_default();
    let tii_len = tii.len();
    let nrm_len = sub_ending(&cfs, ".nrm").map_or(0, <[u8]>::len);
    let fnm = sub_ending(&cfs, ".fnm").expect(".fnm");
    let fields = read_field_infos(fnm).unwrap();
    let field_names: Vec<String> = fields.iter().map(|f| f.name.clone()).collect();
    let indexed = fields.iter().filter(|f| f.is_indexed).count();
    // .tis header: [i32 marker][u64 term_count][i32 index_interval]…
    let term_count = u64::from_be_bytes(tis[4..12].try_into().unwrap());
    let index_interval = i32::from_be_bytes(tis[12..16].try_into().unwrap());

    // ---- phase timings (warm medians) ----
    let params = QueryParams {
        text: MISSING.to_string(),
        where_groups: vec![],
        in_groups: vec![],
        fuzzy_similarity: 0.5,
        fuzzy_prefix_len: 3,
        wildcard_min_prefix: 0,
        accent_insensitive: false,
        synonyms: false,
        field_weights: std::collections::HashMap::new(),
        similarity: sdsearch_core::score::Similarity::Bm25,
        range_filters: vec![],
        match_all: vec![],
        sort: None,
        boolean_tree: None,
    };
    let query = build_query(&params).unwrap();

    let (dict_med, dict_min, dict_max) = median_ms(iters, || {
        let d = TermDict::open(&tis, &tii, &field_names).unwrap();
        std::hint::black_box(&d);
    });
    let (open_med, open_min, open_max) = median_ms(iters, || {
        let idx = ZslIndex::open(&dir).unwrap();
        std::hint::black_box(&idx);
    });
    let idx = ZslIndex::open(&dir).unwrap(); // pre-opened, reused → isolates query cost
    let (q_med, q_min, q_max) = median_ms(iters, || {
        let hits = search(&idx, &query, 0.0, 20);
        std::hint::black_box(&hits);
    });
    let (full_med, full_min, full_max) = median_ms(iters, || {
        let hits = search_index(&dir, &params, 0.0, 20).unwrap();
        std::hint::black_box(&hits);
    });

    println!(
        "\n==== profile: \"none\" query (0 hits) on an OPTIMIZED {n}-doc single-segment index ====\n"
    );
    println!("index metadata:");
    println!("  indexed fields : {indexed}");
    println!("  term_count     : {term_count}   (unique terms in the dictionary)");
    println!("  index_interval : {index_interval}   (.tii has ~1 entry per this many terms)");
    println!(
        "  .tis size      : {:>10} bytes   (eager reader loads ALL of this)",
        tis.len()
    );
    println!("  .tii size      : {tii_len:>10} bytes   (Zend/lazy reader loads ~this)");
    println!("  .nrm size      : {nrm_len:>10} bytes");
    if tii_len > 0 {
        println!(
            "  .tis / .tii    : {:.1}×   (how much MORE an eager reader reads)",
            tis.len() as f64 / tii_len as f64
        );
    }
    println!("\nphase timings (warm median / min / max, ms, iters={iters}):");
    println!(
        "  dict_open (TermDict::open, lazy)  : {dict_med:8.3} / {dict_min:.3} / {dict_max:.3}"
    );
    println!(
        "  open_full (ZslIndex::open)        : {open_med:8.3} / {open_min:.3} / {open_max:.3}"
    );
    println!("  query_only (search, index open)   : {q_med:8.3} / {q_min:.3} / {q_max:.3}");
    println!(
        "  search_index_full (open + query)  : {full_med:8.3} / {full_min:.3} / {full_max:.3}"
    );
    println!("\ninterpretation (from the numbers above):");
    println!(
        "  dict_read / open_full   = {:.0}%   (share of open spent loading the term dict)",
        100.0 * dict_med / open_med
    );
    println!(
        "  open_full / full        = {:.0}%   (share of a per-request search spent opening)",
        100.0 * open_med / full_med
    );
    println!(
        "  query_only / full       = {:.0}%   (share spent on the actual query)",
        100.0 * q_med / full_med
    );

    std::fs::remove_dir_all(&dir).ok();
}
