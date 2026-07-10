//! Native reader for the differential diff-testing harness: opens a ZSL index, runs a battery
//! of queries (all-fields, via build_query) and dumps doc-sets keyed by `docid` plus the
//! term-dict (field,text,docFreq) as ONE JSON line. Used by tools/diff_writer.php for the
//! 2×2 matrix (triangulation).
//! Usage: diff_read <index_dir> <queries.json>
//!   queries.json: {"fields":["body","tag"],"queries":[{"name":"work","text":"work"},...]}

use sdsearch_core::index::IndexReader;
use sdsearch_core::query::{build_query, search, QueryParams};
use sdsearch_core::zsl::index::ZslIndex;
use std::collections::BTreeMap;
use std::path::Path;

#[derive(serde::Deserialize)]
struct QSpec {
    fields: Vec<String>,
    queries: Vec<Q>,
}
#[derive(serde::Deserialize)]
struct Q {
    name: String,
    text: String,
}

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
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).expect("usage: diff_read <index_dir> <queries.json>");
    let qpath = args.get(2).expect("usage: diff_read <index_dir> <queries.json>");

    let raw = std::fs::read_to_string(qpath).expect("could not read queries.json");
    let spec: QSpec = serde_json::from_str(&raw).expect("invalid queries.json");
    let index = ZslIndex::open(Path::new(dir)).expect("could not open the index");

    // doc-sets keyed by docid (stored field 'docid'), in the order returned by search().
    let mut doc_sets: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for q in &spec.queries {
        let query = build_query(&params(&q.text)).expect("build_query failed");
        let hits: Vec<String> = search(&index, &query, 0.0, 10_000)
            .into_iter()
            .filter_map(|h: sdsearch_core::search::Hit| h.fields.get("docid").cloned())
            .collect();
        doc_sets.insert(q.name.clone(), hits);
    }

    // term-dict: for each field, all terms (prefix "") plus docFreq. Excludes 'docid'.
    let mut term_dict: BTreeMap<String, usize> = BTreeMap::new();
    for field in &spec.fields {
        if field == "docid" {
            continue;
        }
        for term in index.terms_with_prefix(field, "") {
            let df = index.doc_freq(field, &term);
            term_dict.insert(format!("{field}\u{0}{term}"), df);
        }
    }

    let out = serde_json::json!({ "doc_sets": doc_sets, "term_dict": term_dict });
    println!("{}", serde_json::to_string(&out).expect("serialize"));
}
