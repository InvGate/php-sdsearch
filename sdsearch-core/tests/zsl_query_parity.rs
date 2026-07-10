//! Parity of the shared engine's queries over a real ZSL index vs the ZSL oracle.
use sdsearch_core::search::{fuzzy_query, term_query, wildcard_query};
use sdsearch_core::zsl::segment::ZslSegment;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(serde::Deserialize)]
struct Expected {
    queries: HashMap<String, Vec<Hit>>,
}
#[derive(serde::Deserialize)]
struct Hit {
    id: usize,
}

fn expected() -> Expected {
    let raw = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/zsl_expected.json"
    ))
    .expect("fixture missing — run sdsearch_dump_zsl_index.php");
    serde_json::from_str(&raw).unwrap()
}

fn seg() -> ZslSegment {
    ZslSegment::open(&PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/zsl_index"
    )))
    .unwrap()
}

fn doc_set(hits: &[sdsearch_core::search::Hit]) -> Vec<usize> {
    let mut v: Vec<usize> = hits.iter().map(|h| h.id).collect();
    v.sort();
    v
}
fn oracle_set(exp: &Expected, key: &str) -> Vec<usize> {
    let mut v: Vec<usize> = exp.queries[key].iter().map(|h| h.id).collect();
    v.sort();
    v
}

#[test]
fn term_query_matches_zsl_docset() {
    let (s, exp) = (seg(), expected());
    let hits = term_query(&s, "title", "new", 0.0, 100);
    assert_eq!(doc_set(&hits), oracle_set(&exp, "term:title:new"));
}

#[test]
fn wildcard_query_matches_zsl_docset() {
    let (s, exp) = (seg(), expected());
    // fidelity: min_prefix 0 (a constant of the target format, not an engine hardcode)
    let hits = wildcard_query(&s, "title", "ne*", 0, 0.0, 100);
    assert_eq!(doc_set(&hits), oracle_set(&exp, "wildcard:title:ne*"));
}

#[test]
fn fuzzy_query_matches_zsl_docset() {
    let (s, exp) = (seg(), expected());
    // fidelity: similarity 0.5, prefix 3 (constants of the target format)
    let hits = fuzzy_query(&s, "title", "new", 0.5, 3, 0.0, 100);
    assert_eq!(doc_set(&hits), oracle_set(&exp, "fuzzy:title:new:0.5:3"));
}
