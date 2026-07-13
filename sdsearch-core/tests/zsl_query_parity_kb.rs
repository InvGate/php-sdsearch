//! Query parity over the real KB index (20 docs) vs the ZSL oracle.
//! Unlike the small incidents fixture (4 docs, all "New workflow"), here the queries
//! DIVERGE: term/wildcard/fuzzy/phrase yield distinct doc-sets, exercising the reader at scale.
use sdsearch_core::search::{fuzzy_query, phrase_query, term_query, wildcard_query};
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
        "/tests/fixtures/zsl_expected_kb.json"
    ))
    .expect("kb oracle missing — run the KB generator");
    serde_json::from_str(&raw).unwrap()
}
fn seg() -> ZslSegment {
    ZslSegment::open(&PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/zsl_index_kb"
    )))
    .unwrap()
}
fn sorted(mut v: Vec<usize>) -> Vec<usize> {
    v.sort_unstable();
    v
}
fn engine_set(hits: &[sdsearch_core::search::Hit]) -> Vec<usize> {
    sorted(hits.iter().map(|h| h.id).collect())
}
fn oracle_set(e: &Expected, k: &str) -> Vec<usize> {
    sorted(e.queries[k].iter().map(|h| h.id).collect())
}

#[test]
fn term_queries_match_oracle_and_diverge() {
    let (s, e) = (seg(), expected());
    assert_eq!(
        engine_set(&term_query(&s, "title", "vpn", 0.0, 100)),
        oracle_set(&e, "term:title:vpn")
    );
    assert_eq!(
        engine_set(&term_query(&s, "title", "laptop", 0.0, 100)),
        oracle_set(&e, "term:title:laptop")
    );
    // real divergence check: vpn and laptop match different docs
    assert_ne!(
        oracle_set(&e, "term:title:vpn"),
        oracle_set(&e, "term:title:laptop")
    );
}

#[test]
fn wildcard_query_matches_oracle() {
    let (s, e) = (seg(), expected());
    // print* expands over several title terms (print, printer, printing, …)
    assert_eq!(
        engine_set(&wildcard_query(&s, "title", "print*", 0, 0.0, 100)),
        oracle_set(&e, "wildcard:title:print*")
    );
    assert!(
        oracle_set(&e, "wildcard:title:print*").len() >= 2,
        "wildcard should expand to multiple docs"
    );
}

#[test]
fn fuzzy_typo_matches_oracle() {
    let (s, e) = (seg(), expected());
    // "passwrd" (typo of "password"), similarity 0.5 / prefix 3
    assert_eq!(
        engine_set(&fuzzy_query(&s, "title", "passwrd", 0.5, 3, 0.0, 100)),
        oracle_set(&e, "fuzzy:title:passwrd:0.5:3")
    );
}

#[test]
fn phrase_queries_match_oracle() {
    let (s, e) = (seg(), expected());
    assert_eq!(
        engine_set(&phrase_query(&s, "title", &["setting", "up"], 0.0, 100)),
        oracle_set(&e, "phrase:title:setting up")
    );
    assert_eq!(
        engine_set(&phrase_query(&s, "title", &["cloud", "storage"], 0.0, 100)),
        oracle_set(&e, "phrase:title:cloud storage")
    );
    // "setting up" matches 2 docs, "cloud storage" 1 → positions/adjacency genuinely exercised
    assert_ne!(
        oracle_set(&e, "phrase:title:setting up"),
        oracle_set(&e, "phrase:title:cloud storage")
    );
}
