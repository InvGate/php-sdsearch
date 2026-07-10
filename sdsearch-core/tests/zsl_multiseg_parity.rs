//! Leaf-query parity over a real MULTI-SEGMENT ZSL index vs the ZSL oracle.
//! Exercises global doc-ids crossing segments, deletion exclusion, and per-base routing.
use sdsearch_core::index::IndexReader;
use sdsearch_core::search::{fuzzy_query, phrase_query, term_query, wildcard_query, Hit};
use sdsearch_core::zsl::index::ZslIndex;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(serde::Deserialize)]
struct Expected {
    num_docs: usize,
    queries: HashMap<String, Vec<HitId>>,
    docs: Vec<DocEntry>,
}
#[derive(serde::Deserialize)]
struct HitId {
    id: usize,
}
#[derive(serde::Deserialize)]
struct DocEntry {
    id: usize,
    stored: HashMap<String, String>,
}

fn expected() -> Expected {
    let raw = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/zsl_expected_multiseg.json"
    ))
    .expect("multiseg oracle missing — run tools/gen_zsl_multiseg_fixture.php");
    serde_json::from_str(&raw).unwrap()
}
fn idx() -> ZslIndex {
    ZslIndex::open(&PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/zsl_index_multiseg"
    )))
    .unwrap()
}
fn sorted(mut v: Vec<usize>) -> Vec<usize> {
    v.sort();
    v
}
fn eng(hits: &[Hit]) -> Vec<usize> {
    sorted(hits.iter().map(|h| h.id).collect())
}
fn ora(e: &Expected, k: &str) -> Vec<usize> {
    sorted(e.queries[k].iter().map(|h| h.id).collect())
}
// RAW order (unsorted) exactly as the engine / oracle returns it — to check ranking coherence.
fn eng_raw(hits: &[Hit]) -> Vec<usize> {
    hits.iter().map(|h| h.id).collect()
}
fn ora_raw(e: &Expected, k: &str) -> Vec<usize> {
    e.queries[k].iter().map(|h| h.id).collect()
}

#[test]
fn num_docs_excludes_deleted() {
    assert_eq!(idx().num_docs(), expected().num_docs);
}

#[test]
fn term_queries_cross_segment_boundary() {
    let (i, e) = (idx(), expected());
    // vpn matches docs from two different segments
    let vpn = ora(&e, "term:title:vpn");
    assert_eq!(eng(&term_query(&i, "title", "vpn", 0.0, 100)), vpn);
    assert!(vpn.len() >= 2, "vpn must cross segments");
    assert_eq!(
        eng(&term_query(&i, "title", "mysql", 0.0, 100)),
        ora(&e, "term:title:mysql")
    );
}

#[test]
fn term_vpn_rank_order_matches_oracle() {
    // guards cross-segment ranking coherence: both docs have the same score,
    // so the engine's score-desc/id-asc tie-break must match ZSL's raw order.
    let (i, e) = (idx(), expected());
    let expected_order = ora_raw(&e, "term:title:vpn");
    assert_eq!(
        eng_raw(&term_query(&i, "title", "vpn", 0.0, 100)),
        expected_order
    );
}

#[test]
fn deleted_doc_term_returns_empty() {
    let (i, e) = (idx(), expected());
    assert_eq!(ora(&e, "term:title:backup"), Vec::<usize>::new());
    assert_eq!(
        eng(&term_query(&i, "title", "backup", 0.0, 100)),
        Vec::<usize>::new()
    );
}

#[test]
fn wildcard_fuzzy_phrase_match_oracle() {
    let (i, e) = (idx(), expected());
    assert_eq!(
        eng(&wildcard_query(&i, "title", "re*", 0, 0.0, 100)),
        ora(&e, "wildcard:title:re*")
    );
    assert_eq!(
        eng(&fuzzy_query(&i, "title", "mysgl", 0.5, 3, 0.0, 100)),
        ora(&e, "fuzzy:title:mysgl:0.5:3")
    );
    assert_eq!(
        eng(&phrase_query(&i, "title", &["how", "to"], 0.0, 100)),
        ora(&e, "phrase:title:how to")
    );
}

#[test]
fn stored_routing_by_base_is_correct() {
    let (i, e) = (idx(), expected());
    // each global id from the oracle must return the same stored id_key
    for d in &e.docs {
        let got = i.stored_fields(d.id);
        assert_eq!(
            got.get("id_key"),
            d.stored.get("id_key"),
            "id {} mal ruteado",
            d.id
        );
    }
}
