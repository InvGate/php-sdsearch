//! Parity of boolean composition (build_query + executor) vs the ZSL boolean oracle
//! (a transcription of Zend Lucene's boolean query builder in the multiseg generator).
use sdsearch_core::query::{InGroup, Occur, QueryParams, WhereGroup, build_query, search};
use sdsearch_core::search::Hit;
use sdsearch_core::zsl::index::ZslIndex;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(serde::Deserialize)]
struct Expected {
    bool_queries: HashMap<String, Vec<HitId>>,
}
#[derive(serde::Deserialize)]
struct HitId {
    id: usize,
}

fn expected() -> Expected {
    let raw = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/zsl_expected_multiseg.json"
    ))
    .expect("multiseg oracle missing");
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
    v.sort_unstable();
    v
}
fn eng(hits: &[Hit]) -> Vec<usize> {
    sorted(hits.iter().map(|h| h.id).collect())
}
fn ora(e: &Expected, k: &str) -> Vec<usize> {
    sorted(e.bool_queries[k].iter().map(|h| h.id).collect())
}

fn base(text: &str) -> QueryParams {
    QueryParams {
        text: text.into(),
        where_groups: vec![],
        in_groups: vec![],
        fuzzy_similarity: 0.5,
        fuzzy_prefix_len: 3,
        wildcard_min_prefix: 0,
        accent_insensitive: false,
        field_weights: std::collections::HashMap::new(),
        similarity: sdsearch_core::score::Similarity::Bm25,
        range_filters: vec![],
        match_all: vec![],
    }
}

#[test]
fn text_only_matches_oracle() {
    let (i, e) = (idx(), expected());
    let q = build_query(&base("vpn")).unwrap();
    assert_eq!(eng(&search(&i, &q, 0.0, 100)), ora(&e, "text-only:vpn"));
}

#[test]
fn text_plus_where_matches_oracle() {
    let (i, e) = (idx(), expected());
    let mut p = base("vpn");
    p.where_groups = vec![WhereGroup {
        field: "lang".into(),
        values: vec!["es".into()],
        occur: Occur::Should,
    }];
    let q = build_query(&p).unwrap();
    assert_eq!(
        eng(&search(&i, &q, 0.0, 100)),
        ora(&e, "text+where:vpn|lang=es")
    );
}

#[test]
fn text_plus_in_matches_oracle() {
    let (i, e) = (idx(), expected());
    let mut p = base("vpn");
    p.in_groups = vec![InGroup {
        field: "cat".into(),
        values: vec!["1".into(), "2".into()],
    }];
    let q = build_query(&p).unwrap();
    assert_eq!(
        eng(&search(&i, &q, 0.0, 100)),
        ora(&e, "text+in:vpn|cat=1,2")
    );
}

#[test]
fn text_plus_multi_in_matches_oracle() {
    // parity for the multi-field IN-clause merge (all `in` groups collapse into a single MultiTerm):
    // cat=3 OR lang=en, as one required MultiTerm.
    let (i, e) = (idx(), expected());
    let mut p = base("vpn");
    p.in_groups = vec![
        InGroup {
            field: "cat".into(),
            values: vec!["3".into()],
        },
        InGroup {
            field: "lang".into(),
            values: vec!["en".into()],
        },
    ];
    let q = build_query(&p).unwrap();
    assert_eq!(
        eng(&search(&i, &q, 0.0, 100)),
        ora(&e, "text+in-multi:vpn|cat=3&lang=en")
    );
}

#[test]
fn where_mustnot_matches_oracle() {
    let (i, e) = (idx(), expected());
    let mut p = base("how");
    p.where_groups = vec![WhereGroup {
        field: "lang".into(),
        values: vec!["en".into()],
        occur: Occur::MustNot,
    }];
    let q = build_query(&p).unwrap();
    assert_eq!(
        eng(&search(&i, &q, 0.0, 100)),
        ora(&e, "where-mustnot:how|lang!=en")
    );
}

#[test]
fn text_multiword_matches_oracle() {
    // multi-word coverage of free text (the previous oracle only had single-word cases).
    let (i, e) = (idx(), expected());
    let q = build_query(&base("how to")).unwrap();
    assert_eq!(
        eng(&search(&i, &q, 0.0, 100)),
        ora(&e, "text-multiword:how to")
    );
}
