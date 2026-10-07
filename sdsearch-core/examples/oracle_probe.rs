//! Local-only differential probe: run `search_index` over a ZSL index dir and print the
//! sorted doc-id set for a query. Paired with tools' Zend oracle to check search parity.
//! Usage: cargo run -p sdsearch-core --example oracle_probe -- <index_dir> <query text...>

use sdsearch_core::query::QueryParams;
use sdsearch_core::score::Similarity;
use sdsearch_core::zsl::runner::search_index;
use std::collections::HashMap;
use std::path::PathBuf;

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().expect("usage: oracle_probe <dir> <query...>"));
    let text = args.collect::<Vec<_>>().join(" ");

    let params = QueryParams {
        text,
        where_groups: vec![],
        in_groups: vec![],
        fuzzy_similarity: 0.5,
        fuzzy_prefix_len: 3,
        wildcard_min_prefix: 0,
        accent_insensitive: false,
        synonyms: false,
        field_weights: HashMap::new(),
        similarity: Similarity::Bm25,
        range_filters: vec![],
        match_all: vec![],
        sort: None,
        boolean_tree: None,
    };

    let hits = search_index(&dir, &params, 0.0, 0).expect("search failed");
    let mut ids: Vec<usize> = hits.iter().map(|h| h.id).collect();
    ids.sort_unstable();
    println!("RUST ids: {ids:?}");
}
