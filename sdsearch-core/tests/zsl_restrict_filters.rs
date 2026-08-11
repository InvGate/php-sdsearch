//! Integration coverage for the `restrict` allow-list (range + matchAll) over a real ZSL index.
//! `MemoryIndex` unit tests cannot express deletes or multi-segment doc-id remapping.

use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

use sdsearch_core::index::IndexReader;
use sdsearch_core::query::{MatchAllFilter, QueryParams, RangeFilter, range_allow_list};
use sdsearch_core::score::Similarity;
use sdsearch_core::zsl::index::ZslIndex;
use sdsearch_core::zsl::runner::search_index_paged;
use sdsearch_core::zsl::writer::{IndexWriter, WriterOpts};

static COUNTER: AtomicU32 = AtomicU32::new(0);

/// copies the ENTIRE KB fixture (incl. `_2.cfs`) to a fresh temp dir.
fn temp_kb_full() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("sdsearch_restrict_{}_{}", std::process::id(), n));
    std::fs::create_dir_all(&dir).unwrap();
    let src = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/zsl_index_kb"
    ));
    for entry in std::fs::read_dir(&src).unwrap() {
        let p = entry.unwrap().path();
        std::fs::copy(&p, dir.join(p.file_name().unwrap())).unwrap();
    }
    dir
}

/// `QueryParams` deliberately has no `Default`, so tests spell it out.
fn params(
    text: &str,
    range_filters: Vec<RangeFilter>,
    match_all: Vec<MatchAllFilter>,
) -> QueryParams {
    QueryParams {
        text: text.to_string(),
        where_groups: vec![],
        in_groups: vec![],
        fuzzy_similarity: 0.5,
        fuzzy_prefix_len: 3,
        wildcard_min_prefix: 2,
        accent_insensitive: false,
        synonyms: false,
        field_weights: HashMap::new(),
        similarity: Similarity::Bm25,
        range_filters,
        match_all,
        sort: None,
    }
}

#[test]
fn range_and_match_all_intersect_over_a_real_zsl_index() {
    let dir = temp_kb_full();

    // Reference: the same text query with no filters. Taking this from observation rather than
    // hardcoding it keeps the test independent of which fields build_query searches.
    let unfiltered = search_index_paged(
        &dir,
        &params("configure", vec![], vec![]),
        0.0,
        0,
        100,
        None,
    )
    .unwrap();
    let text_hits: BTreeSet<usize> = unfiltered.hits.iter().map(|h| h.id).collect();
    assert!(
        !text_hits.is_empty(),
        "text query must match something, else the test is vacuous"
    );

    // Fixture facts: created_at_key of doc i is 1700000000 + i*3600, so this bound selects 0..=9.
    let in_range: BTreeSet<usize> = (0..=9).collect();
    // Fixture fact: description:"configure" has df=4 on docs 0, 9, 12, 17.
    let has_word: BTreeSet<usize> = [0, 9, 12, 17].into_iter().collect();

    let expected: BTreeSet<usize> = text_hits
        .iter()
        .filter(|d| in_range.contains(d) && has_word.contains(d))
        .copied()
        .collect();
    assert!(
        !expected.is_empty(),
        "filters must leave something, else the test is vacuous"
    );
    assert!(
        expected.len() < text_hits.len(),
        "filters must actually remove something"
    );

    let filtered = params(
        "configure",
        vec![RangeFilter {
            field: "created_at_key".into(),
            lower: Some("1700000000".into()),
            upper: Some("1700032400".into()),
        }],
        vec![MatchAllFilter {
            field: "description".into(),
            text: "configure".into(),
        }],
    );
    let out = search_index_paged(&dir, &filtered, 0.0, 0, 100, None).unwrap();
    let got: BTreeSet<usize> = out.hits.iter().map(|h| h.id).collect();

    assert_eq!(got, expected, "range AND matchAll must intersect");
    assert_eq!(out.total, expected.len(), "total counts the restricted set");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn short_circuit_compares_against_live_docs_not_maxdoc() {
    let dir = temp_kb_full();

    // delete one of the fixture's 20 docs: num_docs becomes 19, maxDoc stays 20
    let mut w = IndexWriter::open(&dir, WriterOpts::default()).unwrap();
    w.delete_document(5);
    w.commit().unwrap();

    let idx = ZslIndex::open(&dir).unwrap();
    assert_eq!(idx.num_docs(), 19, "one doc deleted");
    assert_eq!(idx.total_docs(), 20, "maxDoc still counts the delete");

    // created_at_key is universal over the fixture, so an unbounded range covers every LIVE doc.
    // Comparing against total_docs() would give 19 != 20 and silently fail to fire.
    let filters = vec![RangeFilter {
        field: "created_at_key".into(),
        lower: None,
        upper: None,
    }];
    assert!(
        range_allow_list(&idx, &filters).is_none(),
        "must short-circuit on live docs"
    );

    std::fs::remove_dir_all(&dir).ok();
}
