//! A sort by a field the index does not have must fail loudly, over a REAL ZSL index.
//!
//! `MemoryIndex` cannot express what makes this worth an integration test: stored-only fields
//! (which `.fnm` flags but `MemoryIndex` does not model), and an index whose documents are all
//! deleted.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

use sdsearch_core::index::IndexReader;
use sdsearch_core::query::QueryParams;
use sdsearch_core::score::Similarity;
use sdsearch_core::search::SortSpec;
use sdsearch_core::zsl::index::ZslIndex;
use sdsearch_core::zsl::runner::search_index_paged;
use sdsearch_core::zsl::writer::{IndexWriter, WriterOpts};

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn temp_kb() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("sdsearch_sortfield_{}_{}", std::process::id(), n));
    std::fs::create_dir_all(&dir).unwrap();
    let src = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/zsl_index_kb"
    ));
    for entry in std::fs::read_dir(&src).unwrap() {
        let p = entry.unwrap().path();
        if p.is_file() {
            std::fs::copy(&p, dir.join(p.file_name().unwrap())).unwrap();
        }
    }
    dir
}

/// `QueryParams` deliberately has no `Default`, so tests spell it out.
fn params(sort: Option<SortSpec>) -> QueryParams {
    QueryParams {
        text: "a".to_string(),
        where_groups: vec![],
        in_groups: vec![],
        range_filters: vec![],
        match_all: vec![],
        fuzzy_similarity: 0.5,
        fuzzy_prefix_len: 3,
        wildcard_min_prefix: 2,
        accent_insensitive: false,
        field_weights: HashMap::new(),
        similarity: Similarity::Bm25,
        sort,
    }
}

#[test]
fn an_unknown_sort_field_is_an_error_not_a_silently_unsorted_result() {
    let dir = temp_kb();

    // `SearchOutcome` is not `Debug`, so `expect_err` is unavailable here.
    let msg = match search_index_paged(
        &dir,
        &params(Some(SortSpec {
            field: "created_at".into(), // the real field is `created_at_key`
            ascending: true,
        })),
        0.0,
        0,
        10,
        None,
    ) {
        Ok(_) => panic!("a misspelled sort field must not succeed"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("created_at"),
        "the error must name the offending field, got: {msg}"
    );

    // Without the check this exact query SUCCEEDS and returns hits — every doc resolves to
    // Missing, they all tie, and the tiebreak order comes back looking like broken sorting.
    // Asserting the same query works under the correct name is what proves the check is
    // discriminating rather than rejecting sorts wholesale.
    let ok = search_index_paged(
        &dir,
        &params(Some(SortSpec {
            field: "created_at_key".into(),
            ascending: true,
        })),
        0.0,
        0,
        10,
        None,
    )
    .expect("the real field must still sort");
    assert!(!ok.hits.is_empty());

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_stored_only_sort_field_is_accepted() {
    // `rev_attr` is in the KB fixture's `.fnm` with the indexed flag CLEAR. This is the usual
    // shape of a sort field, and it is exactly what a check written against `indexed_fields`
    // would have rejected.
    let dir = temp_kb();
    let idx = ZslIndex::open(&dir).unwrap();
    assert!(idx.has_field("rev_attr"));
    assert!(
        !idx.indexed_fields().contains(&"rev_attr".to_string()),
        "fixture assumption: rev_attr is stored-only"
    );
    drop(idx);

    search_index_paged(
        &dir,
        &params(Some(SortSpec {
            field: "rev_attr".into(),
            ascending: false,
        })),
        0.0,
        0,
        10,
        None,
    )
    .expect("a stored-only field must be sortable");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn no_sort_spec_is_never_rejected() {
    let dir = temp_kb();
    search_index_paged(&dir, &params(None), 0.0, 0, 10, None).expect("relevance order is exempt");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn an_index_with_no_live_documents_does_not_reject_the_sort() {
    // The exemption exists so a sorted query over an empty index returns nothing instead of
    // erroring. Emptied by deleting every doc, since the writer cannot bootstrap a segment-less
    // index from nothing.
    let dir = temp_kb();
    {
        let idx = ZslIndex::open(&dir).unwrap();
        let n = idx.total_docs();
        drop(idx);
        let mut w = IndexWriter::open(&dir, WriterOpts::default()).unwrap();
        for d in 0..n {
            w.delete_document(d);
        }
        w.commit().unwrap();
    }
    let idx = ZslIndex::open(&dir).unwrap();
    assert_eq!(idx.num_docs(), 0, "every doc must be deleted");
    drop(idx);

    let out = search_index_paged(
        &dir,
        &params(Some(SortSpec {
            field: "no_such_field_anywhere".into(),
            ascending: true,
        })),
        0.0,
        0,
        10,
        None,
    )
    .expect("an empty index must return no hits, not an error");
    assert!(out.hits.is_empty());
    assert_eq!(out.total, 0);

    std::fs::remove_dir_all(&dir).ok();
}
