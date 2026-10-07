//! Field sort over many matches reads the sort values from the field's terms
//! (`IndexReader::numeric_sort_values`) instead of one `.fdt` read per match. Over a REAL ZSL
//! index, because what decides whether the table is trustworthy — stored-only fields, segment
//! boundaries, repeated values — is what `MemoryIndex` does not model.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

use sdsearch_core::index::IndexReader;
use sdsearch_core::query::QueryParams;
use sdsearch_core::score::Similarity;
use sdsearch_core::search::SortSpec;
use sdsearch_core::zsl::index::ZslIndex;
use sdsearch_core::zsl::runner::search_index_paged;
use sdsearch_core::zsl::writer::{IndexWriter, WriterDoc, WriterField, WriterOpts};

static COUNTER: AtomicU32 = AtomicU32::new(0);

/// The KB fixture (20 docs, none with the test fields) plus `batches`, one commit — so one
/// segment — per batch.
fn temp_index(batches: &[&[Vec<WriterField>]]) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("sdsearch_sorttable_{}_{}", std::process::id(), n));
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
    for batch in batches {
        let mut w = IndexWriter::open(&dir, WriterOpts::default()).unwrap();
        for fields in *batch {
            w.add_document(WriterDoc {
                fields: fields.clone(),
            })
            .unwrap();
        }
        w.commit().unwrap();
    }
    dir
}

fn doc(n: Option<&str>) -> Vec<WriterField> {
    let mut f = vec![WriterField::text("body", "zebra")];
    if let Some(v) = n {
        f.push(WriterField::keyword("n_key", v));
    }
    f
}

#[test]
fn the_table_holds_what_stored_value_holds_for_every_doc() {
    // variable width, a doc without the field, and two segments on top of the KB one
    let dir = temp_index(&[
        &[doc(Some("30")), doc(Some("4")), doc(None), doc(Some("100"))],
        &[doc(Some("7")), doc(Some("-2"))],
    ]);
    let index = ZslIndex::open(&dir).unwrap();
    let table = index
        .numeric_sort_values("n_key")
        .expect("a numeric keyword field must produce a table");
    assert_eq!(table.len(), index.total_docs());
    for (id, value) in table.iter().enumerate() {
        let stored = index
            .stored_value(id, "n_key")
            .map(|v| v.parse::<i64>().unwrap());
        assert_eq!(*value, stored, "doc {id}");
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn no_table_where_the_terms_would_order_differently_from_the_stored_values() {
    let dir = temp_index(&[&[vec![
        WriterField::keyword("s_key", "open"),
        WriterField::keyword("m_key", "1"),
        WriterField::keyword("m_key", "2"),
        WriterField::unindexed("u_key", "5"),
    ]]]);
    let index = ZslIndex::open(&dir).unwrap();
    assert_eq!(
        index.numeric_sort_values("s_key"),
        None,
        "non-numeric value"
    );
    assert_eq!(
        index.numeric_sort_values("m_key"),
        None,
        "two values in one doc"
    );
    assert_eq!(
        index.numeric_sort_values("u_key"),
        None,
        "stored-only: no terms"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_sort_over_many_matches_orders_by_value_with_missing_last() {
    let dir = temp_index(&[
        &[doc(Some("30")), doc(Some("4")), doc(None), doc(Some("100"))],
        &[doc(Some("7")), doc(Some("-2"))],
    ]);
    let index = ZslIndex::open(&dir).unwrap();
    let sorted = |ascending: bool| -> Vec<Option<String>> {
        let p = QueryParams {
            text: "zebra".into(),
            where_groups: vec![],
            in_groups: vec![],
            range_filters: vec![],
            match_all: vec![],
            fuzzy_similarity: 0.5,
            fuzzy_prefix_len: 3,
            wildcard_min_prefix: 2,
            accent_insensitive: false,
            synonyms: false,
            field_weights: HashMap::new(),
            similarity: Similarity::Bm25,
            sort: Some(SortSpec {
                field: "n_key".into(),
                ascending,
            }),
            boolean_tree: None,
        };
        search_index_paged(&dir, &p, 0.0, 0, 10, None)
            .unwrap()
            .hits
            .iter()
            .map(|h| index.stored_value(h.id, "n_key"))
            .collect()
    };
    let v = |s: &str| Some(s.to_string());
    assert_eq!(
        sorted(true),
        [v("-2"), v("4"), v("7"), v("30"), v("100"), None]
    );
    assert_eq!(
        sorted(false),
        [v("100"), v("30"), v("7"), v("4"), v("-2"), None]
    );
    std::fs::remove_dir_all(&dir).ok();
}
