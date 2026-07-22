//! runner: orchestrates ZslIndex + build_query + executor, reproducing the host
//! application's Zend Lucene search adapter (min_score filtering, limit==0 = unlimited).

use crate::mlt::{MltParams, more_like_this};
use crate::query::{
    InGroup, QueryParams, build_query, search, search_with_weights, search_with_weights_paged,
};
use crate::search::{Hit, SearchOutcome};
use crate::zsl::index::ZslIndex;
use std::path::Path;

/// Searches a ZSL index reproducing the host application's Zend Lucene search adapter:
/// build_query -> executor; filters min_score (`>=`), limit==0 = unlimited.
///
/// The legacy adapter had an empty-result fallback that re-parsed the query string. We do
/// NOT reproduce it: it kept the `where`/`in` filters required (an excluding filter is never
/// bypassed — see the oracle), and the only text it relaxed was already a subset of what
/// `text_subquery` matches in the primary pass. So the fallback could only ever return an
/// empty set here, while a naive text-only fallback would silently leak documents past an
/// excluding filter — which we must never do.
pub fn search_index(
    index_dir: &Path,
    params: &QueryParams,
    min_score: f32,
    limit: usize,
) -> Result<Vec<Hit>, Box<dyn std::error::Error>> {
    let index = ZslIndex::open(index_dir)?;
    let query = build_query(params)?;
    let lim = if limit == 0 { usize::MAX } else { limit };
    let hits = search_with_weights(
        &index,
        &query,
        &params.field_weights,
        params.similarity,
        min_score,
        lim,
    );
    Ok(hits)
}

/// Paged variant of `search_index`: returns the page `[offset, offset+limit)` plus the
/// (optionally capped) total match count. `limit == 0` = unlimited, as in `search_index`.
/// `total_cap`: `None` = exact count; `Some(cap)` = saturated at `cap`.
pub fn search_index_paged(
    index_dir: &Path,
    params: &QueryParams,
    min_score: f32,
    offset: usize,
    limit: usize,
    total_cap: Option<usize>,
) -> Result<SearchOutcome, Box<dyn std::error::Error>> {
    let index = ZslIndex::open(index_dir)?;
    let query = build_query(params)?;
    let lim = if limit == 0 { usize::MAX } else { limit };
    Ok(search_with_weights_paged(
        &index,
        &query,
        &params.field_weights,
        params.similarity,
        min_score,
        offset,
        lim,
        total_cap,
    ))
}

/// Resolves an id-field value to an internal doc id via an `InGroup` over
/// `<id_field>_key` (build_query adds the suffix), like the writer's resolver.
fn resolve_reference_doc(
    index: &ZslIndex,
    id_field: &str,
    id_value: &str,
) -> Result<Option<usize>, Box<dyn std::error::Error>> {
    let params = QueryParams {
        text: String::new(),
        where_groups: Vec::new(),
        in_groups: vec![InGroup {
            field: id_field.to_string(),
            values: vec![id_value.to_string()],
        }],
        fuzzy_similarity: 0.5,
        fuzzy_prefix_len: 3,
        wildcard_min_prefix: 0,
        accent_insensitive: false,
        field_weights: std::collections::HashMap::new(),
        similarity: crate::score::Similarity::Bm25,
    };
    let query = build_query(&params)?;
    let hits = search(index, &query, 0.0, 1);
    Ok(hits.first().map(|h| h.id))
}

/// Opens a ZSL index, resolves the reference id-field value to an internal doc id,
/// and runs a More Like This query. Returns an empty vec if the reference doc is not found.
pub fn more_like_this_index(
    index_dir: &Path,
    id_field: &str,
    id_value: &str,
    params: &MltParams,
) -> Result<Vec<Hit>, Box<dyn std::error::Error>> {
    let index = ZslIndex::open(index_dir)?;
    match resolve_reference_doc(&index, id_field, id_value)? {
        Some(doc_id) => Ok(more_like_this(&index, doc_id, params)),
        None => Ok(Vec::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::InGroup;
    use std::path::PathBuf;

    fn multiseg() -> PathBuf {
        PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/zsl_index_multiseg"
        ))
    }
    fn params(text: &str) -> QueryParams {
        QueryParams {
            text: text.into(),
            where_groups: vec![],
            in_groups: vec![],
            fuzzy_similarity: 0.5,
            fuzzy_prefix_len: 3,
            wildcard_min_prefix: 0,
            accent_insensitive: false,
            field_weights: std::collections::HashMap::new(),
            similarity: crate::score::Similarity::Bm25,
        }
    }
    fn ids(hits: &[Hit]) -> Vec<usize> {
        let mut v: Vec<usize> = hits.iter().map(|h| h.id).collect();
        v.sort_unstable();
        v
    }

    // Bootstrap from the KB fixture (the writer only appends) and add two docs carrying a
    // `cat_key` keyword filter field and a `body` text field: cat 1 -> "alpha", cat 2 ->
    // "zebra". So "zebra" exists ONLY outside cat 1 -> a cat=1 filter must exclude it.
    fn temp_index_with_cat_docs(tag: &str) -> PathBuf {
        let src = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/zsl_index_kb"
        ));
        let dir =
            std::env::temp_dir().join(format!("sdsearch_catfilter_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for entry in std::fs::read_dir(&src).unwrap() {
            let p = entry.unwrap().path();
            if p.is_file() {
                std::fs::copy(&p, dir.join(p.file_name().unwrap())).unwrap();
            }
        }
        let mut w = IndexWriter::open(&dir, WriterOpts::default()).unwrap();
        for (cat, body) in [("1", "alpha"), ("2", "zebra")] {
            w.add_document(WriterDoc {
                fields: vec![
                    WriterField::keyword("cat_key", cat),
                    WriterField::text("body", body),
                ],
            })
            .unwrap();
        }
        w.commit().unwrap();
        dir
    }

    // Bootstrap from the KB fixture and add a doc whose body carries a colon-bearing token.
    // The analyzer keeps ':' inside a token, so "c:drive" is a SINGLE indexed term.
    fn temp_index_with_punct_doc(tag: &str) -> PathBuf {
        let src = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/zsl_index_kb"
        ));
        let dir = std::env::temp_dir().join(format!("sdsearch_punct_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for entry in std::fs::read_dir(&src).unwrap() {
            let p = entry.unwrap().path();
            if p.is_file() {
                std::fs::copy(&p, dir.join(p.file_name().unwrap())).unwrap();
            }
        }
        let mut w = IndexWriter::open(&dir, WriterOpts::default()).unwrap();
        w.add_document(WriterDoc {
            fields: vec![WriterField::text("body", "c:drive restore")],
        })
        .unwrap();
        w.commit().unwrap();
        dir
    }

    #[test]
    fn prefix_reaches_a_colon_bearing_token() {
        // Dropping the query-operator escaping lets a prefix like "c:dr" reach the indexed
        // token "c:drive" through the wildcard leaf. The old escaping (`c\:dr*`) suppressed
        // this because no indexed term carries a backslash.
        let dir = temp_index_with_punct_doc("colon");
        let hits = search_index(&dir, &params("c:dr"), 0.0, 0).unwrap();
        assert!(
            !hits.is_empty(),
            "prefix 'c:dr' should reach the 'c:drive' token via the wildcard leaf"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn text_only_matches_across_segments() {
        // "vpn" crosses segments -> [0,2] (same doc-set as the text-only boolean oracle).
        let hits = search_index(&multiseg(), &params("vpn"), 0.0, 0).unwrap();
        assert_eq!(ids(&hits), vec![0, 2]);
    }

    #[test]
    fn empty_primary_with_excluding_filter_stays_empty() {
        // text "vpn" (Must) + in cat=999 (Must, no doc) => empty primary. An excluding
        // filter must NOT be relaxed: the result stays empty rather than leaking the
        // text-only "vpn" matches [0,2] (which is what the removed fallback used to do).
        let mut p = params("vpn");
        p.in_groups = vec![InGroup {
            field: "cat".into(),
            values: vec!["999".into()],
        }];
        let hits = search_index(&multiseg(), &p, 0.0, 0).unwrap();
        assert!(
            hits.is_empty(),
            "excluding filter must not be bypassed: {:?}",
            ids(&hits)
        );
    }

    #[test]
    fn empty_primary_never_bypasses_an_in_filter() {
        // "zebra" lives ONLY in the cat=2 doc; a cat=1 filter must exclude it.
        // The empty-primary path must NOT relax the filter and leak the cat=2 doc
        // (parity with the legacy Zend adapter, whose fallback keeps the filter required).
        let dir = temp_index_with_cat_docs("infilter");
        let mut p = params("zebra");
        p.in_groups = vec![InGroup {
            field: "cat".into(),
            values: vec!["1".into()],
        }];
        let hits = search_index(&dir, &p, 0.0, 0).unwrap();
        assert!(
            hits.is_empty(),
            "cat=1 filter must exclude the cat=2 'zebra' doc, got {} hit(s)",
            hits.len()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn text_matches_the_cat_doc_when_unfiltered() {
        // Guard for the test above: proves "zebra" really does match a doc, so the
        // empty result there comes from the filter, not from the term being absent.
        let dir = temp_index_with_cat_docs("unfiltered");
        let hits = search_index(&dir, &params("zebra"), 0.0, 0).unwrap();
        assert!(
            !hits.is_empty(),
            "zebra should match the cat=2 doc unfiltered"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn limit_zero_is_unlimited() {
        // "how" matches the two "how to ..." docs even with limit=0.
        let hits = search_index(&multiseg(), &params("how"), 0.0, 0).unwrap();
        assert!(hits.len() >= 2, "limit=0 must return all matches");
    }

    #[test]
    fn search_index_paged_reports_total_and_offset() {
        // "how" matches the two "how to ..." docs in the multiseg fixture.
        let full = search_index_paged(&multiseg(), &params("how"), 0.0, 0, 0, None).unwrap();
        assert!(
            full.total >= 2,
            "total counts all matches, got {}",
            full.total
        );
        assert!(!full.total_capped);
        let full_ids = ids(&full.hits);

        // offset 1 with a large limit drops exactly the first hit of the ranking.
        let paged = search_index_paged(&multiseg(), &params("how"), 0.0, 1, 100, None).unwrap();
        assert_eq!(paged.hits.len(), full.hits.len() - 1);
        assert_eq!(paged.total, full.total, "total is independent of the page");

        // a cap of 1 saturates the total and flags it, without changing the page size.
        let capped = search_index_paged(&multiseg(), &params("how"), 0.0, 0, 100, Some(1)).unwrap();
        assert_eq!(capped.total, 1);
        assert!(capped.total_capped);
        assert_eq!(ids(&capped.hits), full_ids, "cap bounds total, not hits");
    }

    use crate::mlt::MltParams;
    use crate::zsl::writer::{IndexWriter, WriterDoc, WriterField, WriterOpts};
    use std::collections::HashMap as StdHashMap;

    fn mlt_params(fields: &[&str]) -> MltParams {
        MltParams {
            fields: fields.iter().map(|s| (*s).to_string()).collect(),
            min_term_freq: 1,
            max_query_terms: 25,
            min_doc_freq: 1,
            max_doc_freq: Some(0),
            posting_budget: Some(0),
            timeout: None,
            term_filters: Vec::new(),
            range_filters: Vec::new(),
            min_should_match: None,
            field_weights: StdHashMap::new(),
            size: 10,
            min_score: 0.0,
        }
    }

    // The writer only appends to an EXISTING index, so bootstrap by copying the KB
    // fixture to a temp dir, then append 3 docs carrying an `id_key` keyword and a
    // `body` text field that shares a rare term ("zebra" in A and B only). The KB
    // docs use a `title` field, so they never collide with our `body` postings.
    fn temp_index_with_mlt_docs() -> PathBuf {
        let src = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/zsl_index_kb"
        ));
        let dir = std::env::temp_dir().join(format!("sdsearch_mlt_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for entry in std::fs::read_dir(&src).unwrap() {
            let p = entry.unwrap().path();
            if p.is_file() {
                std::fs::copy(&p, dir.join(p.file_name().unwrap())).unwrap();
            }
        }
        let mut w = IndexWriter::open(&dir, WriterOpts::default()).unwrap();
        for (id, body) in [
            ("A", "zebra alpha"),
            ("B", "zebra beta"),
            ("C", "cat gamma"),
        ] {
            w.add_document(WriterDoc {
                fields: vec![
                    WriterField::keyword("id_key", id),
                    WriterField::text("body", body),
                ],
            })
            .unwrap();
        }
        w.commit().unwrap();
        dir
    }

    #[test]
    fn more_like_this_index_finds_similar_and_excludes_source() {
        let dir = temp_index_with_mlt_docs();

        // reference doc "A" (resolved via id -> id_key) shares "zebra" with "B" only.
        let hits = more_like_this_index(&dir, "id", "A", &mlt_params(&["body"])).unwrap();
        let ids: Vec<String> = hits
            .iter()
            .filter_map(|h| h.fields.get("id_key").cloned())
            .collect();
        assert!(ids.contains(&"B".to_string()), "expected B among {ids:?}");
        assert!(
            !ids.contains(&"A".to_string()),
            "source A must be excluded: {ids:?}"
        );

        // unknown reference id -> empty, not an error.
        let none = more_like_this_index(&dir, "id", "ZZZ", &mlt_params(&["body"])).unwrap();
        assert!(none.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
