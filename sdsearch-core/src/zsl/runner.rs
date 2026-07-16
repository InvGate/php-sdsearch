//! runner: orchestrates ZslIndex + build_query + executor, reproducing the host
//! application's Zend Lucene search adapter (empty-result fallback, min_score,
//! limit==0 = unlimited).

use crate::analysis::analyze;
use crate::mlt::{MltParams, more_like_this};
use crate::query::{InGroup, Occur, Query, QueryParams, build_query, search, search_with_weights};
use crate::search::Hit;
use crate::zsl::index::ZslIndex;
use std::collections::HashSet;
use std::path::Path;

/// Searches a ZSL index reproducing the host application's Zend Lucene search adapter:
/// build_query -> executor -> if empty, all-fields fallback; filters min_score (`>=`),
/// limit==0 = unlimited.
pub fn search_index(
    index_dir: &Path,
    params: &QueryParams,
    min_score: f32,
    limit: usize,
) -> Result<Vec<Hit>, Box<dyn std::error::Error>> {
    let index = ZslIndex::open(index_dir)?;
    let query = build_query(params)?;
    let lim = if limit == 0 { usize::MAX } else { limit };
    let mut hits = search_with_weights(&index, &query, &params.field_weights, min_score, lim);
    if hits.is_empty() {
        if let Some(fb) = fallback_query(&params.text) {
            hits = search_with_weights(&index, &fb, &params.field_weights, min_score, lim);
        }
    }
    Ok(hits)
}

/// Search-adapter fallback: an all-fields Boolean of terms (Should) over the UNIQUE
/// tokens of the text (dedup by text, like the re-parsed MultiTerm in the search adapter).
fn fallback_query(text: &str) -> Option<Query> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut clauses: Vec<(Occur, Query)> = Vec::new();
    for tok in analyze(text) {
        if seen.insert(tok.clone()) {
            clauses.push((
                Occur::Should,
                Query::Term {
                    field: None,
                    text: tok,
                },
            ));
        }
    }
    if clauses.is_empty() {
        None
    } else {
        Some(Query::Boolean { clauses })
    }
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
        }
    }
    fn ids(hits: &[Hit]) -> Vec<usize> {
        let mut v: Vec<usize> = hits.iter().map(|h| h.id).collect();
        v.sort_unstable();
        v
    }

    #[test]
    fn text_only_matches_across_segments() {
        // "vpn" crosses segments -> [0,2] (same doc-set as the text-only boolean oracle).
        let hits = search_index(&multiseg(), &params("vpn"), 0.0, 0).unwrap();
        assert_eq!(ids(&hits), vec![0, 2]);
    }

    #[test]
    fn empty_primary_triggers_fallback() {
        // text "vpn" (Must) + in cat=999 (Must, no doc) => empty primary;
        // the all-fields fallback "vpn" recovers [0,2].
        let mut p = params("vpn");
        p.in_groups = vec![InGroup {
            field: "cat".into(),
            values: vec!["999".into()],
        }];
        let hits = search_index(&multiseg(), &p, 0.0, 0).unwrap();
        assert_eq!(ids(&hits), vec![0, 2]);
    }

    #[test]
    fn limit_zero_is_unlimited() {
        // "how" matches the two "how to ..." docs even with limit=0.
        let hits = search_index(&multiseg(), &params("how"), 0.0, 0).unwrap();
        assert!(hits.len() >= 2, "limit=0 must return all matches");
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
