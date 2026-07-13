//! runner: orchestrates ZslIndex + build_query + executor, reproducing the host
//! application's Zend Lucene search adapter (empty-result fallback, min_score,
//! limit==0 = unlimited).

use crate::analysis::analyze;
use crate::query::{Occur, Query, QueryParams, build_query, search_with_weights};
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
}
