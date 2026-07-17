//! Pseudo-relevance feedback (PRF) semantic search: a two-pass query that harvests
//! distinctive terms from the first pass's top hits and re-runs an augmented query.
//! Opt-in and self-contained — the plain search path is untouched. No index-time or
//! ambient cost; feedback-term selection reuses MLT's `select_terms` (and its
//! `max_doc_freq`/`posting_budget` memory guards), so a single PRF request cannot
//! load ~O(N) posting lists.

use crate::index::IndexReader;
use crate::mlt::{MltParams, select_terms};
use crate::query::{Occur, Query, QueryParams, build_query, search_with_weights};
use crate::search::Hit;
use std::collections::HashMap;

/// Parameters for a PRF search. Defaults: `top_k=5`, `num_terms=10`,
/// `feedback_weight=0.3`. `fields` empty = read all indexed fields. `max_doc_freq`/
/// `posting_budget` are `None` = infer safety defaults from collection size (see
/// `select_terms`).
#[derive(Debug, Clone)]
pub struct PrfParams {
    /// number of top hits from pass 1 treated as pseudo-relevant.
    pub top_k: usize,
    /// maximum number of feedback terms added to the augmented query.
    pub num_terms: usize,
    /// score multiplier for the feedback-term subtree (original terms stay at 1.0).
    pub feedback_weight: f32,
    /// source fields to harvest terms from; empty = all indexed fields.
    pub fields: Vec<String>,
    pub min_term_freq: u32,
    pub min_doc_freq: usize,
    pub max_doc_freq: Option<usize>,
    pub posting_budget: Option<usize>,
}

impl Default for PrfParams {
    fn default() -> Self {
        Self {
            top_k: 5,
            num_terms: 10,
            feedback_weight: 0.3,
            fields: Vec::new(),
            min_term_freq: 1,
            min_doc_freq: 1,
            max_doc_freq: None,
            posting_budget: None,
        }
    }
}

/// Runs a two-pass PRF search. Degrades gracefully to a plain search (never errors,
/// never worse than `query`, only slower) when PRF cannot contribute: `top_k==0`,
/// `num_terms==0`, no pass-1 hits, or no feedback terms survive the guards.
pub fn search_prf(
    index: &impl IndexReader,
    params: &QueryParams,
    prf: &PrfParams,
    min_score: f32,
    limit: usize,
) -> Vec<Hit> {
    let lim = if limit == 0 { usize::MAX } else { limit };
    let Ok(base) = build_query(params) else {
        return Vec::new();
    };

    // Plain search closure (the graceful-degradation fallback and the final pass share it).
    let plain = |q: &Query, ms: f32, l: usize| {
        search_with_weights(index, q, &params.field_weights, params.similarity, ms, l)
    };

    if prf.top_k == 0 || prf.num_terms == 0 {
        return plain(&base, min_score, lim);
    }

    // Pass 1: take the top_k best hits as pseudo-relevant (min_score 0.0 so feedback
    // isn't starved; the real min_score is applied on pass 2).
    let pass1 = plain(&base, 0.0, prf.top_k);
    if pass1.is_empty() {
        return Vec::new();
    }
    let doc_ids: Vec<usize> = pass1.iter().map(|h| h.id).collect();

    // Feedback-term selection reuses MLT's engine. Only the selection knobs matter;
    // the MLT-specific fields (filters/size/min_score) are left at no-op values.
    let fields = if prf.fields.is_empty() {
        index.indexed_fields()
    } else {
        prf.fields.clone()
    };
    let cfg = MltParams {
        fields,
        min_term_freq: prf.min_term_freq,
        max_query_terms: prf.num_terms,
        min_doc_freq: prf.min_doc_freq,
        max_doc_freq: prf.max_doc_freq,
        posting_budget: prf.posting_budget,
        timeout: None,
        term_filters: Vec::new(),
        range_filters: Vec::new(),
        min_should_match: None,
        field_weights: HashMap::new(),
        size: 0,
        min_score: 0.0,
    };
    let selected = select_terms(index, &doc_ids, &cfg);
    if selected.is_empty() {
        return plain(&base, min_score, lim);
    }

    // Augmented query: original at full weight (Should), feedback terms as a
    // down-weighted Should subtree. Feedback-only docs (vocabulary mismatch) enter
    // via the feedback subtree; docs matching both are boosted by the boolean coord.
    let feedback_clauses: Vec<(Occur, Query)> = selected
        .iter()
        .map(|s| {
            (
                Occur::Should,
                Query::Term {
                    field: Some(s.field.clone()),
                    text: s.term.clone(),
                },
            )
        })
        .collect();
    let feedback = Query::Boosted {
        boost: prf.feedback_weight,
        inner: Box::new(Query::Boolean {
            clauses: feedback_clauses,
        }),
    };
    let augmented = Query::Boolean {
        clauses: vec![(Occur::Should, base), (Occur::Should, feedback)],
    };
    plain(&augmented, min_score, lim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doc::{Document, FieldKind};
    use crate::index::MemoryIndex;
    use crate::query::search;
    use crate::score::Similarity;

    /// doc0 has the query term "printer" plus jargon; doc1 has only the jargon
    /// (vocabulary mismatch). Filler docs make the jargon distinctive (positive idf).
    fn corpus() -> MemoryIndex {
        let mut m = MemoryIndex::new();
        for text in [
            "printer paper jam toner",     // doc0: matches "printer"
            "paper jam toner replacement", // doc1: NO "printer", shares jargon
            "network cable ethernet",      // filler
            "monitor display hdmi",        // filler
            "keyboard mouse usb",          // filler
            "battery charger power",       // filler
        ] {
            let mut d = Document::new();
            d.add("body", text, FieldKind::Text);
            m.add_document(d);
        }
        m
    }

    // QueryParams deliberately has no `Default` (see query.rs) — mirrors the
    // explicit-field test helper used in query.rs's own test module.
    fn params(text: &str) -> QueryParams {
        QueryParams {
            text: text.into(),
            where_groups: vec![],
            in_groups: vec![],
            fuzzy_similarity: 0.5,
            fuzzy_prefix_len: 3,
            wildcard_min_prefix: 0,
            accent_insensitive: false,
            field_weights: HashMap::new(),
            similarity: Similarity::Bm25,
        }
    }

    fn ids(hits: &[Hit]) -> Vec<usize> {
        hits.iter().map(|h| h.id).collect()
    }

    #[test]
    fn prf_retrieves_vocabulary_mismatch_doc() {
        let idx = corpus();
        // Plain search for "printer" cannot reach doc1 (it has no "printer" token).
        let plain = search(&idx, &build_query(&params("printer")).unwrap(), 0.0, 100);
        assert!(
            !ids(&plain).contains(&1),
            "plain must miss doc1: {:?}",
            ids(&plain)
        );
        // PRF harvests paper/jam/toner from doc0 and surfaces doc1.
        let prf = search_prf(&idx, &params("printer"), &PrfParams::default(), 0.0, 100);
        assert!(
            ids(&prf).contains(&1),
            "PRF must retrieve doc1: {:?}",
            ids(&prf)
        );
    }

    #[test]
    fn prf_off_matches_plain_search() {
        // top_k = 0 disables feedback: identical result ids to a plain search.
        let idx = corpus();
        let off = PrfParams {
            top_k: 0,
            ..PrfParams::default()
        };
        let prf = search_prf(&idx, &params("printer"), &off, 0.0, 100);
        let plain = search(&idx, &build_query(&params("printer")).unwrap(), 0.0, 100);
        assert_eq!(ids(&prf), ids(&plain));
    }

    #[test]
    fn zero_num_terms_matches_plain_search() {
        let idx = corpus();
        let off = PrfParams {
            num_terms: 0,
            ..PrfParams::default()
        };
        let prf = search_prf(&idx, &params("printer"), &off, 0.0, 100);
        let plain = search(&idx, &build_query(&params("printer")).unwrap(), 0.0, 100);
        assert_eq!(ids(&prf), ids(&plain));
    }

    #[test]
    fn no_pass1_hits_returns_empty() {
        let idx = corpus();
        let prf = search_prf(
            &idx,
            &params("zzzznonexistent"),
            &PrfParams::default(),
            0.0,
            100,
        );
        assert!(prf.is_empty());
    }

    #[test]
    fn no_feedback_terms_degrades_to_plain() {
        // min_doc_freq far above the collection filters out every feedback candidate,
        // so PRF must fall back to plain-search results (not empty, not an error).
        let idx = corpus();
        let strict = PrfParams {
            min_doc_freq: 9999,
            ..PrfParams::default()
        };
        let prf = search_prf(&idx, &params("printer"), &strict, 0.0, 100);
        let plain = search(&idx, &build_query(&params("printer")).unwrap(), 0.0, 100);
        assert_eq!(ids(&prf), ids(&plain));
    }

    #[test]
    fn original_match_still_present() {
        // The doc that matched the original term must never be dropped by PRF.
        let idx = corpus();
        let prf = search_prf(&idx, &params("printer"), &PrfParams::default(), 0.0, 100);
        assert!(
            ids(&prf).contains(&0),
            "original match doc0 missing: {:?}",
            ids(&prf)
        );
    }
}
