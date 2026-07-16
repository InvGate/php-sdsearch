//! More Like This: pick a source doc's most distinctive terms (tf*idf) and
//! search for similar docs. Re-analyzes stored fields (no term vectors).

use crate::analysis::analyze;
use crate::index::IndexReader;
use crate::score::idf;
use crate::search::{Hit, finalize, term_scores};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

/// Parameters for a More Like This query. `max_doc_freq == 0` = unbounded;
/// `max_query_terms == 0` = no cap; `posting_budget == 0` = off; `size == 0` = unlimited.
#[derive(Debug, Clone)]
pub struct MltParams {
    pub fields: Vec<String>,
    pub min_term_freq: u32,
    pub max_query_terms: usize,
    pub min_doc_freq: usize,
    pub max_doc_freq: usize,
    pub posting_budget: usize,
    pub timeout: Option<std::time::Duration>,
    pub term_filters: Vec<(String, String)>,
    pub field_weights: HashMap<String, f32>,
    pub size: usize,
    pub min_score: f32,
}

/// A selected candidate term for the MLT query, with its collection doc frequency.
#[derive(Debug, Clone)]
pub(crate) struct Selected {
    pub field: String,
    pub term: String,
    pub doc_freq: usize,
}

/// Extracts the source doc's most distinctive terms: reads its stored text per
/// requested field, counts in-doc term frequency, scores each candidate by
/// `tf * idf`, filters by the freq knobs, ranks, caps at `max_query_terms`, then
/// applies the posting budget (always keeping at least the top term).
pub(crate) fn select_terms(
    index: &impl IndexReader,
    source_doc: usize,
    p: &MltParams,
) -> Vec<Selected> {
    let stored = index.stored_fields(source_doc);
    let n = index.total_docs() as f32;
    let mut scored: Vec<(f32, Selected)> = Vec::new();

    for field in &p.fields {
        let Some(text) = stored.get(field) else { continue };
        let mut tf: HashMap<String, u32> = HashMap::new();
        for tok in analyze(text) {
            *tf.entry(tok).or_insert(0) += 1;
        }
        for (term, freq) in tf {
            if freq < p.min_term_freq {
                continue;
            }
            let df = index.doc_freq(field, &term);
            if df < p.min_doc_freq {
                continue;
            }
            if p.max_doc_freq > 0 && df > p.max_doc_freq {
                continue;
            }
            let sel = freq as f32 * idf(n, df as f32);
            scored.push((
                sel,
                Selected {
                    field: field.clone(),
                    term,
                    doc_freq: df,
                },
            ));
        }
    }

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    if p.max_query_terms > 0 {
        scored.truncate(p.max_query_terms);
    }

    let mut out: Vec<Selected> = Vec::new();
    let mut spent = 0usize;
    for (_, s) in scored {
        if p.posting_budget > 0 && !out.is_empty() && spent + s.doc_freq > p.posting_budget {
            break;
        }
        spent += s.doc_freq;
        out.push(s);
    }
    out
}

/// field weight (defaults to 1.0 when not listed) — local mirror of the query module's helper.
fn weight_of(weights: &HashMap<String, f32>, field: &str) -> f32 {
    weights.get(field).copied().unwrap_or(1.0)
}

/// Runs a More Like This query for `source_doc`: selects distinctive terms, unions
/// their postings (per-field weighted) into a score map, excludes the source doc,
/// applies term filters (each must match), normalizes the top hit to 1.0, and
/// finalizes (filter min_score, sort, truncate to `size`, hydrate stored fields).
///
/// Best-effort timeout: the top-ranked term is always processed; before each
/// subsequent term the wall-clock deadline is checked and the union stops early
/// if it has passed. Early stops yield approximate scores (a runaway guard).
pub fn more_like_this(index: &impl IndexReader, source_doc: usize, p: &MltParams) -> Vec<Hit> {
    let selected = select_terms(index, source_doc, p);
    if selected.is_empty() {
        return Vec::new();
    }

    let deadline = p.timeout.map(|d| Instant::now() + d);
    let mut score: HashMap<usize, f32> = HashMap::new();
    for (i, s) in selected.iter().enumerate() {
        if i > 0 {
            if let Some(dl) = deadline {
                if Instant::now() >= dl {
                    break;
                }
            }
        }
        let w = weight_of(&p.field_weights, &s.field);
        for (id, sc) in term_scores(index, &s.field, &s.term) {
            *score.entry(id).or_insert(0.0) += sc * w;
        }
    }

    score.remove(&source_doc);

    for (field, value) in &p.term_filters {
        let matching: HashSet<usize> = index
            .postings_for(field, value)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        score.retain(|id, _| matching.contains(id));
    }

    let top = score.values().copied().fold(0.0f32, f32::max);
    let normalized: Vec<(usize, f32)> = if top > 0.0 {
        score.into_iter().map(|(id, s)| (id, s / top)).collect()
    } else {
        score.into_iter().collect()
    };

    let limit = if p.size == 0 { usize::MAX } else { p.size };
    finalize(index, normalized, p.min_score, limit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doc::{Document, FieldKind};
    use crate::index::MemoryIndex;

    fn params(fields: &[&str]) -> MltParams {
        MltParams {
            fields: fields.iter().map(|s| (*s).to_string()).collect(),
            min_term_freq: 1,
            max_query_terms: 25,
            min_doc_freq: 1,
            max_doc_freq: 0,
            posting_budget: 0,
            timeout: None,
            term_filters: Vec::new(),
            field_weights: HashMap::new(),
            size: 10,
            min_score: 0.0,
        }
    }

    // doc 0 is the source; "zebra" is rare (df 1 -> only the source) while "the"
    // is common (df 6 -> every doc, so low idf). Selection must prefer the
    // distinctive term. The collection is deliberately large enough that the rare
    // term's idf outweighs the common term — the tf*idf shape needs some spread.
    fn idx() -> MemoryIndex {
        let mut m = MemoryIndex::new();
        for text in ["zebra the", "the cat", "the dog", "the fish", "the bird", "the frog"] {
            let mut d = Document::new();
            d.add("body", text, FieldKind::Text);
            m.add_document(d);
        }
        m
    }

    #[test]
    fn selects_distinctive_terms_over_common_ones() {
        let m = idx();
        let terms = select_terms(&m, 0, &params(&["body"]));
        let picked: Vec<&str> = terms.iter().map(|t| t.term.as_str()).collect();
        assert!(picked.contains(&"zebra"), "expected 'zebra', got {picked:?}");
        // "zebra" (rarer) must rank above "the" (common)
        assert_eq!(terms[0].term, "zebra");
    }

    #[test]
    fn min_doc_freq_filters_rare_terms() {
        let m = idx();
        let mut p = params(&["body"]);
        p.min_doc_freq = 2; // "zebra" has df 1 -> filtered out
        let picked: Vec<String> = select_terms(&m, 0, &p).into_iter().map(|t| t.term).collect();
        assert!(!picked.contains(&"zebra".to_string()), "got {picked:?}");
    }

    #[test]
    fn max_query_terms_caps_the_selection() {
        let m = idx();
        let mut p = params(&["body"]);
        p.max_query_terms = 1;
        assert_eq!(select_terms(&m, 0, &p).len(), 1);
    }

    #[test]
    fn posting_budget_keeps_at_least_the_top_term() {
        let m = idx();
        let mut p = params(&["body"]);
        p.posting_budget = 1; // tiny budget; still keep the single top term
        let terms = select_terms(&m, 0, &p);
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].term, "zebra");
    }

    // doc 0 shares "zebra" with doc 1 only; doc 2/3 are unrelated.
    fn sim_idx() -> MemoryIndex {
        let mut m = MemoryIndex::new();
        for text in ["zebra alpha", "zebra beta", "cat gamma", "dog delta"] {
            let mut d = Document::new();
            d.add("body", text, FieldKind::Text);
            m.add_document(d);
        }
        m
    }

    #[test]
    fn returns_similar_docs_excluding_source() {
        let m = sim_idx();
        let hits = more_like_this(&m, 0, &params(&["body"]));
        let ids: Vec<usize> = hits.iter().map(|h| h.id).collect();
        assert!(!ids.contains(&0), "source doc must be excluded: {ids:?}");
        assert!(ids.contains(&1), "doc 1 shares 'zebra': {ids:?}");
    }

    #[test]
    fn empty_when_no_terms_survive_filters() {
        let m = sim_idx();
        let mut p = params(&["body"]);
        p.min_term_freq = 99; // nothing qualifies
        assert!(more_like_this(&m, 0, &p).is_empty());
    }

    #[test]
    fn term_filters_restrict_results() {
        let mut m = MemoryIndex::new();
        // doc 0 source; docs 1 and 2 both share "zebra" but differ on status_key.
        for (body, status) in [
            ("zebra alpha", "open"),
            ("zebra beta", "open"),
            ("zebra gamma", "closed"),
        ] {
            let mut d = Document::new();
            d.add("body", body, FieldKind::Text);
            d.add("status_key", status, FieldKind::Keyword);
            m.add_document(d);
        }
        let mut p = params(&["body"]);
        p.term_filters = vec![("status_key".to_string(), "open".to_string())];
        let ids: Vec<usize> = more_like_this(&m, 0, &p).iter().map(|h| h.id).collect();
        assert!(ids.contains(&1), "doc 1 is open: {ids:?}");
        assert!(!ids.contains(&2), "doc 2 is closed, filtered out: {ids:?}");
    }

    #[test]
    fn zero_timeout_returns_early_without_error() {
        use std::time::Duration;
        // source doc shares two distinct terms with two different docs; a zero deadline
        // must still process the single top term (best effort), not panic or return empty.
        let mut m = MemoryIndex::new();
        for text in ["zebra quokka", "zebra only", "quokka only", "unrelated"] {
            let mut d = Document::new();
            d.add("body", text, FieldKind::Text);
            m.add_document(d);
        }
        let mut p = params(&["body"]);
        p.timeout = Some(Duration::from_millis(0));
        let hits = more_like_this(&m, 0, &p);
        assert!(
            !hits.is_empty(),
            "best-effort timeout must still return the top term's matches"
        );
        assert!(hits.iter().all(|h| h.id != 0), "source still excluded");
    }
}
