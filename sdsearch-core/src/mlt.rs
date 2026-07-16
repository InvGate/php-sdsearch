//! More Like This: pick a source doc's most distinctive terms (tf*idf) and
//! search for similar docs. Re-analyzes stored fields (no term vectors).

use crate::analysis::analyze;
use crate::index::IndexReader;
use crate::score::idf;
use std::collections::HashMap;

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
}
