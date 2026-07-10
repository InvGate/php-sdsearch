//! tf-idf scoring in the style of Lucene DefaultSimilarity.
//! the goal is NOT to match ZSL's exact floats, but to preserve the
//! tf-idf shape: higher frequency raises the score, a longer field lowers it.

use crate::index::IndexReader;

/// idf of a term. `total_docs` is the collection's N (incl. deletes, like ZSL).
/// Computed ONCE per term (not per doc): it is constant over the posting list.
pub fn idf(total_docs: f32, doc_freq: f32) -> f32 {
    1.0 + (total_docs / (doc_freq + 1.0)).ln()
}

/// score of a doc given an already-computed idf, the term-freq, and the field length.
/// Separates the per-term part (idf) from the per-doc part (tf, field_norm) to avoid
/// recomputing idf/doc_freq on every posting.
pub fn score_with_idf(idf: f32, term_freq: u32, field_len: u32) -> f32 {
    if term_freq == 0 {
        return 0.0;
    }
    let tf = (term_freq as f32).sqrt();
    let field_norm = 1.0 / (field_len.max(1) as f32).sqrt();
    tf * idf * idf * field_norm
}

/// tf-idf score of a (term, doc). Convenience for callers that score a single doc;
/// the hot path uses `idf` + `score_with_idf` to hoist the idf per term.
pub fn score_term(
    index: &impl IndexReader,
    field: &str,
    term: &str,
    doc_id: usize,
    term_freq: u32,
) -> f32 {
    let idf = idf(
        index.total_docs() as f32,
        index.doc_freq(field, term) as f32,
    );
    score_with_idf(idf, term_freq, index.field_len(doc_id, field))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doc::{Document, FieldKind};
    use crate::index::MemoryIndex;

    fn index_with(docs: &[&str]) -> MemoryIndex {
        let mut idx = MemoryIndex::new();
        for text in docs {
            let mut d = Document::new();
            d.add("body", text, FieldKind::Text);
            idx.add_document(d);
        }
        idx
    }

    #[test]
    fn higher_term_frequency_scores_higher() {
        // both docs contain "foo"; the doc with more tf (and same relative length) scores higher
        let idx = index_with(&["foo foo foo x", "foo x y z"]);
        let hi = score_term(&idx, "body", "foo", 0, 3);
        let lo = score_term(&idx, "body", "foo", 1, 1);
        assert!(hi > lo, "hi={hi} lo={lo}");
    }

    #[test]
    fn shorter_field_scores_higher_for_same_tf() {
        // same tf=1, shorter field => higher field_norm => higher score
        let idx = index_with(&["foo", "foo a b c d e"]);
        let short = score_term(&idx, "body", "foo", 0, 1);
        let long = score_term(&idx, "body", "foo", 1, 1);
        assert!(short > long, "short={short} long={long}");
    }

    #[test]
    fn missing_term_scores_zero() {
        let idx = index_with(&["foo"]);
        assert_eq!(score_term(&idx, "body", "foo", 0, 0), 0.0);
    }
}
