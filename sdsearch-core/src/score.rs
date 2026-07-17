//! tf-idf scoring in the style of Lucene DefaultSimilarity.
//! the goal is NOT to match ZSL's exact floats, but to preserve the
//! tf-idf shape: higher frequency raises the score, a longer field lowers it.

use crate::index::IndexReader;

/// idf of a term. `total_docs` is the collection's N (incl. deletes, like ZSL).
/// Computed ONCE per term (not per doc): it is constant over the posting list.
pub fn idf(total_docs: f32, doc_freq: f32) -> f32 {
    1.0 + (total_docs / (doc_freq + 1.0)).ln()
}

/// Selectable relevance scoring. `Bm25` is the default; `TfIdf` preserves the
/// legacy Lucene DefaultSimilarity shape. `k1`/`b` are fixed at the standard values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Similarity {
    #[default]
    Bm25,
    TfIdf,
}

impl Similarity {
    /// idf of a term. `total_docs` is the collection N (incl. deletes, like ZSL).
    pub fn idf(self, total_docs: f32, doc_freq: f32) -> f32 {
        match self {
            Similarity::TfIdf => idf(total_docs, doc_freq),
            // BM25 probabilistic idf. `.max(0.0)` guards the rare df > N/2 case
            // (would go slightly negative) so a matched term never subtracts.
            Similarity::Bm25 => (1.0 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
                .ln()
                .max(0.0),
        }
    }

    /// score of a (term, doc) given the already-computed idf, term-freq, this
    /// doc's field length, and the collection's average field length (BM25 only;
    /// TfIdf ignores it).
    pub fn score(self, idf: f32, term_freq: u32, field_len: u32, avg_field_len: f32) -> f32 {
        if term_freq == 0 {
            return 0.0;
        }
        match self {
            Similarity::TfIdf => {
                let tf = (term_freq as f32).sqrt();
                let field_norm = 1.0 / (field_len.max(1) as f32).sqrt();
                tf * idf * idf * field_norm
            }
            Similarity::Bm25 => {
                const K1: f32 = 1.2;
                const B: f32 = 0.75;
                let tf = term_freq as f32;
                let avg = if avg_field_len > 0.0 {
                    avg_field_len
                } else {
                    1.0
                };
                let len_ratio = field_len.max(1) as f32 / avg;
                let denom = tf + K1 * (1.0 - B + B * len_ratio);
                idf * (tf * (K1 + 1.0) / denom)
            }
        }
    }
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

    #[test]
    fn bm25_idf_decreases_with_doc_freq() {
        // rarer term (lower df) has higher idf
        let rare = Similarity::Bm25.idf(1000.0, 1.0);
        let common = Similarity::Bm25.idf(1000.0, 500.0);
        assert!(rare > common, "rare={rare} common={common}");
    }

    #[test]
    fn bm25_tf_saturates() {
        // TF-IDF grows ~sqrt(tf) unbounded; BM25 saturates. A doc with tf=50 must NOT
        // score 50x a tf=1 doc. With k1=1.2 the tf factor caps below (k1+1)=2.2.
        let idf = 1.0;
        let s1 = Similarity::Bm25.score(idf, 1, 10, 10.0);
        let s50 = Similarity::Bm25.score(idf, 50, 10, 10.0);
        assert!(s50 > s1, "more tf still scores higher");
        assert!(s50 < 10.0 * s1, "but saturates: s50={s50} s1={s1}");
    }

    #[test]
    fn bm25_longer_field_scores_lower_for_same_tf() {
        let idf = 1.0;
        let short = Similarity::Bm25.score(idf, 2, 3, 10.0);
        let long = Similarity::Bm25.score(idf, 2, 30, 10.0);
        assert!(short > long, "short={short} long={long}");
    }

    #[test]
    fn bm25_zero_tf_scores_zero() {
        assert_eq!(Similarity::Bm25.score(1.0, 0, 5, 10.0), 0.0);
    }

    #[test]
    fn tfidf_method_matches_legacy_shape() {
        // the TfIdf variant preserves the old behavior: sqrt(tf) * idf^2 * 1/sqrt(len)
        let idf = idf(3.0, 1.0); // free fn, unchanged
        let got = Similarity::TfIdf.score(idf, 3, 4, 999.0); // avg ignored by TfIdf
        let want = (3.0_f32).sqrt() * idf * idf * (1.0 / (4.0_f32).sqrt());
        assert!((got - want).abs() < 1e-6, "got={got} want={want}");
    }

    #[test]
    fn default_similarity_is_bm25() {
        assert_eq!(Similarity::default(), Similarity::Bm25);
    }
}
