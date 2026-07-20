//! query execution over an IndexReader.
//!
//! Perf design: scoring is SEPARATE from field hydration. The `*_scores`/`*_terms`
//! functions only compute `(doc_id, score)` or the set of candidate terms, without
//! touching `stored_fields`. `finalize` filters by min_score, sorts (score desc, id asc),
//! TRUNCATES to `limit`, and only then hydrates `stored_fields` for the survivors ONLY.
//! This way a query matching tens of thousands of docs does not decode tens of thousands
//! of full documents (which is what the previous version did, catastrophic at scale).

use crate::distance::levenshtein_bytes;
use crate::index::IndexReader;
use crate::score::Similarity;
use std::collections::HashMap;

pub struct Hit {
    pub id: usize,
    pub score: f32,
    pub fields: HashMap<String, String>,
}

/// raw scores (doc_id, score) of a term in a field. No sort/filter/hydration.
/// The idf is computed ONCE (constant over the posting list), not per doc.
pub(crate) fn term_scores(
    index: &impl IndexReader,
    sim: Similarity,
    field: &str,
    term: &str,
) -> Vec<(usize, f32)> {
    let idf = sim.idf(
        index.total_docs() as f32,
        index.doc_freq(field, term) as f32,
    );
    let avg = index.avg_field_len(field);
    index
        .postings_for(field, term)
        .into_iter()
        .map(|(doc_id, tf)| {
            (
                doc_id,
                sim.score(idf, tf, index.field_len(doc_id, field), avg),
            )
        })
        .collect()
}

/// union of several terms in a field, summing scores per doc ("should" semantics).
/// idf hoisted per term.
pub(crate) fn union_scores(
    index: &impl IndexReader,
    sim: Similarity,
    field: &str,
    terms: &[&str],
) -> HashMap<usize, f32> {
    let mut scored: HashMap<usize, f32> = HashMap::new();
    let avg = index.avg_field_len(field);
    for term in terms {
        let idf = sim.idf(
            index.total_docs() as f32,
            index.doc_freq(field, term) as f32,
        );
        for (doc_id, tf) in index.postings_for(field, term) {
            *scored.entry(doc_id).or_insert(0.0) +=
                sim.score(idf, tf, index.field_len(doc_id, field), avg);
        }
    }
    scored
}

/// terms of `field` matching the wildcard pattern (without scoring). Replicates
/// Zend_Search_Lucene Wildcard::rewrite: literal prefix before the first `*`/`?`;
/// if (in bytes) it is shorter than `min_prefix_len` → empty. The pattern compiles to a
/// regex (`?`->`.`, `*`->`.*`, anchored) and the prefix bucket is filtered.
pub(crate) fn wildcard_terms(
    index: &impl IndexReader,
    field: &str,
    pattern: &str,
    min_prefix_len: usize,
) -> Vec<String> {
    let first_wild = pattern.find(['*', '?']);
    let prefix = match first_wild {
        Some(i) => &pattern[..i],
        None => pattern,
    };
    if prefix.len() < min_prefix_len {
        return Vec::new();
    }
    // preg_quote + wildcard replacement (equivalent to ZSL)
    let escaped = regex::escape(pattern);
    let regex_str = format!("^{}$", escaped.replace("\\*", ".*").replace("\\?", "."));
    let Ok(re) = regex::Regex::new(&regex_str) else {
        return Vec::new();
    };
    index
        .terms_with_prefix(field, prefix)
        .into_iter()
        .filter(|t| re.is_match(t))
        .collect()
}

/// terms of `field` matching fuzzy (without scoring). Faithful port of
/// Zend_Search_Lucene Fuzzy::rewrite (non-empty prefix branch): exact prefix of
/// `prefix_length` chars, classic byte-based Levenshtein over the rest, maxDistance
/// varying per candidate, and match iff `similarity > min_similarity` (strict).
pub(crate) fn fuzzy_terms(
    index: &impl IndexReader,
    field: &str,
    term: &str,
    min_similarity: f32,
    prefix_length: usize,
) -> Vec<String> {
    let min_sim = f64::from(min_similarity);
    // exact prefix = first prefix_length UTF-8 chars
    let prefix: String = term.chars().take(prefix_length).collect();
    let prefix_byte_len = prefix.len();
    let prefix_utf8_len = prefix.chars().count();
    let term_rest = &term.as_bytes()[prefix_byte_len..];
    let term_rest_len = term_rest.len();

    let mut matched: Vec<(f64, String)> = Vec::new();
    for cand in index.terms_with_prefix(field, &prefix) {
        let target = &cand.as_bytes()[prefix_byte_len..];
        let target_len = target.len();
        // maxDistance = (int)((1-minSim)*(min(termRest,target)+prefixUtf8Len))
        let max_distance =
            ((1.0 - min_sim) * ((term_rest_len.min(target_len) + prefix_utf8_len) as f64)) as i64;
        let similarity: f64 = if term_rest_len == 0 {
            if prefix_utf8_len == 0 {
                0.0
            } else {
                1.0 - (target_len as f64) / (prefix_utf8_len as f64)
            }
        } else if target_len == 0 {
            if prefix_utf8_len == 0 {
                0.0
            } else {
                1.0 - (term_rest_len as f64) / (prefix_utf8_len as f64)
            }
        } else if max_distance < (term_rest_len as i64 - target_len as i64).abs() {
            0.0
        } else {
            let d = levenshtein_bytes(term_rest, target) as f64;
            1.0 - d / ((prefix_utf8_len + term_rest_len.min(target_len)) as f64)
        };
        if similarity > min_sim {
            matched.push((similarity, cand));
        }
    }
    // ZSL Fuzzy parity: keep at most the 1024 most similar terms.
    const MAX_FUZZY_TERMS: usize = 1024;
    if matched.len() > MAX_FUZZY_TERMS {
        matched.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        matched.truncate(MAX_FUZZY_TERMS);
    }
    matched.into_iter().map(|(_, t)| t).collect()
}

/// raw scores of a phrase (exact adjacency). The terms must appear at consecutive
/// positions (p, p+1, ...) and in order, in the same field/doc.
pub(crate) fn phrase_scores(
    index: &impl IndexReader,
    sim: Similarity,
    field: &str,
    terms: &[&str],
) -> HashMap<usize, f32> {
    let mut scored: HashMap<usize, f32> = HashMap::new();
    if terms.is_empty() {
        return scored;
    }
    let avg = index.avg_field_len(field);
    // decode doc->positions of each term ONCE (avoids re-walking the posting per doc),
    // and hoist the idf per term.
    let per_term: Vec<(HashMap<usize, Vec<u32>>, f32)> = terms
        .iter()
        .map(|t| {
            let positions = index.positions_all(field, t);
            let idf = sim.idf(index.total_docs() as f32, index.doc_freq(field, t) as f32);
            (positions, idf)
        })
        .collect();

    // candidate docs = intersection of the docs of all terms (the rarest one's first)
    let mut candidates: Vec<usize> = per_term[0].0.keys().copied().collect();
    for (positions, _) in &per_term[1..] {
        candidates.retain(|d| positions.contains_key(d));
    }

    let empty: Vec<u32> = Vec::new();
    for doc in candidates {
        let first = per_term[0].0.get(&doc).unwrap_or(&empty);
        let is_match = first.iter().any(|&p| {
            (1..terms.len()).all(|i| {
                per_term[i]
                    .0
                    .get(&doc)
                    .unwrap_or(&empty)
                    .contains(&(p + i as u32))
            })
        });
        if is_match {
            let s: f32 = (0..terms.len())
                .map(|i| {
                    let tf = per_term[i].0.get(&doc).map_or(0, |v| v.len() as u32);
                    sim.score(per_term[i].1, tf, index.field_len(doc, field), avg)
                })
                .sum();
            scored.insert(doc, s);
        }
    }
    scored
}

/// filters by min_score, sorts (score desc, id asc), truncates to `limit`, and hydrates
/// `stored_fields` ONLY for the surviving hits.
pub(crate) fn finalize(
    index: &impl IndexReader,
    scored: impl IntoIterator<Item = (usize, f32)>,
    min_score: f32,
    limit: usize,
) -> Vec<Hit> {
    let mut ranked: Vec<(usize, f32)> = scored
        .into_iter()
        .filter(|(_, s)| *s >= min_score)
        .collect();
    // score desc, id asc — the single comparator used by both the partition and the sort.
    let cmp = |a: &(usize, f32), b: &(usize, f32)| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    };
    // Top-k: partition in O(M) so the best `limit` land in [0..limit], then sort only those.
    // `select_nth_unstable_by(limit, …)` needs `limit < len`; otherwise the full sort is the
    // whole result anyway. (`limit` can be usize::MAX for "unlimited", which takes this branch.)
    if limit < ranked.len() {
        ranked.select_nth_unstable_by(limit, cmp);
        ranked.truncate(limit);
    }
    ranked.sort_by(cmp);
    ranked
        .into_iter()
        .map(|(id, score)| Hit {
            id,
            score,
            fields: index.stored_fields(id),
        })
        .collect()
}

/// Term query: docs containing `term` in `field`, ordered by score desc / id asc.
pub fn term_query(
    index: &impl IndexReader,
    field: &str,
    term: &str,
    min_score: f32,
    limit: usize,
) -> Vec<Hit> {
    finalize(
        index,
        term_scores(index, Similarity::Bm25, field, term),
        min_score,
        limit,
    )
}

/// MultiTerm: union of docs matching any term of the field (scores summed).
pub fn multi_term_query(
    index: &impl IndexReader,
    field: &str,
    terms: &[&str],
    min_score: f32,
    limit: usize,
) -> Vec<Hit> {
    finalize(
        index,
        union_scores(index, Similarity::Bm25, field, terms),
        min_score,
        limit,
    )
}

/// Wildcard query: terms matching the pattern, joined as a MultiTerm.
pub fn wildcard_query(
    index: &impl IndexReader,
    field: &str,
    pattern: &str,
    min_prefix_len: usize,
    min_score: f32,
    limit: usize,
) -> Vec<Hit> {
    let terms = wildcard_terms(index, field, pattern, min_prefix_len);
    let refs: Vec<&str> = terms.iter().map(std::string::String::as_str).collect();
    finalize(
        index,
        union_scores(index, Similarity::Bm25, field, &refs),
        min_score,
        limit,
    )
}

/// Fuzzy query: terms within the similarity, joined as a MultiTerm.
pub fn fuzzy_query(
    index: &impl IndexReader,
    field: &str,
    term: &str,
    min_similarity: f32,
    prefix_length: usize,
    min_score: f32,
    limit: usize,
) -> Vec<Hit> {
    let terms = fuzzy_terms(index, field, term, min_similarity, prefix_length);
    let refs: Vec<&str> = terms.iter().map(std::string::String::as_str).collect();
    finalize(
        index,
        union_scores(index, Similarity::Bm25, field, &refs),
        min_score,
        limit,
    )
}

/// terms in `field` that are accent variants of `token` and actually exist in the
/// dictionary. Spanish's single-tilde rule keeps the candidate set linear
/// (`analysis::accent_variants`); filtering by `doc_freq > 0` keeps only real
/// terms, so the caller's `union_scores` never reads empty postings. Read-only,
/// no reindex: works over the existing ZendLucene indexes.
pub(crate) fn accent_variant_terms(
    index: &impl IndexReader,
    field: &str,
    token: &str,
) -> Vec<String> {
    crate::analysis::accent_variants(token)
        .into_iter()
        .filter(|term| index.doc_freq(field, term) > 0)
        .collect()
}

/// Phrase query with exact adjacency.
pub fn phrase_query(
    index: &impl IndexReader,
    field: &str,
    terms: &[&str],
    min_score: f32,
    limit: usize,
) -> Vec<Hit> {
    finalize(
        index,
        phrase_scores(index, Similarity::Bm25, field, terms),
        min_score,
        limit,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doc::{Document, FieldKind};
    use crate::index::MemoryIndex;

    fn build() -> MemoryIndex {
        let mut idx = MemoryIndex::new();
        for text in ["foo bar", "foo foo baz", "unrelated"] {
            let mut d = Document::new();
            d.add("body", text, FieldKind::Text);
            idx.add_document(d);
        }
        idx
    }

    #[test]
    fn returns_only_matching_docs_ranked() {
        let hits = term_query(&build(), "body", "foo", 0.0, 10);
        let ids: Vec<usize> = hits.iter().map(|h| h.id).collect();
        // doc 1 (tf=2) before doc 0 (tf=1); doc 2 does not match
        assert_eq!(ids, vec![1, 0]);
    }

    fn accent_corpus() -> MemoryIndex {
        let mut idx = MemoryIndex::new();
        for text in ["el avión despega", "reserva de avion", "gestión de flota"] {
            let mut d = Document::new();
            d.add("body", text, FieldKind::Text);
            idx.add_document(d);
        }
        idx
    }

    #[test]
    fn accent_variant_terms_keeps_only_existing_terms() {
        let idx = accent_corpus();
        let mut got = accent_variant_terms(&idx, "body", "avion");
        got.sort();
        // both the plain and the accented form exist in the corpus; ávion/avíon do not.
        assert_eq!(got, vec!["avion".to_string(), "avión".to_string()]);
    }

    #[test]
    fn accent_variant_terms_bridges_from_accented_query() {
        let idx = accent_corpus();
        // user types the accented form; the plain "avion" (doc 1) must still surface.
        let got = accent_variant_terms(&idx, "body", "avión");
        assert!(got.contains(&"avion".to_string()));
        assert!(got.contains(&"avión".to_string()));
    }

    #[test]
    fn accent_variant_terms_empty_when_nothing_matches() {
        let idx = accent_corpus();
        assert!(accent_variant_terms(&idx, "body", "zzz").is_empty());
    }

    #[test]
    fn respects_limit() {
        let hits = term_query(&build(), "body", "foo", 0.0, 1);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, 1);
    }

    #[test]
    fn min_score_filters_out_low_hits() {
        // impossible threshold => no results
        let hits = term_query(&build(), "body", "foo", 1e9, 10);
        assert!(hits.is_empty());
    }

    #[test]
    fn returns_stored_fields() {
        let hits = term_query(&build(), "body", "foo", 0.0, 10);
        assert_eq!(
            hits[0].fields.get("body").map(String::as_str),
            Some("foo foo baz")
        );
    }

    #[test]
    fn multi_term_unions_and_sums_scores() {
        // corpus: doc0 has "foo", doc1 has "foo" and "bar", doc2 "unrelated"
        let mut idx = MemoryIndex::new();
        for text in ["foo x", "foo bar", "unrelated"] {
            let mut d = Document::new();
            d.add("body", text, FieldKind::Text);
            idx.add_document(d);
        }
        let hits = multi_term_query(&idx, "body", &["foo", "bar"], 0.0, 10);
        let ids: Vec<usize> = hits.iter().map(|h| h.id).collect();
        // doc1 matches foo+bar (summed score) => first; doc0 only foo; doc2 does not appear
        assert_eq!(ids, vec![1, 0]);
    }

    fn wildcard_corpus() -> MemoryIndex {
        let mut idx = MemoryIndex::new();
        for text in [
            "testing guide",
            "tested feature",
            "text editor",
            "team meeting",
        ] {
            let mut d = Document::new();
            d.add("body", text, FieldKind::Text);
            idx.add_document(d);
        }
        idx
    }

    #[test]
    fn wildcard_prefix_star() {
        // "test*" => terms testing, tested => docs 0 and 1
        let hits = wildcard_query(&wildcard_corpus(), "body", "test*", 2, 0.0, 10);
        let mut ids: Vec<usize> = hits.iter().map(|h| h.id).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![0, 1]);
    }

    #[test]
    fn wildcard_question_mark() {
        // "te?t" => ^te.t$ => "text" (doc 2); "team" does not end in t
        let hits = wildcard_query(&wildcard_corpus(), "body", "te?t", 2, 0.0, 10);
        let ids: Vec<usize> = hits.iter().map(|h| h.id).collect();
        assert_eq!(ids, vec![2]);
    }

    #[test]
    fn wildcard_short_prefix_returns_empty() {
        // literal prefix "a" (len 1) < min_prefix 2 => empty (as today)
        let hits = wildcard_query(&wildcard_corpus(), "body", "a*b*", 2, 0.0, 10);
        assert!(hits.is_empty());
    }

    #[test]
    fn fuzzy_matches_typo_within_similarity() {
        let mut idx = MemoryIndex::new();
        let mut d = Document::new();
        d.add("body", "testing framework", FieldKind::Text);
        idx.add_document(d);

        // "testintg" (one extra letter) shares prefix "tes"; high similarity => matches doc 0
        let hits = fuzzy_query(&idx, "body", "testintg", 0.6, 3, 0.0, 10);
        assert_eq!(hits.iter().map(|h| h.id).collect::<Vec<_>>(), vec![0]);
    }

    #[test]
    fn fuzzy_rejects_below_similarity() {
        let mut idx = MemoryIndex::new();
        let mut d = Document::new();
        d.add("body", "testing", FieldKind::Text);
        idx.add_document(d);

        // "tesla": shares prefix "tes" with "testing" but similarity < 0.6 => no matches
        let hits = fuzzy_query(&idx, "body", "tesla", 0.6, 3, 0.0, 10);
        assert!(hits.is_empty());
    }

    #[test]
    fn fuzzy_exact_term_matches() {
        let mut idx = MemoryIndex::new();
        let mut d = Document::new();
        d.add("body", "testing", FieldKind::Text);
        idx.add_document(d);

        // the exact term always matches (similarity 1.0)
        let hits = fuzzy_query(&idx, "body", "testing", 0.6, 3, 0.0, 10);
        assert_eq!(hits.iter().map(|h| h.id).collect::<Vec<_>>(), vec![0]);
    }

    fn phrase_corpus() -> MemoryIndex {
        let mut idx = MemoryIndex::new();
        for text in ["quick brown fox", "brown fox jumps", "the lazy dog"] {
            let mut d = Document::new();
            d.add("body", text, FieldKind::Text);
            idx.add_document(d);
        }
        idx
    }

    #[test]
    fn phrase_matches_adjacent_in_order() {
        // "brown fox": doc0 (brown@1,fox@2) and doc1 (brown@0,fox@1)
        let hits = phrase_query(&phrase_corpus(), "body", &["brown", "fox"], 0.0, 10);
        let mut ids: Vec<usize> = hits.iter().map(|h| h.id).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![0, 1]);
    }

    #[test]
    fn phrase_rejects_wrong_order() {
        // "fox brown": no doc has fox followed by brown
        let hits = phrase_query(&phrase_corpus(), "body", &["fox", "brown"], 0.0, 10);
        assert!(hits.is_empty());
    }

    #[test]
    fn phrase_rejects_non_adjacent() {
        // "quick fox": in doc0 quick@0 and fox@2 are not adjacent
        let hits = phrase_query(&phrase_corpus(), "body", &["quick", "fox"], 0.0, 10);
        assert!(hits.is_empty());
    }

    #[test]
    fn finalize_topk_matches_full_sort() {
        // 6 docs so stored_fields hydrate; scores include ties (ids 1,2,5 all 0.9) to
        // exercise the id-asc tiebreak that select_nth must preserve.
        let mut idx = MemoryIndex::new();
        for _ in 0..6 {
            let mut d = Document::new();
            d.add("body", "x", FieldKind::Text);
            idx.add_document(d);
        }
        let scored = vec![
            (0usize, 0.5f32),
            (1, 0.9),
            (2, 0.9),
            (3, 0.1),
            (4, 0.7),
            (5, 0.9),
        ];
        // reference order: score desc, id asc
        let mut reference = scored.clone();
        reference.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        for limit in [1usize, 2, 3, 5, 6, 100] {
            let hits = finalize(&idx, scored.clone(), 0.0, limit);
            let got: Vec<usize> = hits.iter().map(|h| h.id).collect();
            let want: Vec<usize> = reference.iter().take(limit).map(|(id, _)| *id).collect();
            assert_eq!(got, want, "limit={limit}");
        }
    }
}
