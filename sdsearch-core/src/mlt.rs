//! More Like This: pick a source doc's most distinctive terms (tf*idf) and
//! search for similar docs. Re-analyzes stored fields (no term vectors).

use crate::analysis::analyze;
use crate::index::IndexReader;
use crate::score::idf;
use crate::search::{Hit, finalize, term_scores};
use std::collections::HashMap;
use std::time::Instant;

/// A numeric range filter over a stored field: a hit's stored `field` value must parse as a
/// number within `[from, to]` (inclusive). Either bound may be `None` (half-open); a doc whose
/// field is missing or non-numeric does not match.
#[derive(Debug, Clone)]
pub struct RangeFilter {
    pub field: String,
    pub from: Option<f64>,
    pub to: Option<f64>,
}

/// Parameters for a More Like This query.
///
/// `max_doc_freq` and `posting_budget` are three-state: `None` = infer a safety default from
/// the collection size (see `select_terms`); `Some(0)` = explicitly unbounded/off;
/// `Some(n)` = explicit limit. `max_query_terms == 0` = no cap; `size == 0` = unlimited.
#[derive(Debug, Clone)]
pub struct MltParams {
    pub fields: Vec<String>,
    pub min_term_freq: u32,
    pub max_query_terms: usize,
    pub min_doc_freq: usize,
    pub max_doc_freq: Option<usize>,
    pub posting_budget: Option<usize>,
    pub timeout: Option<std::time::Duration>,
    pub term_filters: Vec<(String, String)>,
    pub range_filters: Vec<RangeFilter>,
    /// Require a hit to match at least this many of the selected terms. `0`/`1` = off (the
    /// default Should union already requires ≥1); a value above the selected-term count
    /// matches nothing.
    pub min_should_match: u32,
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
    let n_docs = index.total_docs();
    let n = n_docs as f32;

    // Safety defaults inferred from the (O(1)) collection size when the caller leaves them
    // unset — so a single request can't load memory proportional to the whole collection:
    // - max_doc_freq -> N/2: a term in more than half the docs is not discriminative and is
    //   the memory bomb if selected (its posting list is ~O(N)); drop it from the candidates.
    // - posting_budget -> N: cap Σ doc_freq of the selected terms at ~one collection's worth.
    // `Some(0)` means the caller explicitly opted out (unbounded); `Some(n)` is an explicit cap.
    let max_doc_freq = p.max_doc_freq.unwrap_or(n_docs / 2);
    let posting_budget = p.posting_budget.unwrap_or(n_docs);

    let mut scored: Vec<(f32, Selected)> = Vec::new();

    for field in &p.fields {
        let Some(text) = stored.get(field) else {
            continue;
        };
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
            if max_doc_freq > 0 && df > max_doc_freq {
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

    // score desc, then a stable tiebreak (field, term) so ties don't make the surviving
    // set after max_query_terms/posting_budget depend on HashMap iteration order.
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.field.cmp(&b.1.field))
            .then_with(|| a.1.term.cmp(&b.1.term))
    });
    if p.max_query_terms > 0 {
        scored.truncate(p.max_query_terms);
    }

    let mut out: Vec<Selected> = Vec::new();
    let mut spent = 0usize;
    for (_, s) in scored {
        if posting_budget > 0
            && !out.is_empty()
            && spent.saturating_add(s.doc_freq) > posting_budget
        {
            break;
        }
        spent = spent.saturating_add(s.doc_freq);
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

    // checked_add: an absurd timeout (Instant + huge Duration overflows) yields None,
    // which we treat as "no deadline" rather than panicking.
    let deadline = p.timeout.and_then(|d| Instant::now().checked_add(d));
    // Only pay for the per-doc match-count bookkeeping (a second map of the same cardinality
    // as `score`) when minimum-should-match is actually on. 0/1 never reads it, so the common
    // path allocates nothing extra (an unused empty HashMap does not allocate).
    let track_msm = p.min_should_match > 1;
    let mut score: HashMap<usize, f32> = HashMap::new();
    let mut matched: HashMap<usize, u32> = HashMap::new();
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
            if track_msm {
                *matched.entry(id).or_insert(0) += 1;
            }
        }
    }

    score.remove(&source_doc);

    // minimum-should-match: keep only docs that matched at least this many selected terms.
    // 0/1 is a no-op (a doc is in `score` only if ≥1 term hit it). Under an early timeout the
    // counts reflect only the terms processed, so msm can empty the result set (best-effort).
    if track_msm {
        score.retain(|id, _| matched.get(id).copied().unwrap_or(0) >= p.min_should_match);
    }

    for (field, value) in &p.term_filters {
        // postings_for returns doc-ids ascending, so membership is a binary search over the
        // posting list itself — no separate O(df) HashSet just to test the (usually small)
        // candidate set against the filter.
        let postings = index.postings_for(field, value);
        score.retain(|id, _| postings.binary_search_by_key(id, |&(d, _)| d).is_ok());
    }

    // Range filters: read each surviving candidate's stored value once and test every range
    // (numeric, inclusive; a missing/non-numeric field fails the filter). The candidate set is
    // already small here, so a stored-field read per candidate is cheap — and it sidesteps
    // enumerating a high-cardinality field's terms.
    if !p.range_filters.is_empty() {
        score.retain(|id, _| {
            let stored = index.stored_fields(*id);
            p.range_filters.iter().all(|rf| {
                stored
                    .get(&rf.field)
                    .and_then(|v| v.parse::<f64>().ok())
                    .is_some_and(|x| rf.from.is_none_or(|f| x >= f) && rf.to.is_none_or(|t| x <= t))
            })
        });
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
            max_doc_freq: Some(0),
            posting_budget: Some(0),
            timeout: None,
            term_filters: Vec::new(),
            range_filters: Vec::new(),
            min_should_match: 0,
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
        for text in [
            "zebra the",
            "the cat",
            "the dog",
            "the fish",
            "the bird",
            "the frog",
        ] {
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
        assert!(
            picked.contains(&"zebra"),
            "expected 'zebra', got {picked:?}"
        );
        // "zebra" (rarer) must rank above "the" (common)
        assert_eq!(terms[0].term, "zebra");
    }

    #[test]
    fn min_doc_freq_filters_rare_terms() {
        let m = idx();
        let mut p = params(&["body"]);
        p.min_doc_freq = 2; // "zebra" has df 1 -> filtered out
        let picked: Vec<String> = select_terms(&m, 0, &p)
            .into_iter()
            .map(|t| t.term)
            .collect();
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
        p.posting_budget = Some(1); // tiny budget; still keep the single top term
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

    // Source doc 0 has a rare top term "zebra" (df 2 -> ranked first) and a common
    // lower-ranked term "common" (df 5). Doc 1 is reachable ONLY via "zebra"; doc 2 is
    // reachable ONLY via "common". With a zero deadline the loop processes the top term
    // and stops, so doc 2 (behind the second term) must be absent — while with no timeout
    // it appears. This FAILS if the deadline check is removed (the whole point of the guard).
    fn ranked_terms_idx() -> MemoryIndex {
        let mut m = MemoryIndex::new();
        for text in [
            "zebra common", // 0: source
            "zebra x",      // 1: only "zebra"
            "common y",     // 2: only "common"
            "common a",     // 3: filler to make "common" common (low idf)
            "common b",     // 4
            "common c",     // 5
        ] {
            let mut d = Document::new();
            d.add("body", text, FieldKind::Text);
            m.add_document(d);
        }
        m
    }

    #[test]
    fn zero_timeout_stops_before_lower_ranked_terms() {
        use std::time::Duration;
        let m = ranked_terms_idx();

        // No timeout: both "zebra" and "common" processed -> doc 2 (via "common") is present.
        let full: Vec<usize> = more_like_this(&m, 0, &params(&["body"]))
            .iter()
            .map(|h| h.id)
            .collect();
        assert!(full.contains(&1), "doc 1 (zebra) expected: {full:?}");
        assert!(
            full.contains(&2),
            "doc 2 (common) expected without timeout: {full:?}"
        );

        // Zero deadline: only the top term "zebra" is processed -> doc 1 present, doc 2 absent.
        let mut p = params(&["body"]);
        p.timeout = Some(Duration::from_millis(0));
        let early: Vec<usize> = more_like_this(&m, 0, &p).iter().map(|h| h.id).collect();
        assert!(early.contains(&1), "top term still yields doc 1: {early:?}");
        assert!(
            !early.contains(&2),
            "lower-ranked term must be skipped under a zero deadline: {early:?}"
        );
    }

    #[test]
    fn max_doc_freq_filters_common_terms() {
        // idx(): "zebra" df 1, "the" df 6. Cap at 1 -> "the" dropped, "zebra" kept.
        let m = idx();
        let mut p = params(&["body"]);
        p.max_doc_freq = Some(1);
        let picked: Vec<String> = select_terms(&m, 0, &p)
            .into_iter()
            .map(|t| t.term)
            .collect();
        assert!(picked.contains(&"zebra".to_string()), "got {picked:?}");
        assert!(
            !picked.contains(&"the".to_string()),
            "'the' too common: {picked:?}"
        );
    }

    #[test]
    fn field_weights_change_the_top_hit() {
        // source doc 0 has title:zebra + body:quokka; doc 1 matches only title:zebra,
        // doc 2 matches only body:quokka. Weighting a field flips which one ranks first.
        let mut m = MemoryIndex::new();
        for (title, body) in [("zebra", "quokka"), ("zebra", "xxx"), ("yyy", "quokka")] {
            let mut d = Document::new();
            d.add("title", title, FieldKind::Text);
            d.add("body", body, FieldKind::Text);
            m.add_document(d);
        }

        let mut p = params(&["title", "body"]);
        p.field_weights.insert("title".to_string(), 10.0);
        let top_title = more_like_this(&m, 0, &p)[0].id;
        assert_eq!(top_title, 1, "title-weighted -> doc 1 (title:zebra) first");

        p.field_weights.clear();
        p.field_weights.insert("body".to_string(), 10.0);
        let top_body = more_like_this(&m, 0, &p)[0].id;
        assert_eq!(top_body, 2, "body-weighted -> doc 2 (body:quokka) first");
    }

    #[test]
    fn min_score_filters_out_all_hits_when_too_high() {
        // scores are normalized so the top hit is exactly 1.0; a threshold above 1.0
        // filters everything, while 0.0 keeps matches. Confirms min_score reaches finalize.
        let m = sim_idx();
        assert!(!more_like_this(&m, 0, &params(&["body"])).is_empty());
        let mut p = params(&["body"]);
        p.min_score = 2.0;
        assert!(
            more_like_this(&m, 0, &p).is_empty(),
            "min_score above the normalized max must drop every hit"
        );
    }

    #[test]
    fn posting_budget_admits_terms_until_the_ceiling() {
        // idx(): "zebra" df 1 (top), "the" df 6. A budget of 5 admits "zebra" (spent=1) but
        // rejects "the" (1+6 > 5); a budget of 7 admits both (1+6 = 7).
        let m = idx();
        let mut p = params(&["body"]);

        p.posting_budget = Some(5);
        let five: Vec<String> = select_terms(&m, 0, &p)
            .into_iter()
            .map(|t| t.term)
            .collect();
        assert_eq!(
            five,
            vec!["zebra".to_string()],
            "budget 5 stops before 'the'"
        );

        p.posting_budget = Some(7);
        let seven: Vec<String> = select_terms(&m, 0, &p)
            .into_iter()
            .map(|t| t.term)
            .collect();
        assert!(
            seven.contains(&"zebra".to_string()) && seven.contains(&"the".to_string()),
            "budget 7 admits both: {seven:?}"
        );
    }

    #[test]
    fn empty_fields_yields_no_hits() {
        // Forgetting `fields` (or passing none) selects no terms -> [] (same as "no match").
        let m = sim_idx();
        let p = params(&[]);
        assert!(select_terms(&m, 0, &p).is_empty());
        assert!(more_like_this(&m, 0, &p).is_empty());
    }

    #[test]
    fn no_shared_terms_yields_no_hits() {
        // Source doc's terms are unique to it, so after excluding the source the score map is
        // empty (top == 0 branch) -> []. Guards the "nothing similar" path end to end.
        let mut m = MemoryIndex::new();
        for text in ["singular unique", "wholly different", "entirely separate"] {
            let mut d = Document::new();
            d.add("body", text, FieldKind::Text);
            m.add_document(d);
        }
        assert!(more_like_this(&m, 0, &params(&["body"])).is_empty());
    }

    #[test]
    fn absurd_timeout_does_not_panic() {
        use std::time::Duration;
        // A timeout so large that Instant::now() + it would overflow must not panic;
        // checked_add yields None (no deadline) and the query runs to completion.
        let m = sim_idx();
        let mut p = params(&["body"]);
        p.timeout = Some(Duration::from_secs(u64::MAX));
        let ids: Vec<usize> = more_like_this(&m, 0, &p).iter().map(|h| h.id).collect();
        assert!(
            ids.contains(&1),
            "runs fully despite absurd timeout: {ids:?}"
        );
    }

    #[test]
    fn dynamic_default_max_doc_freq_drops_over_common_terms() {
        // With max_doc_freq unset (None) the engine infers total_docs/2. In idx() (N=6) that
        // is 3, so "the" (df 6, in every doc) is dropped as too common while "zebra" (df 1)
        // survives — no explicit cap from the caller. This is the memory safety default.
        let m = idx();
        let mut p = params(&["body"]);
        p.max_doc_freq = None; // infer from index size
        let picked: Vec<String> = select_terms(&m, 0, &p)
            .into_iter()
            .map(|t| t.term)
            .collect();
        assert!(picked.contains(&"zebra".to_string()), "got {picked:?}");
        assert!(
            !picked.contains(&"the".to_string()),
            "dynamic default must drop >50%-of-docs terms: {picked:?}"
        );
    }

    // doc 0 source; docs 1..3 share "zebra" but carry different numeric `created` values.
    fn range_idx() -> MemoryIndex {
        let mut m = MemoryIndex::new();
        for (body, created) in [
            ("zebra alpha", "1000"), // 0: source
            ("zebra beta", "1500"),  // 1
            ("zebra gamma", "2500"), // 2
            ("zebra delta", "oops"), // 3: non-numeric created
        ] {
            let mut d = Document::new();
            d.add("body", body, FieldKind::Text);
            d.add("created", created, FieldKind::Keyword);
            m.add_document(d);
        }
        m
    }

    fn range(field: &str, from: Option<f64>, to: Option<f64>) -> RangeFilter {
        RangeFilter {
            field: field.to_string(),
            from,
            to,
        }
    }

    #[test]
    fn range_filter_keeps_only_docs_in_bounds() {
        let m = range_idx();
        let mut p = params(&["body"]);
        // inclusive [1200, 2000]: doc 1 (1500) in; doc 2 (2500) out; doc 3 (non-numeric) out.
        p.range_filters = vec![range("created", Some(1200.0), Some(2000.0))];
        let ids: Vec<usize> = more_like_this(&m, 0, &p).iter().map(|h| h.id).collect();
        assert!(ids.contains(&1), "doc 1 (1500) is in range: {ids:?}");
        assert!(!ids.contains(&2), "doc 2 (2500) is above range: {ids:?}");
        assert!(
            !ids.contains(&3),
            "doc 3 (non-numeric) must not match: {ids:?}"
        );
    }

    #[test]
    fn range_filter_bounds_are_inclusive_and_half_open() {
        let m = range_idx();

        // from-only (>= 2500) keeps doc 2 exactly at the bound; drops doc 1.
        let mut p = params(&["body"]);
        p.range_filters = vec![range("created", Some(2500.0), None)];
        let ids: Vec<usize> = more_like_this(&m, 0, &p).iter().map(|h| h.id).collect();
        assert!(
            ids.contains(&2),
            "inclusive lower bound keeps 2500: {ids:?}"
        );
        assert!(!ids.contains(&1), "1500 < 2500 dropped: {ids:?}");

        // to-only (<= 1500) keeps doc 1; drops doc 2.
        let mut p2 = params(&["body"]);
        p2.range_filters = vec![range("created", None, Some(1500.0))];
        let ids2: Vec<usize> = more_like_this(&m, 0, &p2).iter().map(|h| h.id).collect();
        assert!(
            ids2.contains(&1),
            "inclusive upper bound keeps 1500: {ids2:?}"
        );
        assert!(!ids2.contains(&2), "2500 > 1500 dropped: {ids2:?}");
    }

    // source "alpha beta" -> 2 selected terms. doc 1 shares both, doc 2 shares only "alpha".
    fn msm_idx() -> MemoryIndex {
        let mut m = MemoryIndex::new();
        for text in ["alpha beta", "alpha beta", "alpha zzz", "gamma delta"] {
            let mut d = Document::new();
            d.add("body", text, FieldKind::Text);
            m.add_document(d);
        }
        m
    }

    #[test]
    fn min_should_match_requires_multiple_term_hits() {
        let m = msm_idx();
        // msm = 2 keeps only docs matching both selected terms (alpha AND beta).
        let mut p = params(&["body"]);
        p.min_should_match = 2;
        let ids: Vec<usize> = more_like_this(&m, 0, &p).iter().map(|h| h.id).collect();
        assert!(ids.contains(&1), "doc 1 matches alpha+beta: {ids:?}");
        assert!(!ids.contains(&2), "doc 2 matches only alpha: {ids:?}");

        // off (default 0) keeps the single-term match too.
        let off: Vec<usize> = more_like_this(&m, 0, &params(&["body"]))
            .iter()
            .map(|h| h.id)
            .collect();
        assert!(
            off.contains(&2),
            "without msm the single-term match survives: {off:?}"
        );
    }

    #[test]
    fn min_should_match_above_selected_count_yields_empty() {
        let m = msm_idx();
        // only 2 terms are selected, so requiring 3 matches nothing.
        let mut p = params(&["body"]);
        p.min_should_match = 3;
        assert!(more_like_this(&m, 0, &p).is_empty());
    }

    #[test]
    fn min_should_match_one_is_a_noop() {
        let m = msm_idx();
        let mut p = params(&["body"]);
        p.min_should_match = 1;
        let one: Vec<usize> = more_like_this(&m, 0, &p).iter().map(|h| h.id).collect();
        let off: Vec<usize> = more_like_this(&m, 0, &params(&["body"]))
            .iter()
            .map(|h| h.id)
            .collect();
        assert_eq!(one, off, "msm=1 must behave exactly like off (0)");
    }

    #[test]
    fn min_should_match_under_timeout_can_empty_results() {
        use std::time::Duration;
        // A zero deadline processes only the top term, so every match count tops at 1;
        // msm=2 then filters everything out. Pins the (surprising) msm+timeout interaction.
        let m = msm_idx();
        let mut p = params(&["body"]);
        p.min_should_match = 2;
        p.timeout = Some(Duration::from_millis(0));
        assert!(
            more_like_this(&m, 0, &p).is_empty(),
            "under a zero deadline only 1 term is processed, so msm=2 matches nothing"
        );
    }
}
