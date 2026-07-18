//! Pseudo-relevance feedback (PRF) semantic search: a two-pass query that harvests
//! distinctive terms from the first pass's top hits and re-runs an augmented query.
//! Opt-in and self-contained — the plain search path is untouched. No index-time or
//! ambient cost; feedback-term selection reuses MLT's `select_terms` (and its
//! `max_doc_freq`/`posting_budget` memory guards), so a single PRF request cannot
//! load ~O(N) posting lists.

use crate::index::IndexReader;
use crate::mlt::{MltParams, select_terms};
use crate::query::{Occur, Query, QueryError, QueryParams, build_query, search_with_weights};
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

/// Runs a two-pass PRF search. An invalid query (empty text, empty where/in field —
/// see `QueryError`) propagates its error, mirroring `search_index`. PRF itself degrades
/// gracefully to a plain search (never errors, only slower) when it cannot contribute:
/// `top_k==0`, `num_terms==0`, no pass-1 hits, or no feedback terms survive the guards —
/// in those cases the result is identical to `query`.
///
/// In the ACTIVE two-pass path the result is a RERANK, not a strict superset: the
/// augmented query is `Boolean[Should(base), Should(Boosted(feedback))]`, so an
/// original-only match's normalized score is reduced by the boolean coord factor
/// (matched/total clauses) relative to plain search. With a nonzero `min_score` or a
/// binding `limit` this can make `search_prf` omit a hit that plain search would have
/// returned. Only at `min_score == 0.0` and an unlimited `limit` (`limit == 0`) is the
/// result guaranteed to be a superset of plain search.
pub fn search_prf(
    index: &impl IndexReader,
    params: &QueryParams,
    prf: &PrfParams,
    min_score: f32,
    limit: usize,
) -> Result<Vec<Hit>, QueryError> {
    let lim = if limit == 0 { usize::MAX } else { limit };
    let base = build_query(params)?;

    // Plain search closure (the graceful-degradation fallback and the final pass share it).
    let plain = |q: &Query, ms: f32, l: usize| {
        search_with_weights(index, q, &params.field_weights, params.similarity, ms, l)
    };

    if prf.top_k == 0 || prf.num_terms == 0 {
        return Ok(plain(&base, min_score, lim));
    }

    // Pass 1: take the top_k best hits as pseudo-relevant (min_score 0.0 so feedback
    // isn't starved; the real min_score is applied on pass 2).
    let pass1 = plain(&base, 0.0, prf.top_k);
    if pass1.is_empty() {
        return Ok(Vec::new());
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
        return Ok(plain(&base, min_score, lim));
    }

    // Augmented query: original at full weight (Should), feedback terms as a
    // down-weighted Should subtree. Feedback-only docs (vocabulary mismatch) enter
    // via the feedback subtree; docs matching both are boosted by the boolean coord.
    let feedback_clauses: Vec<(Occur, Query)> = selected
        .into_iter()
        .map(|s| {
            (
                Occur::Should,
                Query::Term {
                    field: Some(s.field),
                    text: s.term,
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
    Ok(plain(&augmented, min_score, lim))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doc::{Document, FieldKind};
    use crate::hybrid::fuse_rrf;
    use crate::index::MemoryIndex;
    use crate::query::{WhereGroup, search};
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
            synonyms: false,
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
        let prf = search_prf(&idx, &params("printer"), &PrfParams::default(), 0.0, 100)
            .expect("valid query");
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
        let prf = search_prf(&idx, &params("printer"), &off, 0.0, 100).expect("valid query");
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
        let prf = search_prf(&idx, &params("printer"), &off, 0.0, 100).expect("valid query");
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
        )
        .expect("valid query");
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
        let prf = search_prf(&idx, &params("printer"), &strict, 0.0, 100).expect("valid query");
        let plain = search(&idx, &build_query(&params("printer")).unwrap(), 0.0, 100);
        assert_eq!(ids(&prf), ids(&plain));
    }

    #[test]
    fn original_match_still_present() {
        // The doc that matched the original term must never be dropped by PRF.
        let idx = corpus();
        let prf = search_prf(&idx, &params("printer"), &PrfParams::default(), 0.0, 100)
            .expect("valid query");
        assert!(
            ids(&prf).contains(&0),
            "original match doc0 missing: {:?}",
            ids(&prf)
        );
    }

    #[test]
    fn posting_budget_bounds_prf_feedback_terms() {
        // Proves the memory guard is actually wired into PRF's own feedback-selection call
        // (not just exercised in isolation by mlt.rs's own tests): builds the exact pass-1
        // doc set `search_prf` computes for "printer" (doc0 only), then calls
        // `select_terms` twice with the same MltParams shape `search_prf` constructs,
        // varying only `posting_budget`.
        //
        // doc0's stored text is "printer paper jam toner" — 4 distinct candidate terms.
        // With an effectively-unbounded budget all 4 survive (proving the corpus can
        // normally select more than one term, so a shrink is meaningful and not vacuous).
        // `select_terms` always keeps at least the single top-scoring term regardless of
        // budget (see its "always keeping at least the top term" doc comment), so a budget
        // of `Some(1)` must leave EXACTLY one term. If the guard were bypassed (e.g. the
        // budget were ignored, or applied only at some other layer), the tight run would
        // select the same 4 terms as the loose run and this assertion would fail.
        let idx = corpus();
        let pass1 = search(&idx, &build_query(&params("printer")).unwrap(), 0.0, 5);
        let doc_ids: Vec<usize> = pass1.iter().map(|h| h.id).collect();
        assert_eq!(
            doc_ids,
            vec![0],
            "pass 1 for \"printer\" must be doc0 only: {doc_ids:?}"
        );

        let base_cfg = MltParams {
            fields: idx.indexed_fields(),
            min_term_freq: 1,
            max_query_terms: 10,
            min_doc_freq: 1,
            max_doc_freq: None,
            posting_budget: Some(9999), // effectively unbounded for this tiny corpus
            timeout: None,
            term_filters: Vec::new(),
            range_filters: Vec::new(),
            min_should_match: None,
            field_weights: HashMap::new(),
            size: 0,
            min_score: 0.0,
        };
        let loose = select_terms(&idx, &doc_ids, &base_cfg);
        let tight_cfg = MltParams {
            posting_budget: Some(1),
            ..base_cfg.clone()
        };
        let tight = select_terms(&idx, &doc_ids, &tight_cfg);

        assert!(
            loose.len() > 1,
            "need >1 terms normally selected to prove the guard actually shrinks the set: {:?}",
            loose.iter().map(|s| &s.term).collect::<Vec<_>>()
        );
        assert!(
            tight.len() < loose.len(),
            "tight posting_budget must shrink the selected-term count: loose={:?} tight={:?}",
            loose.iter().map(|s| &s.term).collect::<Vec<_>>(),
            tight.iter().map(|s| &s.term).collect::<Vec<_>>()
        );
        assert_eq!(
            tight.len(),
            1,
            "posting_budget=1 must bound feedback terms to just the top-scoring one: {:?}",
            tight.iter().map(|s| &s.term).collect::<Vec<_>>()
        );
    }

    #[test]
    fn invalid_query_propagates_err() {
        // Empty text -> QueryError::Empty (mirrors build_query itself). PRF must not
        // swallow an invalid query into an empty Ok(vec![]) result.
        let idx = corpus();
        let empty_text = search_prf(&idx, &params(""), &PrfParams::default(), 0.0, 100);
        assert!(matches!(empty_text, Err(QueryError::Empty)));

        // Empty WHERE-group field name -> QueryError::EmptyField (mirrors query.rs's
        // own build_query_empty_field_is_error test).
        let mut p = params("printer");
        p.where_groups = vec![WhereGroup {
            field: String::new(),
            values: vec!["x".into()],
            occur: Occur::Should,
        }];
        let empty_field = search_prf(&idx, &p, &PrfParams::default(), 0.0, 100);
        assert!(matches!(empty_field, Err(QueryError::EmptyField)));
    }

    #[test]
    fn feedback_weight_increases_vocabulary_mismatch_score() {
        // doc1 (vocabulary mismatch, no "printer") only enters the result through the
        // feedback subtree, which is scaled by feedback_weight. If feedback_weight were
        // not actually threaded into the augmented query (e.g. hardcoded, or applied to
        // the wrong subtree), doc1's normalized score would stay flat as it's varied.
        let idx = corpus();
        let low = PrfParams {
            feedback_weight: 0.05,
            ..PrfParams::default()
        };
        let high = PrfParams {
            feedback_weight: 5.0,
            ..PrfParams::default()
        };
        let low_hits = search_prf(&idx, &params("printer"), &low, 0.0, 100).expect("valid query");
        let high_hits = search_prf(&idx, &params("printer"), &high, 0.0, 100).expect("valid query");
        let score_of = |hits: &[Hit], id: usize| {
            hits.iter()
                .find(|h| h.id == id)
                .unwrap_or_else(|| panic!("doc{id} missing: {:?}", ids(hits)))
                .score
        };
        let low_score = score_of(&low_hits, 1);
        let high_score = score_of(&high_hits, 1);
        assert!(
            high_score > low_score,
            "doc1's score must strictly increase with feedback_weight: low(0.05)={low_score} high(5.0)={high_score}"
        );
    }

    /// Corpus for the coord-dilution test: doc0 is the sole base match for "widget" and
    /// contributes NOTHING else to its own stored text. doc1..4 each add "widget" plus one
    /// unique "extraN" term. Crucially, "widget"'s doc frequency (5) EXCEEDS the default
    /// max_doc_freq guard (n_docs/2 = 9/2 = 4), so "widget" is EXCLUDED from the harvest —
    /// only the extraN terms are selected. That exclusion is what makes the test valid:
    /// doc0's only term never enters the feedback subtree, so doc0 can match ONLY the base
    /// clause and gets coord-diluted (matched/total = 1/2). If "widget" were harvested
    /// instead, doc0 would match both clauses and would NOT be diluted.
    fn dilution_corpus() -> MemoryIndex {
        let mut m = MemoryIndex::new();
        for text in [
            "widget",        // doc0: target — contributes nothing but the base term itself
            "widget extra1", // doc1
            "widget extra2", // doc2
            "widget extra3", // doc3
            "widget extra4", // doc4 (5 pass-1 docs -> df(widget) = 5)
            "filler5",
            "filler6",
            "filler7",
            "filler8", // n_docs=9 -> max_doc_freq=4
        ] {
            let mut d = Document::new();
            d.add("body", text, FieldKind::Text);
            m.add_document(d);
        }
        m
    }

    #[test]
    fn feedback_weight_zero_is_not_top_k_zero() {
        // Pins the documented footgun: feedback_weight=0.0 is NOT one of the documented
        // degrade-to-plain conditions (top_k==0, num_terms==0, no pass-1 hits, no
        // surviving feedback terms) — the augmented Boolean is still built and evaluated,
        // it just zeroes the feedback subtree's score contribution. Since doc0 only
        // matches ONE of the pass-2 augmented query's two top-level Should clauses (base;
        // it never shares a harvested term with itself, by construction of
        // dilution_corpus), the boolean coord factor (matched/total = 1/2) dilutes its
        // score relative to a TRUE plain search (top_k=0), which has no Boolean coord at
        // all for a single-term query.
        let idx = dilution_corpus();
        let true_plain = PrfParams {
            top_k: 0,
            ..PrfParams::default()
        };
        let fw_zero = PrfParams {
            feedback_weight: 0.0,
            ..PrfParams::default()
        };
        let a = search_prf(&idx, &params("widget"), &true_plain, 0.0, 100).expect("valid query");
        let b = search_prf(&idx, &params("widget"), &fw_zero, 0.0, 100).expect("valid query");
        let doc0_a = a
            .iter()
            .find(|h| h.id == 0)
            .expect("doc0 present (a)")
            .score;
        let doc0_b = b
            .iter()
            .find(|h| h.id == 0)
            .expect("doc0 present (b)")
            .score;
        assert!(
            doc0_b < doc0_a,
            "feedback_weight=0.0 must coord-dilute doc0 relative to true plain (top_k=0): \
             top_k=0 -> {doc0_a}, feedback_weight=0.0 -> {doc0_b}"
        );
    }

    /// Corpus for the rerank-contract test: docA is a long, single "zx" (base) match
    /// padded with a repeated filler ("pad") that ALSO appears in a few other docs (so
    /// its collection doc frequency exceeds the low `max_doc_freq` cap used below and it
    /// is excluded from the harvest — it only exists to lengthen docA and depress its
    /// BM25 per-term scores via length normalization). docB shares "common1" with docA
    /// but is very short and has no "zx" at all (feedback-only, vocabulary mismatch). The
    /// many one-token "noise" docs pull the collection's average field length down,
    /// amplifying docA's length penalty and docB's length bonus.
    fn rerank_corpus() -> MemoryIndex {
        let mut m = MemoryIndex::new();
        let pad = "pad ".repeat(50);
        let mut texts: Vec<String> = vec![
            format!("zx common1 {pad}"), // docA (id0): base match, long -> modest per-term score
            "common1".to_string(), // docB (id1): feedback-only, short -> strong per-term score
            "pad".to_string(),
            "pad".to_string(),
            "pad".to_string(), // push df(pad) up; excluded via max_doc_freq below regardless
        ];
        for i in 0..30 {
            texts.push(format!("noise{i}")); // short fillers depress the collection's avg field length
        }
        for text in &texts {
            let mut d = Document::new();
            d.add("body", text, FieldKind::Text);
            m.add_document(d);
        }
        m
    }

    #[test]
    fn binding_limit_can_drop_a_plain_hit() {
        // Pins the documented rerank contract: with a binding limit, PRF may omit a hit
        // that plain search would have returned. Plain search for "zx" (limit=1) can only
        // ever return docA (the sole doc containing "zx"). Under PRF with a large enough
        // feedback_weight, docB's boosted score (via the shared "common1" harvest term)
        // overtakes docA's, so the SAME limit=1 call returns docB instead — docA, which
        // plain search guarantees, is dropped from the top-1.
        let idx = rerank_corpus();
        let plain = search(&idx, &build_query(&params("zx")).unwrap(), 0.0, 1);
        assert_eq!(
            ids(&plain),
            vec![0],
            "plain search limit=1 must return docA: {:?}",
            ids(&plain)
        );

        let prf = PrfParams {
            feedback_weight: 10.0,
            max_doc_freq: Some(2), // exclude "pad" (df well above 2); "zx"/"common1" (df<=2) survive
            ..PrfParams::default()
        };
        let reranked = search_prf(&idx, &params("zx"), &prf, 0.0, 1).expect("valid query");
        assert_eq!(
            ids(&reranked),
            vec![1],
            "PRF limit=1 must drop docA in favor of the reranked docB: {:?}",
            ids(&reranked)
        );
    }

    #[test]
    fn max_doc_freq_bounds_prf_feedback_terms() {
        // Mirrors posting_budget_bounds_prf_feedback_terms's structure and reasoning, but
        // varies max_doc_freq instead: builds the same pass-1 doc set search_prf computes
        // for "printer" (doc0 only: printer/paper/jam/toner, doc_freq 1/2/2/2), then calls
        // select_terms twice with the exact MltParams shape search_prf constructs, varying
        // only max_doc_freq. A cap of 1 must exclude every doc_freq=2 candidate
        // (paper/jam/toner), leaving only "printer" (doc_freq=1) — proving the guard is
        // wired into PRF's own selection call, not just exercised by mlt.rs in isolation.
        let idx = corpus();
        let pass1 = search(&idx, &build_query(&params("printer")).unwrap(), 0.0, 5);
        let doc_ids: Vec<usize> = pass1.iter().map(|h| h.id).collect();
        assert_eq!(
            doc_ids,
            vec![0],
            "pass 1 for \"printer\" must be doc0 only: {doc_ids:?}"
        );

        let base_cfg = MltParams {
            fields: idx.indexed_fields(),
            min_term_freq: 1,
            max_query_terms: 10,
            min_doc_freq: 1,
            max_doc_freq: None,
            posting_budget: Some(9999), // unbounded for this tiny corpus: isolate max_doc_freq
            timeout: None,
            term_filters: Vec::new(),
            range_filters: Vec::new(),
            min_should_match: None,
            field_weights: HashMap::new(),
            size: 0,
            min_score: 0.0,
        };
        let loose = select_terms(&idx, &doc_ids, &base_cfg);
        let tight_cfg = MltParams {
            max_doc_freq: Some(1),
            ..base_cfg.clone()
        };
        let tight = select_terms(&idx, &doc_ids, &tight_cfg);

        assert!(
            loose.len() > 1,
            "need >1 terms normally selected to prove the guard actually shrinks the set: {:?}",
            loose.iter().map(|s| &s.term).collect::<Vec<_>>()
        );
        assert!(
            tight.len() < loose.len(),
            "max_doc_freq=1 must shrink the selected-term count: loose={:?} tight={:?}",
            loose.iter().map(|s| &s.term).collect::<Vec<_>>(),
            tight.iter().map(|s| &s.term).collect::<Vec<_>>()
        );
        assert_eq!(
            tight.iter().map(|s| s.term.as_str()).collect::<Vec<_>>(),
            vec!["printer"],
            "max_doc_freq=1 must leave only the doc_freq=1 candidate: {:?}",
            tight.iter().map(|s| &s.term).collect::<Vec<_>>()
        );
    }

    /// Two-field corpus: doc0 (base match) has "printer" in `title` and a distinctive
    /// bridge term ("jamword") ONLY in `body`. doc1 shares "jamword" in `body` but has no
    /// "printer" anywhere (vocabulary mismatch, reachable only via that bridge term).
    fn fields_corpus() -> MemoryIndex {
        let mut m = MemoryIndex::new();
        for (title, body) in [
            ("printer", "jamword"),    // doc0: base match; bridge term lives only in body
            ("unrelated1", "jamword"), // doc1: mismatch, reachable only via body's "jamword"
            ("unrelated2", "otherbody1"),
            ("unrelated3", "otherbody2"),
        ] {
            let mut d = Document::new();
            d.add("title", title, FieldKind::Text);
            d.add("body", body, FieldKind::Text);
            m.add_document(d);
        }
        m
    }

    #[test]
    fn fields_restricts_harvest_source() {
        let idx = fields_corpus();
        // fields: [] (default) harvests from every indexed field, including body, and
        // reaches doc1 via "jamword".
        let all_fields = search_prf(&idx, &params("printer"), &PrfParams::default(), 0.0, 100)
            .expect("valid query");
        assert!(
            ids(&all_fields).contains(&1),
            "fields=[] must harvest from body and retrieve doc1: {:?}",
            ids(&all_fields)
        );

        // fields: ["title"] restricts the harvest to title only ("printer" alone, already
        // in the base query) and must NOT reach doc1's body-only bridge term.
        let title_only = PrfParams {
            fields: vec!["title".to_string()],
            ..PrfParams::default()
        };
        let title_restricted =
            search_prf(&idx, &params("printer"), &title_only, 0.0, 100).expect("valid query");
        assert!(
            !ids(&title_restricted).contains(&1),
            "fields=[\"title\"] must NOT reach doc1 (its bridge term lives only in body): {:?}",
            ids(&title_restricted)
        );
    }

    /// doc0 (base match) has 4 distinctive candidate terms (termA..D), each shared with
    /// exactly one otherwise-unrelated vocabulary-mismatch doc (doc1..4). doc_freq(termA..D)
    /// = 2 each, at (not over) the default max_doc_freq (n_docs/2 = 2), so none are
    /// excluded by that guard — isolating num_terms as the only variable in play.
    fn num_terms_corpus() -> MemoryIndex {
        let mut m = MemoryIndex::new();
        for text in [
            "trigger termA termB termC termD", // doc0: base match, many candidate terms
            "termA",                           // doc1: mismatch bridged by termA
            "termB",                           // doc2: mismatch bridged by termB
            "termC",                           // doc3: mismatch bridged by termC
            "termD",                           // doc4: mismatch bridged by termD
        ] {
            let mut d = Document::new();
            d.add("body", text, FieldKind::Text);
            m.add_document(d);
        }
        m
    }

    #[test]
    fn num_terms_caps_feedback_terms() {
        // "trigger" (doc_freq=1) outranks termA..D (doc_freq=2, lower idf) in the
        // tf*idf selection order, so num_terms=1 keeps ONLY "trigger" — which bridges to
        // no other doc (doc1..4 don't contain it), so PRF retrieves just doc0. A cap of
        // 10 (default) comfortably fits all 5 candidates, so all 4 mismatch docs surface.
        // posting_budget is pinned high on both sides to isolate num_terms specifically
        // (this tiny corpus's inferred default would otherwise also cap the loose run).
        let idx = num_terms_corpus();
        let tight = PrfParams {
            num_terms: 1,
            posting_budget: Some(9999),
            ..PrfParams::default()
        };
        let loose = PrfParams {
            num_terms: 10,
            posting_budget: Some(9999),
            ..PrfParams::default()
        };
        let tight_hits =
            search_prf(&idx, &params("trigger"), &tight, 0.0, 100).expect("valid query");
        let loose_hits =
            search_prf(&idx, &params("trigger"), &loose, 0.0, 100).expect("valid query");
        assert_eq!(
            ids(&tight_hits),
            vec![0],
            "num_terms=1 must retrieve only doc0 (its one selected term bridges nothing): {:?}",
            ids(&tight_hits)
        );
        let mismatch_docs: Vec<usize> = loose_hits
            .iter()
            .map(|h| h.id)
            .filter(|id| *id != 0)
            .collect();
        assert_eq!(
            mismatch_docs.len(),
            4,
            "num_terms=10 must retrieve all 4 vocabulary-mismatch docs: {:?}",
            ids(&loose_hits)
        );
    }

    /// docX1/docX2/docX3 all match "trigger" but rank 1st/2nd/3rd (shortest to longest,
    /// same term frequency -> BM25 favors the shortest). Each carries one unique
    /// bridge term (term1/term2/term3) shared with a corresponding vocabulary-mismatch
    /// doc (docY1/docY2/docY3).
    fn top_k_corpus() -> MemoryIndex {
        let mut m = MemoryIndex::new();
        for text in [
            "trigger term1",           // docX1 (id0): rank 1 (shortest)
            "trigger term2 xpad",      // docX2 (id1): rank 2
            "trigger term3 xpad ypad", // docX3 (id2): rank 3 (longest)
            "term1",                   // docY1 (id3): mismatch bridged by term1
            "term2",                   // docY2 (id4): mismatch bridged by term2
            "term3",                   // docY3 (id5): mismatch bridged by term3
        ] {
            let mut d = Document::new();
            d.add("body", text, FieldKind::Text);
            m.add_document(d);
        }
        m
    }

    #[test]
    fn top_k_truncates_pass1_harvest_source() {
        let idx = top_k_corpus();
        // Confirm the intended pass-1 ranking before relying on it.
        let pass1 = search(&idx, &build_query(&params("trigger")).unwrap(), 0.0, 100);
        assert_eq!(
            ids(&pass1),
            vec![0, 1, 2],
            "\"trigger\" must match exactly docX1/X2/X3, ranked shortest-first: {:?}",
            ids(&pass1)
        );

        // max_doc_freq/posting_budget pinned high: this test isolates top_k, not the
        // other guards (this tiny corpus's inferred defaults would otherwise interfere).
        let top1 = PrfParams {
            top_k: 1,
            max_doc_freq: Some(9999),
            posting_budget: Some(9999),
            ..PrfParams::default()
        };
        let top3 = PrfParams {
            top_k: 3,
            max_doc_freq: Some(9999),
            posting_budget: Some(9999),
            ..PrfParams::default()
        };
        let hits_top1 = search_prf(&idx, &params("trigger"), &top1, 0.0, 100).expect("valid query");
        let hits_top3 = search_prf(&idx, &params("trigger"), &top3, 0.0, 100).expect("valid query");

        // docY3 (id5) is bridged only by "term3", harvested only from the 3rd-ranked
        // docX3 — top_k=1 truncates pass-1 to docX1 alone, so it can never surface.
        assert!(
            !ids(&hits_top1).contains(&5),
            "top_k=1 must NOT reach docY3 (its bridge term lives only on the 3rd-ranked pass-1 doc): {:?}",
            ids(&hits_top1)
        );
        assert!(
            ids(&hits_top3).contains(&5),
            "top_k=3 must reach docY3 once its harvest source is included in pass 1: {:?}",
            ids(&hits_top3)
        );
    }

    // ------------------------------------------------------------------------------------
    // Differential test: lexical vs PRF vs hybrid (RRF), on one synthetic corpus/query.
    //
    // The three modes are NESTED, not disjoint: search_prf's augmented query keeps every
    // original lexical clause (Should(base)), so at min_score=0.0/limit=0 every lexical hit
    // is also a PRF hit (see `original_match_still_present` above), and hybrid is just
    // fuse_rrf([lexical, prf]) — so lexical-ids subseteq prf-ids subseteq hybrid-ids
    // (roughly; hybrid can only add ids prf already had). Therefore "only lexical finds X"
    // is impossible to construct here, and the honest demonstration is about RANKING and
    // RECALL AT A CUTOFF, not binary find/no-find — see the three cases in
    // `three_mode_comparison` below.
    // ------------------------------------------------------------------------------------

    /// Corpus for the three-mode comparison, query Q = "gadget":
    /// - docA (id0) "gadget jam": the exact/short match — carries Q plus one jargon word
    ///   ("jam") and nothing else. It is the ONLY doc containing "gadget", so lexical search
    ///   trivially ranks it #1 (a large, structural margin — not a close score comparison).
    /// - docC (id1) "jam replacement": term-disjoint from Q (no "gadget" at all) — invisible
    ///   to lexical search, reachable only via the "jam" term PRF harvests from docA.
    ///
    /// A handful of unrelated filler docs keep "jam"/"gadget" from being trivially universal
    /// (positive idf), mirroring `corpus()` above.
    fn mode_comparison_corpus() -> MemoryIndex {
        let mut m = MemoryIndex::new();
        for text in [
            "gadget jam",             // id0 docA: exact/short match + jargon
            "jam replacement",        // id1 docC: term-disjoint, jargon bridge
            "network cable ethernet", // filler
            "monitor display hdmi",   // filler
            "keyboard mouse usb",     // filler
            "battery charger power",  // filler
        ] {
            let mut d = Document::new();
            d.add("body", text, FieldKind::Text);
            m.add_document(d);
        }
        m
    }

    #[test]
    fn three_mode_comparison() {
        const Q: &str = "gadget";
        const LIMIT: usize = 100;
        const K: usize = 60; // RRF's canonical default (hybrid.rs's own HybridParams default)
        const DOC_A: usize = 0; // exact/short match: "gadget jam"
        const DOC_C: usize = 1; // term-disjoint, jargon bridge: "jam replacement"

        let idx = mode_comparison_corpus();
        let lexical = search(&idx, &build_query(&params(Q)).unwrap(), 0.0, LIMIT);
        let prf =
            search_prf(&idx, &params(Q), &PrfParams::default(), 0.0, LIMIT).expect("valid query");
        let hybrid = fuse_rrf(&[lexical.clone(), prf.clone()], K, LIMIT);

        // --- Case (a): lexical MISSES a term-disjoint doc that PRF + hybrid recover. ---
        // docC contains no "gadget" at all; lexical search for "gadget" cannot reach it.
        // PRF harvests "jam" from docA (the only pass-1 hit) and re-queries, surfacing docC
        // via that shared jargon term; hybrid inherits it from the PRF ranking.
        assert!(
            !ids(&lexical).contains(&DOC_C),
            "lexical must miss the term-disjoint docC: {:?}",
            ids(&lexical)
        );
        assert!(
            ids(&prf).contains(&DOC_C),
            "PRF must recover docC via the harvested jargon term: {:?}",
            ids(&prf)
        );
        assert!(
            ids(&hybrid).contains(&DOC_C),
            "hybrid must inherit docC from the PRF ranking: {:?}",
            ids(&hybrid)
        );

        // --- Case (c): lexical ranks the exact/short-match doc #1. ---
        // docA is the ONLY doc containing "gadget" at all, so lexical trivially ranks it
        // first — a large, structural margin (not a close score comparison).
        assert_eq!(
            lexical[0].id,
            DOC_A,
            "lexical must rank the exact/short match doc0 first: {:?}",
            ids(&lexical)
        );

        // --- Case (b): hybrid combines lexical's PRECISION with PRF's RECALL — a
        // combination neither pure mode delivers alone. ---
        // This is the honest "hybrid wins": not a thin rank-margin between two near-tied
        // docs (fragile under routine BM25/PRF scoring changes), but a structural
        // presence/first-position combination with a large margin on both sides.
        //   - hybrid keeps lexical's precision: the exact-match doc (docA) is still #1.
        //   - hybrid ALSO gains PRF's recall: the term-disjoint doc (docC) is present.
        //   - lexical alone has the precision but NOT the recall (docC is asserted absent
        //     above — case (a)).
        // Neither pure mode gives both signals at once; hybrid does, by construction (it
        // is the RRF fusion of exactly these two rankings).
        assert_eq!(
            hybrid[0].id,
            DOC_A,
            "hybrid must keep lexical's precision: the exact match stays on top: {:?}",
            ids(&hybrid)
        );
        assert!(
            ids(&hybrid).contains(&DOC_C),
            "hybrid must also gain PRF's recall: the term-disjoint doc is present: {:?}",
            ids(&hybrid)
        );
    }
}
