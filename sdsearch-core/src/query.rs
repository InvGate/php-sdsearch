//! boolean composition (should/must/must-not + coord) over the IndexReader trait,
//! and build_query: a port of Zend Lucene's boolean query builder for the surface the host application uses.

use crate::index::IndexReader;
use crate::score::Similarity;
use crate::search::{
    Hit, accent_variant_terms, finalize, fuzzy_terms, phrase_scores, term_scores, union_scores,
    wildcard_terms,
};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Occur {
    Should,
    Must,
    MustNot,
}

#[derive(Debug, Clone)]
pub enum Query {
    /// exact term; field None = all indexed fields (ZSL null-field rewrite).
    Term {
        field: Option<String>,
        text: String,
    },
    /// accent-insensitive term: expands to the existing accent variants of `text`
    /// (Spanish single-tilde rule, filtered against the dictionary). field None =
    /// all indexed fields, like `Term`.
    AccentTerm {
        field: Option<String>,
        text: String,
    },
    Wildcard {
        field: Option<String>,
        pattern: String,
        min_prefix_len: usize,
    },
    Fuzzy {
        field: Option<String>,
        text: String,
        similarity: f32,
        prefix_len: usize,
    },
    Phrase {
        field: String,
        terms: Vec<String>,
    },
    Boolean {
        clauses: Vec<(Occur, Query)>,
    },
}

/// target fields of a leaf: the given one, or all indexed fields if None.
fn target_fields(index: &impl IndexReader, field: &Option<String>) -> Vec<String> {
    match field {
        Some(f) => vec![f.clone()],
        None => index.indexed_fields(),
    }
}

/// per-field score multiplier; a field absent from `weights` scores at 1.0.
fn field_weight(weights: &HashMap<String, f32>, field: &str) -> f32 {
    weights.get(field).copied().unwrap_or(1.0)
}

/// evaluates a query to a doc_id -> score map (without filtering min_score or truncating).
/// `weights` scales each field's leaf contribution (see `field_weight`); `sim` selects the
/// scoring algorithm applied at every leaf.
fn eval(
    index: &impl IndexReader,
    q: &Query,
    weights: &HashMap<String, f32>,
    sim: Similarity,
) -> HashMap<usize, f32> {
    match q {
        Query::Term { field, text } => {
            let mut acc: HashMap<usize, f32> = HashMap::new();
            for f in target_fields(index, field) {
                let w = field_weight(weights, &f);
                for (id, s) in term_scores(index, sim, &f, text) {
                    *acc.entry(id).or_insert(0.0) += s * w;
                }
            }
            acc
        }
        Query::AccentTerm { field, text } => {
            let mut acc: HashMap<usize, f32> = HashMap::new();
            for f in target_fields(index, field) {
                let w = field_weight(weights, &f);
                let terms = accent_variant_terms(index, &f, text);
                let refs: Vec<&str> = terms.iter().map(std::string::String::as_str).collect();
                for (id, s) in union_scores(index, sim, &f, &refs) {
                    *acc.entry(id).or_insert(0.0) += s * w;
                }
            }
            acc
        }
        Query::Wildcard {
            field,
            pattern,
            min_prefix_len,
        } => {
            let mut acc: HashMap<usize, f32> = HashMap::new();
            for f in target_fields(index, field) {
                let w = field_weight(weights, &f);
                let terms = wildcard_terms(index, &f, pattern, *min_prefix_len);
                let refs: Vec<&str> = terms.iter().map(std::string::String::as_str).collect();
                for (id, s) in union_scores(index, sim, &f, &refs) {
                    *acc.entry(id).or_insert(0.0) += s * w;
                }
            }
            acc
        }
        Query::Fuzzy {
            field,
            text,
            similarity,
            prefix_len,
        } => {
            let mut acc: HashMap<usize, f32> = HashMap::new();
            for f in target_fields(index, field) {
                let w = field_weight(weights, &f);
                let terms = fuzzy_terms(index, &f, text, *similarity, *prefix_len);
                let refs: Vec<&str> = terms.iter().map(std::string::String::as_str).collect();
                for (id, s) in union_scores(index, sim, &f, &refs) {
                    *acc.entry(id).or_insert(0.0) += s * w;
                }
            }
            acc
        }
        Query::Phrase { field, terms } => {
            let w = field_weight(weights, field);
            let refs: Vec<&str> = terms.iter().map(std::string::String::as_str).collect();
            phrase_scores(index, sim, field, &refs)
                .into_iter()
                .map(|(id, s)| (id, s * w))
                .collect()
        }
        Query::Boolean { clauses } => eval_boolean(index, clauses, weights, sim),
    }
}

/// Lucene-style boolean semantics: must (intersection, required), should (sum/coord),
/// must-not (exclusion); if there is no must, at least one should must match.
fn eval_boolean(
    index: &impl IndexReader,
    clauses: &[(Occur, Query)],
    weights: &HashMap<String, f32>,
    sim: Similarity,
) -> HashMap<usize, f32> {
    let musts: Vec<HashMap<usize, f32>> = clauses
        .iter()
        .filter(|(o, _)| *o == Occur::Must)
        .map(|(_, q)| eval(index, q, weights, sim))
        .collect();
    let shoulds: Vec<HashMap<usize, f32>> = clauses
        .iter()
        .filter(|(o, _)| *o == Occur::Should)
        .map(|(_, q)| eval(index, q, weights, sim))
        .collect();
    let mustnots: Vec<HashMap<usize, f32>> = clauses
        .iter()
        .filter(|(o, _)| *o == Occur::MustNot)
        .map(|(_, q)| eval(index, q, weights, sim))
        .collect();

    // accumulated score and matched (should+must) clauses per doc, for the coord.
    let mut score: HashMap<usize, f32> = HashMap::new();
    let mut matched: HashMap<usize, usize> = HashMap::new();

    if musts.is_empty() {
        // union of shoulds
        for m in &shoulds {
            for (&d, &s) in m {
                *score.entry(d).or_insert(0.0) += s;
                *matched.entry(d).or_insert(0) += 1;
            }
        }
    } else {
        // intersection of musts
        let mut candidates: std::collections::HashSet<usize> = musts[0].keys().copied().collect();
        for m in &musts[1..] {
            candidates.retain(|d| m.contains_key(d));
        }
        for &d in &candidates {
            let s: f32 = musts.iter().map(|m| m[&d]).sum();
            score.insert(d, s);
            matched.insert(d, musts.len());
        }
        // shoulds only add to docs that are already candidates
        for m in &shoulds {
            for (&d, &s) in m {
                if let Some(sc) = score.get_mut(&d) {
                    *sc += s;
                    *matched.entry(d).or_insert(0) += 1;
                }
            }
        }
    }

    // exclude must-not
    for m in &mustnots {
        for d in m.keys() {
            score.remove(d);
            matched.remove(d);
        }
    }

    // coord: multiply by (matched clauses / total should+must)
    let total = (musts.len() + shoulds.len()).max(1) as f32;
    score
        .into_iter()
        .map(|(d, s)| {
            (
                d,
                s * (matched.get(&d).copied().unwrap_or(0) as f32 / total),
            )
        })
        .collect()
}

/// runs a query: normalizes the top hit's score to 1.0, filters min_score (>=),
/// sorts score desc / id asc, truncates to limit, and hydrates `stored_fields` only for the
/// final hits (via `finalize`).
///
/// Normalizing the top hit to 1.0 gives SCALE parity with ZSL (Lucene.php:982-986, which
/// divides each score by the maximum). ZSL only does it when `topScore > 1` because its raw
/// scores already live ~[0, >1]; ours are ~0.005 (simplified tf-idf, no queryNorm), so we
/// ALWAYS normalize to land on the same [0,1] scale and make a `min_score` calibrated to ZSL
/// behave the same. It is monotonic (dividing by a constant): it does NOT change the relative
/// order — RANKING fidelity is a separate matter (score shape, not scale). It happens at the
/// boolean (top) level; the leaves in `search.rs` score raw, because normalizing per leaf
/// would distort the boolean composition.
pub fn search(index: &impl IndexReader, query: &Query, min_score: f32, limit: usize) -> Vec<Hit> {
    search_with_weights(
        index,
        query,
        &HashMap::new(),
        Similarity::Bm25,
        min_score,
        limit,
    )
}

/// like `search`, but applies per-field score multipliers (`weights`, field -> factor;
/// missing = 1.0) and a selectable scoring algorithm (`sim`). The multiplier is applied at
/// the leaves, BEFORE the top-hit→1.0 normalization, so it only reorders and keeps the
/// `min_score` scale intact.
pub fn search_with_weights(
    index: &impl IndexReader,
    query: &Query,
    weights: &HashMap<String, f32>,
    sim: Similarity,
    min_score: f32,
    limit: usize,
) -> Vec<Hit> {
    let scored = eval(index, query, weights, sim);
    let top = scored.values().copied().fold(0.0f32, f32::max);
    let normalized: Vec<(usize, f32)> = if top > 0.0 {
        scored.into_iter().map(|(id, s)| (id, s / top)).collect()
    } else {
        scored.into_iter().collect()
    };
    finalize(index, normalized, min_score, limit)
}

/// WHERE group: values over a `_key` field, with the group sign (occur).
pub struct WhereGroup {
    pub field: String,
    pub values: Vec<String>,
    pub occur: Occur,
}

/// IN group: OR values over a `_key` field (required group).
pub struct InGroup {
    pub field: String,
    pub values: Vec<String>,
}

/// parameters of a host-application search (the supported surface).
#[derive(Default)]
pub struct QueryParams {
    pub text: String,
    pub where_groups: Vec<WhereGroup>,
    pub in_groups: Vec<InGroup>,
    pub fuzzy_similarity: f32,
    pub fuzzy_prefix_len: usize,
    pub wildcard_min_prefix: usize,
    /// when true, the per-token text clauses become accent-insensitive (Spanish):
    /// `avion` also matches `avión` and vice-versa. Off = current ZSL behavior.
    pub accent_insensitive: bool,
    /// optional per-field score multipliers (field -> weight); missing field = 1.0.
    /// Empty map = every field weighted equally (current behavior).
    pub field_weights: HashMap<String, f32>,
    /// scoring algorithm; defaults to Bm25.
    pub similarity: Similarity,
}

#[derive(Debug, PartialEq, Eq)]
pub enum QueryError {
    /// no text, no where, no in: nothing to search.
    Empty,
    /// a where/in group has an empty field name.
    EmptyField,
}

impl std::fmt::Display for QueryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QueryError::Empty => write!(f, "empty query"),
            QueryError::EmptyField => write!(f, "empty field name in where/in group"),
        }
    }
}
impl std::error::Error for QueryError {}

/// text subtree (port of the host's fuzzy-text subquery builder): per-word fuzzy + phrase
/// fuzzy + prefix wildcard (all over the lowercased and escaped text), plus the
/// QueryParser::parse piece (one all-fields term per ANALYZER token over the RAW text). All Should.
fn text_subquery(p: &QueryParams) -> Query {
    let lc = p.text.to_lowercase();
    // escaping mirrors the host's query builder: str_replace ':', ',', '-' -> '\:', '\,', '\-'.
    let esc = lc
        .replace(':', "\\:")
        .replace(',', "\\,")
        .replace('-', "\\-");
    let esc_words: Vec<&str> = esc.split_whitespace().collect();
    let mut clauses: Vec<(Occur, Query)> = Vec::new();

    if esc_words.len() > 1 {
        for w in &esc_words {
            clauses.push((
                Occur::Should,
                Query::Fuzzy {
                    field: None,
                    text: (*w).to_string(),
                    similarity: p.fuzzy_similarity,
                    prefix_len: p.fuzzy_prefix_len,
                },
            ));
        }
    }
    clauses.push((
        Occur::Should,
        Query::Fuzzy {
            field: None,
            text: esc.clone(),
            similarity: p.fuzzy_similarity,
            prefix_len: p.fuzzy_prefix_len,
        },
    ));
    clauses.push((
        Occur::Should,
        Query::Wildcard {
            field: None,
            pattern: format!("{esc}*"),
            min_prefix_len: p.wildcard_min_prefix,
        },
    ));
    // QueryParser::parse(RAW text): the analyzer tokenizes the original text ->
    // one all-fields term (default-OR) per token. With accent_insensitive on, the
    // per-token clause becomes an AccentTerm so `avion` also reaches `avión`.
    for tok in crate::analysis::analyze(&p.text) {
        let leaf = if p.accent_insensitive {
            Query::AccentTerm {
                field: None,
                text: tok,
            }
        } else {
            Query::Term {
                field: None,
                text: tok,
            }
        };
        clauses.push((Occur::Should, leaf));
    }
    Query::Boolean { clauses }
}

/// key-field name (for IN): appends "_key" only if the field does not already contain it.
fn key_field_in(field: &str) -> String {
    if field.contains("_key") {
        field.to_string()
    } else {
        format!("{field}_key")
    }
}

/// builds the boolean Query equivalent to Zend Lucene's boolean query builder for the supported surface.
pub fn build_query(p: &QueryParams) -> Result<Query, QueryError> {
    let has_text = !p.text.trim().is_empty();
    if !has_text && p.where_groups.is_empty() && p.in_groups.is_empty() {
        return Err(QueryError::Empty);
    }

    let mut top: Vec<(Occur, Query)> = Vec::new();

    if has_text {
        top.push((Occur::Must, text_subquery(p)));
    }

    for wg in &p.where_groups {
        if wg.field.trim().is_empty() {
            return Err(QueryError::EmptyField);
        }
        // WHERE: appends "_key" unconditionally (the host's WHERE builder does `$field."_key"`).
        let field = format!("{}_key", wg.field);
        let clauses: Vec<(Occur, Query)> = wg
            .values
            .iter()
            .map(|v| {
                (
                    Occur::Should,
                    Query::Term {
                        field: Some(field.clone()),
                        text: v.clone(),
                    },
                )
            })
            .collect();
        top.push((wg.occur, Query::Boolean { clauses }));
    }

    // IN (parity with the IN-clause merge, where all `in` groups collapse into a single
    // MultiTerm): ZSL joins ALL `in` groups into ONE MultiTerm (OR over all (field,value)),
    // added ONCE as required. It is NOT an AND between groups. The host application emits
    // several in() calls in one query (category/visibility/responsible), so this matters: a
    // doc passes if it matches AT LEAST ONE (field,value) of any in group.
    let mut in_clauses: Vec<(Occur, Query)> = Vec::new();
    for ig in &p.in_groups {
        if ig.field.trim().is_empty() {
            return Err(QueryError::EmptyField);
        }
        // IN: conditional key-field naming (appends "_key" only when missing).
        let field = key_field_in(&ig.field);
        for v in &ig.values {
            in_clauses.push((
                Occur::Should,
                Query::Term {
                    field: Some(field.clone()),
                    text: v.clone(),
                },
            ));
        }
    }
    if !in_clauses.is_empty() {
        top.push((
            Occur::Must,
            Query::Boolean {
                clauses: in_clauses,
            },
        ));
    }

    Ok(Query::Boolean { clauses: top })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doc::{Document, FieldKind};
    use crate::index::MemoryIndex;

    fn corpus() -> MemoryIndex {
        // doc0 title="vpn guide" lang="es"; doc1 title="vpn setup" lang="en";
        // doc2 title="mysql notes" lang="es"
        let mut idx = MemoryIndex::new();
        let rows = [
            ("vpn guide", "es"),
            ("vpn setup", "en"),
            ("mysql notes", "es"),
        ];
        for (title, lang) in rows {
            let mut d = Document::new();
            d.add("title", title, FieldKind::Text);
            d.add("lang_key", lang, FieldKind::Keyword);
            idx.add_document(d);
        }
        idx
    }
    fn ids(hits: &[crate::search::Hit]) -> Vec<usize> {
        let mut v: Vec<usize> = hits.iter().map(|h| h.id).collect();
        v.sort_unstable();
        v
    }

    /// true if the tree contains any Term/Wildcard/Fuzzy over `field`.
    fn query_mentions_field(q: &Query, field: &str) -> bool {
        match q {
            Query::Term { field: Some(f), .. }
            | Query::Wildcard { field: Some(f), .. }
            | Query::Fuzzy { field: Some(f), .. } => f == field,
            Query::Boolean { clauses } => {
                clauses.iter().any(|(_, c)| query_mentions_field(c, field))
            }
            _ => false,
        }
    }

    fn accent_index() -> MemoryIndex {
        let mut idx = MemoryIndex::new();
        for title in ["el avión despega", "un avion barato", "otra cosa"] {
            let mut d = Document::new();
            d.add("title", title, FieldKind::Text);
            idx.add_document(d);
        }
        idx
    }

    fn query_has_accent_term(q: &Query) -> bool {
        match q {
            Query::AccentTerm { .. } => true,
            Query::Boolean { clauses } => clauses.iter().any(|(_, c)| query_has_accent_term(c)),
            _ => false,
        }
    }

    #[test]
    fn build_query_accent_flag_emits_accent_terms() {
        let mut p = params("avion");
        p.accent_insensitive = true;
        assert!(query_has_accent_term(&build_query(&p).unwrap()));
    }

    #[test]
    fn build_query_without_accent_flag_has_no_accent_terms() {
        assert!(!query_has_accent_term(
            &build_query(&params("avion")).unwrap()
        ));
    }

    #[test]
    fn accent_term_matches_plain_and_accented() {
        // "avion" must reach both the accented (doc 0) and plain (doc 1) documents.
        let q = Query::AccentTerm {
            field: Some("title".into()),
            text: "avion".into(),
        };
        assert_eq!(ids(&search(&accent_index(), &q, 0.0, 100)), vec![0, 1]);
    }

    #[test]
    fn plain_term_does_not_bridge_accents() {
        // guard: without AccentTerm, a plain "avion" term only hits the plain doc.
        let q = Query::Term {
            field: Some("title".into()),
            text: "avion".into(),
        };
        assert_eq!(ids(&search(&accent_index(), &q, 0.0, 100)), vec![1]);
    }

    fn two_field_index() -> MemoryIndex {
        // doc0 has the term in `title`, doc1 has it in `body`; equal field length so
        // raw scores tie and the field weight alone decides the order.
        let mut idx = MemoryIndex::new();
        let mut d0 = Document::new();
        d0.add("title", "vpn", FieldKind::Text);
        d0.add("body", "x", FieldKind::Text);
        idx.add_document(d0);
        let mut d1 = Document::new();
        d1.add("title", "x", FieldKind::Text);
        d1.add("body", "vpn", FieldKind::Text);
        idx.add_document(d1);
        idx
    }

    #[test]
    fn field_weight_decides_top_hit() {
        let idx = two_field_index();
        let q = Query::Term {
            field: None,
            text: "vpn".into(),
        };
        let weight = |field: &str, w: f32| {
            let mut m = HashMap::new();
            m.insert(field.to_string(), w);
            m
        };
        // weighting `title` heavily brings the title doc (0) to the top...
        let hits =
            search_with_weights(&idx, &q, &weight("title", 10.0), Similarity::Bm25, 0.0, 100);
        assert_eq!(hits[0].id, 0);
        // ...and weighting `body` flips it to the body doc (1).
        let hits = search_with_weights(&idx, &q, &weight("body", 10.0), Similarity::Bm25, 0.0, 100);
        assert_eq!(hits[0].id, 1);
    }

    #[test]
    fn empty_weights_match_unweighted_search() {
        let idx = two_field_index();
        let q = Query::Term {
            field: None,
            text: "vpn".into(),
        };
        let weighted = search_with_weights(&idx, &q, &HashMap::new(), Similarity::Bm25, 0.0, 100);
        let plain = search(&idx, &q, 0.0, 100);
        let w_ids: Vec<usize> = weighted.iter().map(|h| h.id).collect();
        let p_ids: Vec<usize> = plain.iter().map(|h| h.id).collect();
        assert_eq!(w_ids, p_ids);
    }

    #[test]
    fn must_requires_all_clauses() {
        // vpn AND lang=es => only doc0
        let q = Query::Boolean {
            clauses: vec![
                (
                    Occur::Must,
                    Query::Term {
                        field: Some("title".into()),
                        text: "vpn".into(),
                    },
                ),
                (
                    Occur::Must,
                    Query::Term {
                        field: Some("lang_key".into()),
                        text: "es".into(),
                    },
                ),
            ],
        };
        assert_eq!(ids(&search(&corpus(), &q, 0.0, 100)), vec![0]);
    }

    #[test]
    fn should_unions_when_no_must() {
        let q = Query::Boolean {
            clauses: vec![
                (
                    Occur::Should,
                    Query::Term {
                        field: Some("title".into()),
                        text: "vpn".into(),
                    },
                ),
                (
                    Occur::Should,
                    Query::Term {
                        field: Some("title".into()),
                        text: "mysql".into(),
                    },
                ),
            ],
        };
        assert_eq!(ids(&search(&corpus(), &q, 0.0, 100)), vec![0, 1, 2]);
    }

    #[test]
    fn mustnot_excludes() {
        // vpn AND NOT lang=en => doc0 (doc1 excluded)
        let q = Query::Boolean {
            clauses: vec![
                (
                    Occur::Must,
                    Query::Term {
                        field: Some("title".into()),
                        text: "vpn".into(),
                    },
                ),
                (
                    Occur::MustNot,
                    Query::Term {
                        field: Some("lang_key".into()),
                        text: "en".into(),
                    },
                ),
            ],
        };
        assert_eq!(ids(&search(&corpus(), &q, 0.0, 100)), vec![0]);
    }

    #[test]
    fn all_fields_term_searches_every_indexed_field() {
        // field None => searches "es" in all indexed fields; matches lang_key of doc0 and doc2
        let q = Query::Term {
            field: None,
            text: "es".into(),
        };
        assert_eq!(ids(&search(&corpus(), &q, 0.0, 100)), vec![0, 2]);
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
            field_weights: HashMap::new(),
            similarity: Similarity::Bm25,
        }
    }

    #[test]
    fn build_query_text_only_matches_word_docs() {
        // "vpn" => text subtree (fuzzy/wildcard/all-fields word) as must
        let q = build_query(&params("vpn")).unwrap();
        assert_eq!(ids(&search(&corpus(), &q, 0.0, 100)), vec![0, 1]);
    }

    #[test]
    fn build_query_text_plus_where_should_does_not_narrow() {
        // a where group with occur=Should is OPTIONAL: it boosts but does NOT filter.
        // build_query suffixes the raw field "lang" -> term over "lang_key".
        let mut p = params("vpn");
        p.where_groups = vec![WhereGroup {
            field: "lang".into(),
            values: vec!["es".into()],
            occur: Occur::Should,
        }];
        let q = build_query(&p).unwrap();
        assert_eq!(ids(&search(&corpus(), &q, 0.0, 100)), vec![0, 1]);
    }

    #[test]
    fn build_query_where_must_narrows() {
        // occur=Must does filter (intersection): vpn AND lang=es => {0}.
        let mut p = params("vpn");
        p.where_groups = vec![WhereGroup {
            field: "lang".into(),
            values: vec!["es".into()],
            occur: Occur::Must,
        }];
        let q = build_query(&p).unwrap();
        assert_eq!(ids(&search(&corpus(), &q, 0.0, 100)), vec![0]);
    }

    #[test]
    fn build_query_where_mustnot() {
        let mut p = params("vpn");
        p.where_groups = vec![WhereGroup {
            field: "lang".into(),
            values: vec!["en".into()],
            occur: Occur::MustNot,
        }];
        let q = build_query(&p).unwrap();
        assert_eq!(ids(&search(&corpus(), &q, 0.0, 100)), vec![0]);
    }

    #[test]
    fn build_query_empty_is_error() {
        assert!(matches!(build_query(&params("")), Err(QueryError::Empty)));
    }

    #[test]
    fn build_query_empty_field_is_error() {
        let mut p = params("vpn");
        p.where_groups = vec![WhereGroup {
            field: String::new(),
            values: vec!["x".into()],
            occur: Occur::Should,
        }];
        assert!(matches!(build_query(&p), Err(QueryError::EmptyField)));
    }

    #[test]
    fn build_query_where_suffixes_key_unconditionally() {
        // WHERE always appends "_key" (parity with the host's WHERE-clause builder): "status" -> "status_key".
        let mut p = params("x");
        p.where_groups = vec![WhereGroup {
            field: "status".into(),
            values: vec!["1".into()],
            occur: Occur::Must,
        }];
        let q = build_query(&p).unwrap();
        // the tree must contain a Term over "status_key"
        assert!(
            query_mentions_field(&q, "status_key"),
            "WHERE must suffix _key"
        );
    }

    #[test]
    fn build_query_in_suffixes_key_conditionally() {
        // IN uses key-field naming: "cat" -> "cat_key"; "id_key" stays "id_key" (already contains it).
        let mut p = params("x");
        p.in_groups = vec![
            InGroup {
                field: "cat".into(),
                values: vec!["1".into()],
            },
            InGroup {
                field: "id_key".into(),
                values: vec!["2".into()],
            },
        ];
        let q = build_query(&p).unwrap();
        assert!(
            query_mentions_field(&q, "cat_key"),
            "IN must suffix _key when missing"
        );
        assert!(
            query_mentions_field(&q, "id_key"),
            "IN must not duplicate _key"
        );
        assert!(
            !query_mentions_field(&q, "id_key_key"),
            "IN must not duplicate _key"
        );
    }

    /// corpus to test score normalization: doc0 with high tf and a short field scores
    /// higher than doc1 (tf1, long field). doc_freq(vpn)=2 in both.
    fn score_corpus() -> MemoryIndex {
        let mut idx = MemoryIndex::new();
        for t in ["vpn vpn vpn", "vpn a b c d e f g"] {
            let mut d = Document::new();
            d.add("title", t, FieldKind::Text);
            idx.add_document(d);
        }
        idx
    }

    #[test]
    fn search_normalizes_top_hit_to_one() {
        // scale parity with ZSL (Lucene.php:982-986): the top hit's score is brought to 1.0
        // and the rest to (0,1). It is monotonic: it does NOT change the order.
        let q = Query::Term {
            field: Some("title".into()),
            text: "vpn".into(),
        };
        let hits = search(&score_corpus(), &q, 0.0, 100);
        assert_eq!(hits.len(), 2);
        assert!(
            (hits[0].score - 1.0).abs() < 1e-6,
            "top a 1.0, got {}",
            hits[0].score
        );
        assert!(
            hits[1].score > 0.0 && hits[1].score < 1.0,
            "resto en (0,1), got {}",
            hits[1].score
        );
    }

    #[test]
    fn bm25_and_tfidf_can_rank_differently() {
        // doc0 "vpn vpn vpn" (tf=3, len=3); doc1 "vpn a b c d e f g" (tf=1, len=8).
        // TF-IDF (sqrt(tf), unsaturated) favors the high-tf short doc even more than
        // BM25 does; both must return both docs, on the normalized [0,1] scale.
        let q = Query::Term {
            field: Some("title".into()),
            text: "vpn".into(),
        };
        let bm25 = search_with_weights(
            &score_corpus(),
            &q,
            &HashMap::new(),
            Similarity::Bm25,
            0.0,
            100,
        );
        let tfidf = search_with_weights(
            &score_corpus(),
            &q,
            &HashMap::new(),
            Similarity::TfIdf,
            0.0,
            100,
        );
        assert_eq!(bm25.len(), 2);
        assert_eq!(tfidf.len(), 2);
        // top score is normalized to 1.0 under both similarities
        assert!((bm25[0].score - 1.0).abs() < 1e-6);
        assert!((tfidf[0].score - 1.0).abs() < 1e-6);
        // The second hit's normalized score is where the algorithms diverge: BM25's tf
        // saturation shrinks the high-tf short doc's dominance, so doc1 sits relatively
        // higher under BM25 than under TF-IDF. If `sim` were ignored (always Bm25), these
        // would be equal — so this assertion is what actually proves per-search selection works.
        assert!(
            bm25[1].score > tfidf[1].score,
            "bm25[1]={} tfidf[1]={}",
            bm25[1].score,
            tfidf[1].score
        );
    }

    #[test]
    fn default_query_params_use_bm25() {
        // building QueryParams with ..Default::default() yields Bm25
        let p = QueryParams {
            text: "x".into(),
            ..Default::default()
        };
        assert_eq!(p.similarity, Similarity::Bm25);
    }

    #[test]
    fn search_min_score_filters_on_normalized_scale() {
        // raw scores are small (~0.1); on the normalized [0,1] scale a min_score calibrated
        // to ZSL behaves the same. Without normalization, min_score=0.5 would empty everything.
        let q = Query::Term {
            field: Some("title".into()),
            text: "vpn".into(),
        };
        assert!(
            !search(&score_corpus(), &q, 0.5, 100).is_empty(),
            "min_score=0.5 must not empty out on the normalized scale"
        );
        // nothing exceeds 1.0 => min_score>1 empties everything (proves the cap is exactly 1.0).
        assert!(
            search(&score_corpus(), &q, 1.0001, 100).is_empty(),
            "no normalized score should exceed 1.0"
        );
    }
}
