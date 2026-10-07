//! boolean composition (should/must/must-not + coord) over the IndexReader trait,
//! and build_query: a port of Zend Lucene's boolean query builder for the surface the host application uses.

use crate::index::IndexReader;
use crate::score::Similarity;
use crate::search::{
    Hit, SearchOutcome, SortSpec, accent_variant_terms, accent_wildcard_terms, finalize_paged,
    finalize_sorted, fuzzy_terms, phrase_scores, term_scores, union_scores, wildcard_terms,
};
use std::collections::HashMap;
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Occur {
    Should,
    Must,
    MustNot,
}

#[derive(Debug, Clone, PartialEq)]
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
    /// `accent_insensitive` expands `pattern` to its accent variants, like `AccentTerm`.
    Wildcard {
        field: Option<String>,
        pattern: String,
        min_prefix_len: usize,
        accent_insensitive: bool,
    },
    Fuzzy {
        field: Option<String>,
        text: String,
        similarity: f32,
        prefix_len: usize,
    },
    /// phrase with exact adjacency. `field` None = the phrase must occur in at least one
    /// indexed field, scored as an OR across fields — the host adapter's multi-field
    /// `match_phrase` (`should` + `minimum_should_match: 1`). A phrase NEVER spans two
    /// fields: adjacency is checked per field.
    /// `accent_insensitive` folds Spanish accents per word, like `AccentTerm`.
    Phrase {
        field: Option<String>,
        terms: Vec<String>,
        accent_insensitive: bool,
    },
    Boolean {
        clauses: Vec<(Occur, Query)>,
    },
    /// Score multiplier: evaluates `inner` and scales every score by `boost`.
    /// Used to down-weight PRF feedback terms relative to the original query.
    Boosted {
        boost: f32,
        inner: Box<Query>,
    },
    /// every live doc, constant score. The base `build_query` gives a query with nothing
    /// positive to match, so `range`/`match_all` narrow it and a `mustnot` subtracts from it.
    MatchAll,
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
    restrict: Option<&HashSet<usize>>,
) -> HashMap<usize, f32> {
    match q {
        Query::Term { field, text } => {
            let mut acc: HashMap<usize, f32> = HashMap::new();
            for f in target_fields(index, field) {
                let w = field_weight(weights, &f);
                for (id, s) in term_scores(index, sim, &f, text, restrict) {
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
                for (id, s) in union_scores(index, sim, &f, &refs, restrict) {
                    *acc.entry(id).or_insert(0.0) += s * w;
                }
            }
            acc
        }
        Query::Wildcard {
            field,
            pattern,
            min_prefix_len,
            accent_insensitive,
        } => {
            let mut acc: HashMap<usize, f32> = HashMap::new();
            for f in target_fields(index, field) {
                let w = field_weight(weights, &f);
                let terms = if *accent_insensitive {
                    accent_wildcard_terms(index, &f, pattern, *min_prefix_len)
                } else {
                    wildcard_terms(index, &f, pattern, *min_prefix_len)
                };
                let refs: Vec<&str> = terms.iter().map(std::string::String::as_str).collect();
                for (id, s) in union_scores(index, sim, &f, &refs, restrict) {
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
                for (id, s) in union_scores(index, sim, &f, &refs, restrict) {
                    *acc.entry(id).or_insert(0.0) += s * w;
                }
            }
            acc
        }
        Query::Phrase {
            field,
            terms,
            accent_insensitive,
        } => {
            let refs: Vec<&str> = terms.iter().map(std::string::String::as_str).collect();
            let mut acc: HashMap<usize, f32> = HashMap::new();
            for f in target_fields(index, field) {
                let w = field_weight(weights, &f);
                for (id, s) in phrase_scores(index, sim, &f, &refs, *accent_insensitive, restrict) {
                    *acc.entry(id).or_insert(0.0) += s * w;
                }
            }
            acc
        }
        Query::Boolean { clauses } => eval_boolean(index, clauses, weights, sim, restrict),
        Query::Boosted { boost, inner } => {
            let mut acc = eval(index, inner, weights, sim, restrict);
            for s in acc.values_mut() {
                *s *= *boost;
            }
            acc
        }
        // `restrict` is caller input on the pub paged API, so it is narrowed like postings
        // would be: only ids that exist and are not deleted.
        Query::MatchAll => {
            let live = |&id: &usize| id < index.total_docs() && !index.is_deleted(id);
            match restrict {
                Some(allowed) => allowed
                    .iter()
                    .copied()
                    .filter(live)
                    .map(|id| (id, 1.0))
                    .collect(),
                None => (0..index.total_docs())
                    .filter(live)
                    .map(|id| (id, 1.0))
                    .collect(),
            }
        }
    }
}

/// Lucene-style boolean semantics: must (intersection, required), should (sum/coord),
/// must-not (exclusion); if there is no must, at least one should must match.
fn eval_boolean(
    index: &impl IndexReader,
    clauses: &[(Occur, Query)],
    weights: &HashMap<String, f32>,
    sim: Similarity,
    restrict: Option<&HashSet<usize>>,
) -> HashMap<usize, f32> {
    let must_qs: Vec<&Query> = clauses
        .iter()
        .filter(|(o, _)| *o == Occur::Must)
        .map(|(_, q)| q)
        .collect();
    let should_qs: Vec<&Query> = clauses
        .iter()
        .filter(|(o, _)| *o == Occur::Should)
        .map(|(_, q)| q)
        .collect();
    let mustnot_qs: Vec<&Query> = clauses
        .iter()
        .filter(|(o, _)| *o == Occur::MustNot)
        .map(|(_, q)| q)
        .collect();

    let mut score: HashMap<usize, f32> = HashMap::new();
    let mut matched: HashMap<usize, usize> = HashMap::new();

    if must_qs.is_empty() {
        // union of shoulds, each already bounded by the inherited restriction.
        for q in &should_qs {
            for (d, s) in eval(index, q, weights, sim, restrict) {
                *score.entry(d).or_insert(0.0) += s;
                *matched.entry(d).or_insert(0) += 1;
            }
        }
    } else {
        // Build a shrinking candidate set, but keep summation in DECLARED clause order so the
        // f32 accumulation is bit-identical to the pre-optimization engine (IEEE-754 add is not
        // associative). Evaluation order is decoupled from summation order: evaluate the first
        // clause LAST, because build_query places the expensive text must first — scoring it
        // against the already-narrowed candidate set is where the memory/CPU win comes from.
        let n = must_qs.len();
        let mut must_maps: Vec<Option<HashMap<usize, f32>>> = vec![None; n];
        let mut cand: Option<HashSet<usize>> = restrict.cloned();
        for i in (1..n).chain(std::iter::once(0)) {
            let m = eval(index, must_qs[i], weights, sim, cand.as_ref());
            let keys: HashSet<usize> = m.keys().copied().collect();
            cand = Some(match cand {
                Some(prev) => intersect_in_place(prev, keys),
                None => keys,
            });
            must_maps[i] = Some(m);
        }
        let final_cand = cand.unwrap_or_default();
        for &d in &final_cand {
            // sum in index (declared) clause order
            let s: f32 = must_maps
                .iter()
                .map(|m| {
                    m.as_ref()
                        .map_or(0.0, |mm| mm.get(&d).copied().unwrap_or(0.0))
                })
                .sum();
            score.insert(d, s);
            matched.insert(d, n);
        }
        // shoulds only add to docs already in the candidate set.
        for q in &should_qs {
            for (d, s) in eval(index, q, weights, sim, Some(&final_cand)) {
                if let Some(sc) = score.get_mut(&d) {
                    *sc += s;
                    *matched.entry(d).or_insert(0) += 1;
                }
            }
        }
    }

    // exclude must-not (bounded to the docs we might keep). Skip entirely when the candidate
    // set is already empty: there is nothing left to remove, and evaluating unrestricted
    // must-not clauses against the whole collection would be wasted work.
    if !score.is_empty() {
        let removal_restrict: HashSet<usize> = score.keys().copied().collect();
        for q in &mustnot_qs {
            for d in eval(index, q, weights, sim, Some(&removal_restrict)).keys() {
                score.remove(d);
                matched.remove(d);
            }
        }
    }

    // coord: multiply by (matched clauses / total should+must)
    let total = (must_qs.len() + should_qs.len()).max(1) as f32;
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
/// `limit == 0` returns an empty result (top-0), NOT "unlimited" — callers wanting unlimited pass `usize::MAX`.
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
    search_with_weights_paged(
        index, query, weights, sim, min_score, 0, limit, None, None, None,
    )
    .hits
}

/// Like `search_with_weights`, but returns a page `[offset, offset+limit)` plus the total
/// match count (`total_cap`: `None` = exact, `Some(cap)` = saturated). The top-hit→1.0
/// normalization is unchanged: it is monotonic, so paging and totals are computed on the
/// same ranking as `search_with_weights`.
#[allow(clippy::too_many_arguments)]
pub fn search_with_weights_paged(
    index: &impl IndexReader,
    query: &Query,
    weights: &HashMap<String, f32>,
    sim: Similarity,
    min_score: f32,
    offset: usize,
    limit: usize,
    total_cap: Option<usize>,
    restrict: Option<&HashSet<usize>>,
    sort: Option<&SortSpec>,
) -> SearchOutcome {
    let scored = eval(index, query, weights, sim, restrict);
    let top = scored.values().copied().fold(0.0f32, f32::max);
    let normalized: Vec<(usize, f32)> = if top > 0.0 {
        scored.into_iter().map(|(id, s)| (id, s / top)).collect()
    } else {
        scored.into_iter().collect()
    };
    // Field sort is a different finalizer, not a variation of the relevance one: `finalize_paged`
    // is reached by exactly the same call as before when `sort` is `None`.
    match sort {
        Some(spec) => finalize_sorted(index, normalized, min_score, spec, offset, limit, total_cap),
        None => finalize_paged(index, normalized, min_score, offset, limit, total_cap),
    }
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

/// Inclusive range over a keyword field's term form: `lower <= <field> <= upper` (either bound
/// `None` = unbounded). `field` is used VERBATIM (already `_key`-suffixed by the caller).
pub struct RangeFilter {
    pub field: String,
    pub lower: Option<String>,
    pub upper: Option<String>,
}

/// Collapses an allow-list that contains every live doc to `None` ("no restriction"), which is
/// what it means. Skips the per-posting `contains` in the scorer and every downstream
/// intersection.
///
/// `live` must be `num_docs()` (live docs), NOT `total_docs()` — the latter is maxDoc including
/// deletes (it feeds the idf denominator), while `postings_for` drops deleted docs, so an
/// allow-list only ever holds live ids. The `is_empty` guard keeps an empty index (`0 == 0`)
/// from turning a legitimate `Some(empty)` into `None`.
fn drop_if_universal(acc: Option<HashSet<usize>>, live: usize) -> Option<HashSet<usize>> {
    match acc {
        Some(s) if !s.is_empty() && s.len() == live => None,
        other => other,
    }
}

/// Doc allow-list for a set of range filters, ANDed together: for each filter, the union of the
/// postings of its in-range terms; intersected across filters. `None` = no filters (no
/// restriction). `Some(empty)` = a filter matched no doc (valid: the query yields nothing).
pub fn range_allow_list(
    index: &impl IndexReader,
    filters: &[RangeFilter],
) -> Option<HashSet<usize>> {
    if filters.is_empty() {
        return None;
    }
    let mut acc: Option<HashSet<usize>> = None;
    for f in filters {
        let mut docs: HashSet<usize> = HashSet::new();
        for term in index.terms_in_range(&f.field, f.lower.as_deref(), f.upper.as_deref()) {
            for (doc_id, _tf) in index.postings_for(&f.field, &term) {
                docs.insert(doc_id);
            }
        }
        acc = Some(match acc {
            Some(prev) => intersect_in_place(prev, docs),
            None => docs,
        });
    }
    drop_if_universal(acc, index.num_docs())
}

/// Field-scoped AND text filter: the analyzed words of `text` must ALL occur in `field`.
/// `field` is used verbatim (a base text field). A pure filter — contributes no score.
pub struct MatchAllFilter {
    pub field: String,
    pub text: String,
}

/// Intersection that reuses an operand's table instead of allocating a third set. Keeps the
/// smaller set and probes the larger, so the cost is O(min(|a|, |b|)) probes with no inserts
/// and no rehashing; the larger table is freed on return. Set intersection is commutative, so
/// which operand survives cannot change the contents — and `HashSet` iteration order cannot
/// reach the results, because `finalize_paged` ranks by the total order `score desc, id asc`.
fn intersect_in_place(a: HashSet<usize>, b: HashSet<usize>) -> HashSet<usize> {
    let (mut keep, probe) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    keep.retain(|d| probe.contains(d));
    keep // `probe` drops here
}

/// Intersection of two optional allow-lists where `None` means "unconstrained": `None` is the
/// identity, two `Some` sets intersect.
pub fn intersect_allow(
    a: Option<HashSet<usize>>,
    b: Option<HashSet<usize>>,
) -> Option<HashSet<usize>> {
    match (a, b) {
        (None, x) | (x, None) => x,
        (Some(x), Some(y)) => Some(intersect_in_place(x, y)),
    }
}

/// The `range`/`match_all` allow-list of `p`, to pass as `restrict`. Every retriever applies it:
/// it is what narrows the `MatchAll` base of a query with nothing positive to match.
pub fn allow_list(index: &impl IndexReader, p: &QueryParams) -> Option<HashSet<usize>> {
    intersect_allow(
        range_allow_list(index, &p.range_filters),
        match_all_allow_list(index, &p.match_all),
    )
}

/// Doc allow-list for a set of matchAll filters, ANDed together. For each filter, the analyzed
/// words must all occur in the field: intersect the words' postings, cheapest (rarest, by
/// `doc_freq`) first, short-circuiting to empty as soon as the running set is empty or a word is
/// absent. `None` = no filters (no constraint); a filter whose text has no tokens adds no
/// constraint; `Some(empty)` = a filter matched nothing.
pub fn match_all_allow_list(
    index: &impl IndexReader,
    filters: &[MatchAllFilter],
) -> Option<HashSet<usize>> {
    let mut acc: Option<HashSet<usize>> = None;
    for f in filters {
        // (doc_freq, token) so we intersect rarest-first (exact rarity, cheap dict lookup).
        let mut toks: Vec<(usize, String)> = crate::analysis::analyze(&f.text)
            .into_iter()
            .map(|t| (index.doc_freq(&f.field, &t), t))
            .collect();
        if toks.is_empty() {
            continue; // vacuous filter contributes no constraint
        }
        toks.sort_by_key(|(df, _)| *df);

        let this: HashSet<usize> = if toks[0].0 == 0 {
            HashSet::new() // rarest word absent ⇒ the AND is unsatisfiable
        } else {
            let mut set: Option<HashSet<usize>> = None;
            for (_, t) in &toks {
                let docs: HashSet<usize> = index
                    .postings_for(&f.field, t)
                    .into_iter()
                    .map(|(d, _)| d)
                    .collect();
                set = Some(match set {
                    None => docs,
                    Some(prev) => intersect_in_place(prev, docs),
                });
                if set.as_ref().is_some_and(HashSet::is_empty) {
                    break; // short-circuit: intersection can only shrink
                }
            }
            set.unwrap_or_default()
        };

        acc = Some(match acc {
            None => this,
            Some(prev) => intersect_in_place(prev, this),
        });
        if acc.as_ref().is_some_and(HashSet::is_empty) {
            break; // short-circuit across filters too
        }
    }
    drop_if_universal(acc, index.num_docs())
}

/// parameters of a host-application search (the supported surface).
///
/// Deliberately does NOT derive `Default`: several fields (`fuzzy_similarity`,
/// `wildcard_min_prefix`, ...) have unsafe zero-defaults (a 0.0 fuzzy threshold
/// matches nearly everything), so callers construct it explicitly. The scoring
/// default (BM25) lives on `Similarity::default()`, tested in `score.rs`.
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
    /// when true, each analyzed query token is additionally expanded with its
    /// bundled synonyms/translations (cross-lingual ES↔EN), each added as a
    /// down-weighted `Should` clause. Off = current behavior. See `synonyms.rs`.
    pub synonyms: bool,
    /// optional per-field score multipliers (field -> weight); missing field = 1.0.
    /// Empty map = every field weighted equally (current behavior).
    pub field_weights: HashMap<String, f32>,
    /// scoring algorithm; defaults to Bm25.
    pub similarity: Similarity,
    /// optional inclusive range filters over keyword fields (ANDed). Empty = no range
    /// restriction. Applied as the initial `restrict` allow-list, not as a scored clause.
    pub range_filters: Vec<RangeFilter>,
    /// optional field-scoped AND text filters (title/description "contains all words").
    /// Empty = none. Applied as part of the `restrict` allow-list, not as a scored clause.
    pub match_all: Vec<MatchAllFilter>,
    /// optional field sort. `None` = relevance (score) order, the default and unchanged path.
    pub sort: Option<SortSpec>,
    /// optional boolean expression tree over phrase leaves. When set it REPLACES the
    /// free-text sub-query — `text` stops participating in matching — mirroring the host
    /// adapter's `booleanTree > exactMatch > full-text` precedence. `where` / `in` /
    /// `range` / `match_all` / `sort` still apply on top, unchanged.
    pub boolean_tree: Option<BoolNode>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum QueryError {
    /// a where/in group has an empty field name.
    EmptyField,
    /// a boolean tree nests deeper than `MAX_TREE_DEPTH`.
    TreeTooDeep,
}

impl std::fmt::Display for QueryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QueryError::EmptyField => write!(f, "empty field name in where/in group"),
            QueryError::TreeTooDeep => write!(f, "boolean tree nested deeper than 32 levels"),
        }
    }
}
impl std::error::Error for QueryError {}

/// Score multiplier applied to every expanded (non-literal) synonym term. A synonym
/// is weaker evidence than the literal token the user typed, so it ranks below it.
pub const SYNONYM_BOOST: f32 = 0.6;

/// A boolean expression tree over phrase leaves — the neutral shape the host's
/// `IBooleanNode::toSearchArray()` emits (`and`/`or` with children, `not` with one child,
/// `term` with a phrase).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoolNode {
    And(Vec<BoolNode>),
    Or(Vec<BoolNode>),
    Not(Box<BoolNode>),
    Term(String),
}

/// Max nesting accepted for a boolean tree. Both `map_bool_node` and `eval` recurse, and a
/// stack overflow ABORTS the process — which `catch_unwind` at the FFI boundary cannot turn
/// into a PHP exception, so it would take the worker down. `serde_json` already caps
/// deserialization at 128 levels; this is the mapper's guard. The host's own parser caps at 20.
const MAX_TREE_DEPTH: usize = 32;

/// `NOT NOT x` == `x`. The host's parser can emit it (`foo AND NOT NOT bar` passes its
/// positive-match check). Left alone it would map to a `MustNot` over a `Boolean` that
/// evaluates to nothing, and a `MustNot` of nothing excludes nothing — so the query would
/// silently widen to `foo` instead of returning fewer hits.
fn strip_double_negation(node: &BoolNode) -> &BoolNode {
    let mut cur = node;
    while let BoolNode::Not(inner) = cur {
        match inner.as_ref() {
            BoolNode::Not(x) => cur = x,
            _ => break,
        }
    }
    cur
}

/// Maps a `BoolNode` tree to the equivalent `Query`. `accent_insensitive` is threaded to
/// every phrase leaf.
///
/// A negation with no positive sibling (`NOT foo`, a `not` under an `or`, an `and` whose
/// children are all `not`) maps to a `Boolean` that `eval_boolean` resolves to zero hits: with
/// no `Must` it takes the union-of-shoulds branch and accumulates nothing. At the ROOT,
/// `build_query` hoists those `MustNot`s next to its `MatchAll` base, so they subtract from
/// every doc; nested under an `or`, a `not` drops out (docs/API.md). The host's parser rejects
/// all of these upstream anyway (`IBooleanNode::requiresPositiveMatch`).
// ponytail: a nested pure negation still returns empty silently. Upgrade path: give `Not` a
// `Must` of `Query::MatchAll` to subtract from, after which `not` is correct in any position.
pub(crate) fn map_bool_node(
    node: &BoolNode,
    accent_insensitive: bool,
    depth: usize,
) -> Result<Query, QueryError> {
    if depth > MAX_TREE_DEPTH {
        return Err(QueryError::TreeTooDeep);
    }
    match strip_double_negation(node) {
        BoolNode::Term(phrase) => Ok(Query::Phrase {
            field: None,
            terms: crate::analysis::analyze(phrase),
            accent_insensitive,
        }),
        BoolNode::Not(inner) => Ok(Query::Boolean {
            clauses: vec![(
                Occur::MustNot,
                map_bool_node(inner, accent_insensitive, depth + 1)?,
            )],
        }),
        BoolNode::And(children) => {
            let mut clauses = Vec::with_capacity(children.len());
            for child in children {
                // a negated child becomes THIS node's MustNot instead of a nested Boolean:
                // that is Lucene's "required minus excluded", which eval_boolean implements.
                match strip_double_negation(child) {
                    BoolNode::Not(inner) => clauses.push((
                        Occur::MustNot,
                        map_bool_node(inner, accent_insensitive, depth + 1)?,
                    )),
                    other => clauses.push((
                        Occur::Must,
                        map_bool_node(other, accent_insensitive, depth + 1)?,
                    )),
                }
            }
            Ok(Query::Boolean { clauses })
        }
        BoolNode::Or(children) => {
            let mut clauses = Vec::with_capacity(children.len());
            for child in children {
                clauses.push((
                    Occur::Should,
                    map_bool_node(child, accent_insensitive, depth + 1)?,
                ));
            }
            Ok(Query::Boolean { clauses })
        }
    }
}

/// text subtree (port of the host's fuzzy-text subquery builder): per-word fuzzy + prefix
/// wildcard, plus one all-fields analyzer term per token. All Should.
///
/// We intentionally drop two vestiges of the host port: (1) the `:`/`,`/`-` escaping — the
/// host escaped these as query operators, but our analyzer keeps them INSIDE tokens
/// (`c:drive`, `back-up` are single terms), so escaping them only inserted backslashes that
/// no indexed term carries; and (2) the whole-text fuzzy/wildcard for multi-word input, which
/// could never match (no indexed term spans the whitespace). Removing them lets the fuzzy and
/// wildcard leaves work on punctuation-bearing tokens, consistently with the analyzer.
///
/// When `synonyms` is on, each analyzer token additionally emits a down-weighted
/// (`SYNONYM_BOOST`) Should clause per bundled synonym/translation (see `synonyms.rs`).
fn text_subquery(p: &QueryParams) -> Query {
    // .then(...) short-circuits: crate::synonyms::global() (and its first-call
    // OnceLock::get_or_init decode of the bundled dictionary) only runs when synonyms
    // is on. synonyms:false must cost nothing, including "nothing parsed" — see synonyms.rs.
    let dict = p.synonyms.then(crate::synonyms::global);
    text_subquery_with_dict(p, dict)
}

/// Testable core of `text_subquery`: takes the synonym dictionary explicitly (as an
/// `Option` so callers/tests can pass `None` without forcing the global to resolve) so
/// unit tests can inject a fixture instead of the bundled global.
fn text_subquery_with_dict(p: &QueryParams, dict: Option<&crate::synonyms::SynonymDict>) -> Query {
    let lc = p.text.to_lowercase();
    let mut clauses: Vec<(Occur, Query)> = Vec::new();

    // Per-word fuzzy for typo tolerance (whitespace split matches how the analyzer keeps
    // punctuation inside a token).
    for w in lc.split_whitespace() {
        clauses.push((
            Occur::Should,
            Query::Fuzzy {
                field: None,
                text: w.to_string(),
                similarity: p.fuzzy_similarity,
                prefix_len: p.fuzzy_prefix_len,
            },
        ));
    }
    // Prefix wildcard over the whole text (useful for single-word prefix search; a no-op for
    // multi-word input, where no indexed term spans the space).
    clauses.push((
        Occur::Should,
        Query::Wildcard {
            field: None,
            pattern: format!("{lc}*"),
            min_prefix_len: p.wildcard_min_prefix,
            accent_insensitive: p.accent_insensitive,
        },
    ));
    // QueryParser::parse(RAW text): the analyzer tokenizes the original text ->
    // one all-fields term (default-OR) per token. With accent_insensitive on, the
    // per-token clause becomes an AccentTerm so `avion` also reaches `avión`.
    for tok in crate::analysis::analyze(&p.text) {
        let leaf = |text: String| {
            if p.accent_insensitive {
                Query::AccentTerm { field: None, text }
            } else {
                Query::Term { field: None, text }
            }
        };
        clauses.push((Occur::Should, leaf(tok.clone())));
        if p.synonyms {
            if let Some(d) = dict {
                for syn in d.expand(&tok) {
                    clauses.push((
                        Occur::Should,
                        Query::Boosted {
                            boost: SYNONYM_BOOST,
                            inner: Box::new(leaf(syn.clone())),
                        },
                    ));
                }
            }
        }
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
    let mut top: Vec<(Occur, Query)> = Vec::new();

    // a boolean tree REPLACES the free-text subquery (host precedence: tree > exact > text).
    if let Some(tree) = &p.boolean_tree {
        match map_bool_node(tree, p.accent_insensitive, 0)? {
            // a root that only negates (`NOT x`, an `and` of nots) is a `mustnot` where in
            // disguise: hoisted to the top, it subtracts from the `MatchAll` base below.
            Query::Boolean { clauses }
                if !clauses.is_empty() && clauses.iter().all(|(o, _)| *o == Occur::MustNot) =>
            {
                top.extend(clauses);
            }
            q => top.push((Occur::Must, q)),
        }
    } else if has_text {
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
    // groups present but no values at all is "allowed in none", not "no filter": the empty
    // Boolean is still required, so it matches nothing (fails closed).
    if !p.in_groups.is_empty() {
        top.push((
            Occur::Must,
            Query::Boolean {
                clauses: in_clauses,
            },
        ));
    }

    // Nothing positive to match (no text/tree, nothing at all, or only `mustnot` groups) means
    // "every doc", narrowed by the `range`/`match_all` allow-list and minus the `mustnot`s — the
    // same answer OpenSearch gives an empty text. A `should`-only where keeps its old meaning.
    if !top
        .iter()
        .any(|(o, _)| matches!(o, Occur::Must | Occur::Should))
    {
        top.push((Occur::Must, Query::MatchAll));
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
            Query::Boosted { inner, .. } => query_has_accent_term(inner),
            _ => false,
        }
    }

    /// true only if some `Boosted` subtree's inner leaf is an `AccentTerm` — unlike
    /// `query_has_accent_term`, this ignores literal (non-boosted) clauses, so it isolates
    /// the synonym leaf's shape specifically instead of also matching the literal token's
    /// own (unwrapped) AccentTerm clause when `accent_insensitive` is on.
    fn boosted_leaf_is_accent_term(q: &Query) -> bool {
        match q {
            Query::Boosted { inner, .. } => matches!(**inner, Query::AccentTerm { .. }),
            Query::Boolean { clauses } => {
                clauses.iter().any(|(_, c)| boosted_leaf_is_accent_term(c))
            }
            _ => false,
        }
    }

    /// true if some `Boosted` subtree's inner leaf is a plain `Term` (used to confirm a
    /// Boosted synonym clause exists at all, so the accompanying `!boosted_leaf_is_accent_term`
    /// check isn't vacuously true).
    fn has_boosted_term_leaf(q: &Query) -> bool {
        match q {
            Query::Boosted { inner, .. } => matches!(**inner, Query::Term { .. }),
            Query::Boolean { clauses } => clauses.iter().any(|(_, c)| has_boosted_term_leaf(c)),
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
    fn accent_insensitive_prefix_reaches_the_same_docs_either_way() {
        // "camioneta" is reachable only through the wildcard leaf (fuzzy stops at distance 3),
        // so before the fold "camion" found it and "camión" did not.
        let mut idx = MemoryIndex::new();
        for title in ["camión rojo", "camioneta azul", "otra cosa"] {
            let mut d = Document::new();
            d.add("title", title, FieldKind::Text);
            idx.add_document(d);
        }
        let hits_for = |text: &str| {
            let mut p = params(text);
            p.accent_insensitive = true;
            ids(&search(&idx, &build_query(&p).unwrap(), 0.0, 100))
        };
        assert_eq!(hits_for("camion"), vec![0, 1]);
        assert_eq!(hits_for("camión"), vec![0, 1]);
    }

    #[test]
    fn accent_insensitive_prefix_keeps_the_min_prefix_gate() {
        // "a" is one byte (gated) but its "á" variant is two: the gate is measured on the
        // folded form, or a one-letter query would expand over every "á…" term.
        let mut idx = MemoryIndex::new();
        let mut d = Document::new();
        d.add("title", "área común", FieldKind::Text);
        idx.add_document(d);
        let mut p = params("a");
        p.accent_insensitive = true;
        p.wildcard_min_prefix = 2;
        assert!(search(&idx, &build_query(&p).unwrap(), 0.0, 100).is_empty());
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
    fn three_must_score_sums_in_declared_order() {
        // A 3-Must boolean scored via eval_boolean must sum each matching doc's must
        // contributions in DECLARED clause order, so the (non-associative) f32 result is
        // stable regardless of the internal evaluation order used for candidate narrowing.
        let idx = corpus(); // doc0 = "vpn guide", lang_key "es"
        let weights = HashMap::new();
        let sim = Similarity::Bm25;
        let m0 = Query::Term {
            field: Some("title".into()),
            text: "vpn".into(),
        };
        let m1 = Query::Term {
            field: Some("title".into()),
            text: "guide".into(),
        };
        let m2 = Query::Term {
            field: Some("lang_key".into()),
            text: "es".into(),
        };
        // per-must contributions, unrestricted
        let e0 = eval(&idx, &m0, &weights, sim, None);
        let e1 = eval(&idx, &m1, &weights, sim, None);
        let e2 = eval(&idx, &m2, &weights, sim, None);
        let q = Query::Boolean {
            clauses: vec![
                (Occur::Must, m0.clone()),
                (Occur::Must, m1.clone()),
                (Occur::Must, m2.clone()),
            ],
        };
        let got = eval(&idx, &q, &weights, sim, None);
        let d = 0usize; // doc0 matches all three musts; coord = 3/3 = 1.0
        let expected = e0[&d] + e1[&d] + e2[&d]; // declared-order f32 sum
        assert_eq!(
            got[&d], expected,
            "3-must score must equal the declared-order f32 sum"
        );
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
            synonyms: false,
            field_weights: HashMap::new(),
            similarity: Similarity::Bm25,
            range_filters: vec![],
            match_all: vec![],
            sort: None,
            boolean_tree: None,
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
    fn build_query_where_ors_within_a_field_and_ands_between_fields() {
        // Invariante del que depende traducir un in() AND-eado a where(): los valores de UN
        // where group son Should entre sí (OR), y el grupo entero entra como Must (AND entre
        // campos). Nunca `type=1 AND type=2`, que no matchearía nada. Es la misma forma que
        // arma Zend (Boolean por campo, sign null por valor -> unión; sign true al grupo) y que
        // arma ES/OpenSearch (un `terms` por campo dentro de bool.filter).
        let mut idx = MemoryIndex::new();
        for (title, ty, st) in [("vpn guide", "1", "4"), ("vpn setup", "2", "9")] {
            let mut d = Document::new();
            d.add("title", title, FieldKind::Text);
            d.add("type_key", ty, FieldKind::Keyword);
            d.add("status_key", st, FieldKind::Keyword);
            idx.add_document(d);
        }
        let group = |field: &str, values: &[&str]| WhereGroup {
            field: field.into(),
            values: values.iter().map(|v| (*v).to_string()).collect(),
            occur: Occur::Must,
        };

        // OR adentro del campo: los dos docs pasan, ninguno tiene type 1 Y 2 a la vez.
        let mut p = params("vpn");
        p.where_groups = vec![group("type", &["1", "2"])];
        assert_eq!(
            ids(&search(&idx, &build_query(&p).unwrap(), 0.0, 100)),
            vec![0, 1],
            "los valores de un where group son OR, no AND"
        );

        // AND entre campos: type in (1,2) AND status in (4) => sólo doc0.
        let mut p = params("vpn");
        p.where_groups = vec![group("type", &["1", "2"]), group("status", &["4"])];
        assert_eq!(
            ids(&search(&idx, &build_query(&p).unwrap(), 0.0, 100)),
            vec![0],
            "dos where groups AND-ean"
        );

        // y el AND es real: un status que no matchea ningún doc vacía el resultado.
        let mut p = params("vpn");
        p.where_groups = vec![group("type", &["1", "2"]), group("status", &["7"])];
        assert!(
            ids(&search(&idx, &build_query(&p).unwrap(), 0.0, 100)).is_empty(),
            "el segundo where group filtra de verdad"
        );
    }

    #[test]
    fn build_query_empty_matches_every_doc() {
        let q = build_query(&params("")).unwrap();
        assert_eq!(ids(&search(&corpus(), &q, 0.0, 100)), vec![0, 1, 2]);
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
    fn build_query_in_group_without_values_matches_nothing() {
        // The host restricts visibility with `in`: "allowed in none of these" must not read as
        // "no filter". It used to drop the group, leaking every text match — and, with no
        // text, the whole index through the `MatchAll` base.
        let mut p = params("");
        p.in_groups = vec![InGroup {
            field: "lang".into(),
            values: vec![],
        }];
        assert!(search(&corpus(), &build_query(&p).unwrap(), 0.0, 100).is_empty());
        p.text = "vpn".into();
        assert!(search(&corpus(), &build_query(&p).unwrap(), 0.0, 100).is_empty());
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

    #[test]
    fn build_query_ors_in_groups_across_distinct_fields() {
        // Dos in() sobre campos DISTINTOS se combinan con OR, no con AND: todos los grupos IN
        // colapsan en un único Boolean de Shoulds agregado una sola vez como Must (ver el
        // comentario junto a `in_clauses`). Es la semántica de Zend_Search_Lucene
        // (ZendLucene::addQueriesIn arma UN MultiTerm con occur=null) y la espejamos a propósito:
        // la búsqueda de KB filtra visibility_type y responsible con in() separados y depende de
        // ese OR. El corpus discrimina las cuatro hipótesis: OR -> [0,1]; AND -> []; sólo el
        // primer in -> [0]; sólo el último -> [1].
        let mut idx = MemoryIndex::new();
        for (title, ty, resp) in [("vpn guide", "8", "100"), ("vpn setup", "5", "101")] {
            let mut d = Document::new();
            d.add("title", title, FieldKind::Text);
            d.add("type_key", ty, FieldKind::Keyword);
            d.add("responsible_key", resp, FieldKind::Keyword);
            idx.add_document(d);
        }

        let mut p = params("vpn");
        p.in_groups = vec![
            InGroup {
                field: "type".into(),
                values: vec!["8".into()],
            },
            InGroup {
                field: "responsible".into(),
                values: vec!["101".into()],
            },
        ];
        let q = build_query(&p).unwrap();
        assert_eq!(
            ids(&search(&idx, &q, 0.0, 100)),
            vec![0, 1],
            "los grupos IN OR-ean entre campos distintos, no AND-ean"
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
    fn boosted_multiplies_leaf_scores() {
        // Boosted{boost, inner} must return inner's score map with every score * boost.
        let idx = score_corpus();
        let base = Query::Term {
            field: Some("title".into()),
            text: "vpn".into(),
        };
        let plain = eval(&idx, &base, &HashMap::new(), Similarity::Bm25, None);
        let boosted = eval(
            &idx,
            &Query::Boosted {
                boost: 3.0,
                inner: Box::new(base),
            },
            &HashMap::new(),
            Similarity::Bm25,
            None,
        );
        assert_eq!(plain.len(), boosted.len());
        for (id, s) in &plain {
            assert!(
                (boosted[id] - s * 3.0).abs() < 1e-6,
                "id={id} plain={s} boosted={}",
                boosted[id]
            );
        }
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

    #[test]
    fn build_query_wildcard_min_prefix_gates_expansion() {
        // "abcdef" is reachable ONLY through wildcard prefix expansion of "ab*":
        // the exact term "ab" is absent, and fuzzy("ab") can't reach it (empty term-rest
        // gives negative similarity), so the wildcard leaf alone decides the match.
        let mut idx = MemoryIndex::new();
        let mut d = Document::new();
        d.add("title", "abcdef", FieldKind::Text);
        idx.add_document(d);

        let mut p = params("ab");
        p.wildcard_min_prefix = 2; // prefix "ab" (len 2) >= 2 => expands
        let q = build_query(&p).unwrap();
        assert_eq!(
            ids(&search(&idx, &q, 0.0, 100)),
            vec![0],
            "min_prefix 2 must let 'ab*' expand and match 'abcdef'"
        );

        let mut p = params("ab");
        p.wildcard_min_prefix = 3; // prefix "ab" (len 2) < 3 => wildcard leaf empty
        let q = build_query(&p).unwrap();
        assert!(
            search(&idx, &q, 0.0, 100).is_empty(),
            "min_prefix 3 must gate 'ab*' off (no other leaf matches 'abcdef')"
        );
    }

    #[test]
    fn eval_restrict_equals_unrestricted_then_filtered() {
        // The restrict allow-list must yield exactly the unrestricted result filtered to the
        // allowed docs, with identical scores (BM25 stats are collection-wide).
        let idx = corpus(); // doc0,doc1 have "vpn"; doc2 does not
        let q = Query::Term {
            field: Some("title".into()),
            text: "vpn".into(),
        };
        let weights = HashMap::new();
        let full = eval(&idx, &q, &weights, Similarity::Bm25, None);
        let allow: std::collections::HashSet<usize> = [0usize].into_iter().collect();
        let restricted = eval(&idx, &q, &weights, Similarity::Bm25, Some(&allow));
        let expected: HashMap<usize, f32> = full
            .into_iter()
            .filter(|(d, _)| allow.contains(d))
            .collect();
        assert_eq!(restricted, expected);
    }

    #[test]
    fn search_with_weights_paged_field_sort() {
        // three docs sharing the text term, with variable-width numeric created_at values so
        // numeric order ("100" < "200" < "300") is what gets asserted, not byte order.
        let mut idx = MemoryIndex::new();
        for ca in ["300", "100", "200"] {
            let mut d = Document::new();
            d.add("title", "ticket", FieldKind::Text);
            d.add("created_at_key", ca, FieldKind::Keyword);
            idx.add_document(d);
        }
        let q = Query::Term {
            field: Some("title".into()),
            text: "ticket".into(),
        };
        let ids = |o: &SearchOutcome| o.hits.iter().map(|h| h.id).collect::<Vec<_>>();

        let asc = SortSpec {
            field: "created_at_key".into(),
            ascending: true,
        };
        let out = search_with_weights_paged(
            &idx,
            &q,
            &HashMap::new(),
            Similarity::Bm25,
            0.0,
            0,
            10,
            None,
            None,
            Some(&asc),
        );
        assert_eq!(ids(&out), vec![1, 2, 0], "ascending by created_at");
        assert_eq!(out.total, 3);

        let desc = SortSpec {
            field: "created_at_key".into(),
            ascending: false,
        };
        let out = search_with_weights_paged(
            &idx,
            &q,
            &HashMap::new(),
            Similarity::Bm25,
            0.0,
            0,
            10,
            None,
            None,
            Some(&desc),
        );
        assert_eq!(ids(&out), vec![0, 2, 1], "descending flips it");

        // `sort: None` must reach `finalize_paged` and produce the relevance order, unchanged.
        // All three docs score identically here, so the relevance tiebreak is id asc.
        let out = search_with_weights_paged(
            &idx,
            &q,
            &HashMap::new(),
            Similarity::Bm25,
            0.0,
            0,
            10,
            None,
            None,
            None,
        );
        assert_eq!(ids(&out), vec![0, 1, 2], "no sort => relevance path");
    }

    #[test]
    fn search_with_weights_paged_reports_total_and_pages() {
        let idx = corpus(); // doc0 "vpn guide", doc1 "vpn setup", doc2 "mysql notes"
        let q = Query::Term {
            field: Some("title".into()),
            text: "vpn".into(),
        };
        // full: two matches (docs 0 and 1), exact total
        let full = search_with_weights_paged(
            &idx,
            &q,
            &HashMap::new(),
            Similarity::Bm25,
            0.0,
            0,
            10,
            None,
            None,
            None,
        );
        assert_eq!(full.hits.len(), 2);
        assert_eq!(full.total, 2);
        assert!(!full.total_capped);

        // offset 1, limit 1 => the 2nd hit only; total still reflects all matches
        let page = search_with_weights_paged(
            &idx,
            &q,
            &HashMap::new(),
            Similarity::Bm25,
            0.0,
            1,
            1,
            None,
            None,
            None,
        );
        assert_eq!(page.hits.len(), 1);
        assert_eq!(page.hits[0].id, full.hits[1].id);
        assert_eq!(page.total, 2);

        // cap of 1 => total saturates, total_capped set, page still honored
        let capped = search_with_weights_paged(
            &idx,
            &q,
            &HashMap::new(),
            Similarity::Bm25,
            0.0,
            0,
            10,
            Some(1),
            None,
            None,
        );
        assert_eq!(capped.total, 1);
        assert!(capped.total_capped);
        assert_eq!(capped.hits.len(), 2, "cap bounds the total, not the page");
    }

    #[test]
    fn range_allow_list_narrows_scored_docs() {
        let mut idx = MemoryIndex::new();
        for ca in ["100", "200", "300"] {
            let mut d = Document::new();
            d.add("body", "ticket", FieldKind::Text);
            d.add("created_at_key", ca, FieldKind::Keyword);
            idx.add_document(d);
        }
        // [150,250] over created_at_key => only doc1 ("200")
        let filters = vec![RangeFilter {
            field: "created_at_key".into(),
            lower: Some("150".into()),
            upper: Some("250".into()),
        }];
        let allow = range_allow_list(&idx, &filters).expect("filters present");
        assert_eq!(allow, [1usize].into_iter().collect::<HashSet<usize>>());

        // used as the restrict on a text query => only doc1, total reflects the narrowed set
        let q = Query::Term {
            field: Some("body".into()),
            text: "ticket".into(),
        };
        let out = search_with_weights_paged(
            &idx,
            &q,
            &HashMap::new(),
            Similarity::Bm25,
            0.0,
            0,
            10,
            None,
            Some(&allow),
            None,
        );
        assert_eq!(out.hits.iter().map(|h| h.id).collect::<Vec<_>>(), vec![1]);
        assert_eq!(out.total, 1);

        // no filters => None (no restriction)
        assert!(range_allow_list(&idx, &[]).is_none());

        // two range filters AND together: created_at in [100,200] AND created_at in [200,300] => {doc1}
        let both = vec![
            RangeFilter {
                field: "created_at_key".into(),
                lower: Some("100".into()),
                upper: Some("200".into()),
            },
            RangeFilter {
                field: "created_at_key".into(),
                lower: Some("200".into()),
                upper: Some("300".into()),
            },
        ];
        assert_eq!(
            range_allow_list(&idx, &both).unwrap(),
            [1usize].into_iter().collect::<HashSet<usize>>()
        );
    }

    #[test]
    fn match_all_allow_list_and_semantics_and_shortcircuit() {
        // doc0 title "vpn setup guide", doc1 "vpn guide", doc2 "setup only", doc3 "vpn"
        let mut idx = MemoryIndex::new();
        for t in ["vpn setup guide", "vpn guide", "setup only", "vpn"] {
            let mut d = Document::new();
            d.add("title", t, FieldKind::Text);
            idx.add_document(d);
        }
        // "vpn guide" (AND) => docs containing BOTH words in title: doc0, doc1
        let f = vec![MatchAllFilter {
            field: "title".into(),
            text: "vpn guide".into(),
        }];
        let mut got: Vec<usize> = match_all_allow_list(&idx, &f)
            .unwrap()
            .into_iter()
            .collect();
        got.sort_unstable();
        assert_eq!(got, vec![0, 1]);

        // a word absent from the field => empty (short-circuit on doc_freq 0)
        let f = vec![MatchAllFilter {
            field: "title".into(),
            text: "vpn absent".into(),
        }];
        assert!(match_all_allow_list(&idx, &f).unwrap().is_empty());

        // two matchAll filters AND together (both on title): "vpn" AND "setup" => doc0 only
        let f = vec![
            MatchAllFilter {
                field: "title".into(),
                text: "vpn".into(),
            },
            MatchAllFilter {
                field: "title".into(),
                text: "setup".into(),
            },
        ];
        assert_eq!(
            match_all_allow_list(&idx, &f).unwrap(),
            [0usize].into_iter().collect::<HashSet<usize>>()
        );

        // no filters => None (no constraint)
        assert!(match_all_allow_list(&idx, &[]).is_none());

        // text analyzing to no tokens => no constraint contributed => None
        let f = vec![MatchAllFilter {
            field: "title".into(),
            text: "   ".into(),
        }];
        assert!(match_all_allow_list(&idx, &f).is_none());
    }

    #[test]
    fn range_allow_list_drops_a_filter_that_excludes_nothing() {
        let mut idx = MemoryIndex::new();
        for ca in ["100", "200", "300"] {
            let mut d = Document::new();
            d.add("body", "ticket", FieldKind::Text);
            d.add("created_at_key", ca, FieldKind::Keyword);
            idx.add_document(d);
        }
        // covers every value present => no doc is excluded => equivalent to no restriction
        let filters = vec![RangeFilter {
            field: "created_at_key".into(),
            lower: Some("050".into()),
            upper: Some("400".into()),
        }];
        assert!(range_allow_list(&idx, &filters).is_none());

        // a bound that excludes one doc must still restrict
        let filters = vec![RangeFilter {
            field: "created_at_key".into(),
            lower: Some("050".into()),
            upper: Some("250".into()),
        }];
        let got = range_allow_list(&idx, &filters).expect("still restricts");
        assert_eq!(got, [0usize, 1].into_iter().collect::<HashSet<usize>>());
    }

    #[test]
    fn drop_if_universal_does_not_collapse_an_empty_index_to_unrestricted() {
        // an empty index (num_docs 0) must NOT collapse a genuinely-unsatisfiable filter to
        // "no restriction" — that's what the `!s.is_empty()` guard in `drop_if_universal` is for.
        assert_eq!(
            drop_if_universal(Some(HashSet::new()), 0),
            Some(HashSet::new())
        );
    }

    #[test]
    fn range_allow_list_keeps_restricting_when_a_doc_lacks_the_field() {
        // OpenSearch parity: a doc with no value in the range field never matches a range
        // filter. An unbounded range therefore is NOT equivalent to "no restriction" here.
        let mut idx = MemoryIndex::new();
        let mut d0 = Document::new();
        d0.add("body", "ticket", FieldKind::Text);
        d0.add("created_at_key", "100", FieldKind::Keyword);
        idx.add_document(d0);
        let mut d1 = Document::new(); // deliberately has no created_at_key
        d1.add("body", "ticket", FieldKind::Text);
        idx.add_document(d1);

        let filters = vec![RangeFilter {
            field: "created_at_key".into(),
            lower: None,
            upper: None,
        }];
        let got = range_allow_list(&idx, &filters).expect("must not collapse to None");
        assert_eq!(got, [0usize].into_iter().collect::<HashSet<usize>>());
    }

    #[test]
    fn match_all_allow_list_drops_a_filter_that_excludes_nothing() {
        let mut idx = MemoryIndex::new();
        for t in ["vpn setup", "vpn guide", "vpn"] {
            let mut d = Document::new();
            d.add("title", t, FieldKind::Text);
            idx.add_document(d);
        }
        // "vpn" is in every doc's title => the filter excludes nothing
        let f = vec![MatchAllFilter {
            field: "title".into(),
            text: "vpn".into(),
        }];
        assert!(match_all_allow_list(&idx, &f).is_none());

        // "setup" is not => still restricts
        let f = vec![MatchAllFilter {
            field: "title".into(),
            text: "setup".into(),
        }];
        let got = match_all_allow_list(&idx, &f).expect("still restricts");
        assert_eq!(got, [0usize].into_iter().collect::<HashSet<usize>>());
    }

    #[test]
    fn intersect_allow_treats_none_as_unconstrained() {
        let a: HashSet<usize> = [1, 2, 3].into_iter().collect();
        let b: HashSet<usize> = [2, 3, 4].into_iter().collect();
        // None is the identity (no constraint)
        assert_eq!(intersect_allow(None, Some(a.clone())), Some(a.clone()));
        assert_eq!(intersect_allow(Some(a.clone()), None), Some(a.clone()));
        assert_eq!(intersect_allow(None, None), None);
        // both present => intersection
        assert_eq!(
            intersect_allow(Some(a), Some(b)),
            Some([2usize, 3].into_iter().collect())
        );
    }

    #[test]
    fn match_all_narrows_a_text_query_via_restrict() {
        // full search: text "guide" matches doc0,doc1; matchAll title contains "setup" => doc0
        let mut idx = MemoryIndex::new();
        for t in ["vpn setup guide", "vpn guide", "setup only"] {
            let mut d = Document::new();
            d.add("title", t, FieldKind::Text);
            idx.add_document(d);
        }
        let allow = match_all_allow_list(
            &idx,
            &[MatchAllFilter {
                field: "title".into(),
                text: "setup".into(),
            }],
        )
        .unwrap();
        let q = Query::Term {
            field: Some("title".into()),
            text: "guide".into(),
        };
        let out = search_with_weights_paged(
            &idx,
            &q,
            &HashMap::new(),
            Similarity::Bm25,
            0.0,
            0,
            10,
            None,
            Some(&allow),
            None,
        );
        assert_eq!(out.hits.iter().map(|h| h.id).collect::<Vec<_>>(), vec![0]);
        assert_eq!(out.total, 1);
    }

    #[test]
    fn intersect_in_place_matches_set_intersection_either_way_round() {
        let set_a: HashSet<usize> = (0..100).collect();
        let set_b: HashSet<usize> = (50..500).collect();
        let expected: HashSet<usize> = (50..100).collect();

        // commutative: which operand is larger must not change the contents
        assert_eq!(
            intersect_in_place(set_a.clone(), set_b.clone()),
            expected,
            "small first"
        );
        assert_eq!(intersect_in_place(set_b, set_a), expected, "large first");

        // disjoint => empty
        let left: HashSet<usize> = (0..10).collect();
        let right: HashSet<usize> = (10..20).collect();
        assert!(intersect_in_place(left, right).is_empty(), "disjoint");

        // empty operand => empty, in either position
        let base: HashSet<usize> = (0..10).collect();
        assert!(
            intersect_in_place(HashSet::new(), base.clone()).is_empty(),
            "empty first"
        );
        assert!(
            intersect_in_place(base, HashSet::new()).is_empty(),
            "empty second"
        );
    }

    fn count_boosted_terms(q: &Query, boost: f32) -> usize {
        match q {
            Query::Boosted { boost: b, .. } if (*b - boost).abs() < f32::EPSILON => 1,
            Query::Boolean { clauses } => clauses
                .iter()
                .map(|(_, c)| count_boosted_terms(c, boost))
                .sum(),
            _ => 0,
        }
    }

    #[test]
    fn synonyms_off_adds_no_boosted_clauses() {
        let dict = crate::synonyms::from_pairs(&[("laptop", &["notebook"])]);
        let p = params("laptop"); // synonyms defaults to false in the helper
        let q = text_subquery_with_dict(&p, Some(&dict));
        assert_eq!(count_boosted_terms(&q, SYNONYM_BOOST), 0);
    }

    /// Structural proxy for laziness: `text_subquery_with_dict` must work with `dict: None`
    /// and produce the exact same (no-Boosted) clause structure as the `Some(&dict)` case
    /// above, i.e. it never dereferences `dict` when `p.synonyms` is false. This is what
    /// `text_subquery` actually passes in production (`p.synonyms.then(crate::synonyms::global)`
    /// short-circuits to `None` without calling `global()`), so this proves the call site
    /// can go through the off path without ever touching (and therefore never parsing) the
    /// bundled dictionary.
    #[test]
    fn synonyms_off_never_touches_the_dict_even_as_none() {
        let p = params("laptop"); // synonyms defaults to false in the helper
        let q = text_subquery_with_dict(&p, None);
        assert_eq!(count_boosted_terms(&q, SYNONYM_BOOST), 0);
    }

    #[test]
    fn synonyms_on_adds_one_boosted_should_per_expansion() {
        let dict = crate::synonyms::from_pairs(&[("laptop", &["notebook", "portátil"])]);
        let mut p = params("laptop");
        p.synonyms = true;
        let q = text_subquery_with_dict(&p, Some(&dict));
        assert_eq!(count_boosted_terms(&q, SYNONYM_BOOST), 2);
    }

    #[test]
    fn synonym_leaf_is_plain_term_when_accents_off() {
        let dict = crate::synonyms::from_pairs(&[("laptop", &["notebook"])]);
        let mut p = params("laptop");
        p.synonyms = true; // accent_insensitive stays false
        let q = text_subquery_with_dict(&p, Some(&dict));
        // a Boosted synonym clause exists at all (not vacuously true below)...
        assert!(has_boosted_term_leaf(&q));
        // ...and specifically its inner leaf is a plain Term, not an AccentTerm.
        assert!(!boosted_leaf_is_accent_term(&q));
    }

    #[test]
    fn synonym_leaf_is_accent_term_when_accents_on() {
        let dict = crate::synonyms::from_pairs(&[("laptop", &["notebook"])]);
        let mut p = params("laptop");
        p.synonyms = true;
        p.accent_insensitive = true;
        let q = text_subquery_with_dict(&p, Some(&dict));
        // the Boosted synonym clause's inner leaf specifically must be an AccentTerm
        // (not just some AccentTerm anywhere in the tree, which the literal per-token
        // clause already provides when accent_insensitive is on).
        assert!(boosted_leaf_is_accent_term(&q));
    }

    #[test]
    fn build_query_synonyms_reach_cross_lingual_doc() {
        // index one doc that only mentions the English term (same pattern as `accent_index`)
        let mut idx = MemoryIndex::new();
        let mut d = Document::new();
        d.add("title", "the office printer is broken", FieldKind::Text);
        idx.add_document(d);

        // query the Spanish term; without synonyms it must NOT match
        let mut p = params("impresora");
        assert_eq!(
            ids(&search(&idx, &build_query(&p).unwrap(), 0.0, 100)),
            Vec::<usize>::new()
        );

        // with synonyms on, "impresora" expands to "printer" and reaches the doc
        p.synonyms = true;
        assert_eq!(
            ids(&search(&idx, &build_query(&p).unwrap(), 0.0, 100)),
            vec![0]
        );
    }

    fn term(p: &str) -> BoolNode {
        BoolNode::Term(p.into())
    }

    fn phrase_leaf(words: &[&str]) -> Query {
        Query::Phrase {
            field: None,
            terms: words.iter().map(|w| (*w).to_string()).collect(),
            accent_insensitive: false,
        }
    }

    #[test]
    fn term_node_maps_to_an_all_fields_phrase() {
        let q = map_bool_node(&term("mi laptop"), false, 0).unwrap();
        assert_eq!(q, phrase_leaf(&["mi", "laptop"]));
    }

    #[test]
    fn and_node_maps_children_to_must() {
        let node = BoolNode::And(vec![term("uno"), term("dos")]);
        let Query::Boolean { clauses } = map_bool_node(&node, false, 0).unwrap() else {
            panic!("and debe mapear a Boolean");
        };
        assert_eq!(clauses.len(), 2);
        assert!(clauses.iter().all(|(o, _)| *o == Occur::Must));
    }

    #[test]
    fn or_node_maps_children_to_should() {
        let node = BoolNode::Or(vec![term("uno"), term("dos")]);
        let Query::Boolean { clauses } = map_bool_node(&node, false, 0).unwrap() else {
            panic!("or debe mapear a Boolean");
        };
        assert_eq!(clauses.len(), 2);
        assert!(clauses.iter().all(|(o, _)| *o == Occur::Should));
    }

    #[test]
    fn not_child_of_an_and_becomes_the_parents_mustnot() {
        // `foo AND NOT bar`: la negación se aplana como MustNot del AND padre, que es
        // exactamente lo que eval_boolean sabe resolver (requerido menos excluido).
        let node = BoolNode::And(vec![term("foo"), BoolNode::Not(Box::new(term("bar")))]);
        let Query::Boolean { clauses } = map_bool_node(&node, false, 0).unwrap() else {
            panic!("and debe mapear a Boolean");
        };
        let occurs: Vec<Occur> = clauses.iter().map(|(o, _)| *o).collect();
        assert_eq!(occurs, vec![Occur::Must, Occur::MustNot]);
        // y el MustNot envuelve la frase directamente, sin un Boolean intermedio
        assert_eq!(clauses[1].1, phrase_leaf(&["bar"]));
    }

    #[test]
    fn double_negation_cancels() {
        // el parser de SD puede emitir `foo AND NOT NOT bar` (pasa requiresPositiveMatch).
        // Sin normalizar, el MustNot de un Boolean vacío no excluye nada y la query
        // colapsa a `foo`: un resultado INCORRECTO, más amplio que lo pedido.
        let node = BoolNode::And(vec![
            term("foo"),
            BoolNode::Not(Box::new(BoolNode::Not(Box::new(term("bar"))))),
        ]);
        let Query::Boolean { clauses } = map_bool_node(&node, false, 0).unwrap() else {
            panic!("and debe mapear a Boolean");
        };
        let occurs: Vec<Occur> = clauses.iter().map(|(o, _)| *o).collect();
        assert_eq!(occurs, vec![Occur::Must, Occur::Must], "NOT NOT x == x");
    }

    #[test]
    fn accent_flag_reaches_every_leaf() {
        let node = BoolNode::And(vec![term("uno"), BoolNode::Or(vec![term("dos")])]);
        let q = map_bool_node(&node, true, 0).unwrap();
        assert!(
            !format!("{q:?}").contains("accent_insensitive: false"),
            "el flag tiene que llegar a TODAS las hojas: {q:?}"
        );
    }

    #[test]
    fn a_tree_deeper_than_the_cap_is_rejected() {
        let mut node = term("hoja");
        for _ in 0..40 {
            node = BoolNode::And(vec![node]);
        }
        assert_eq!(map_bool_node(&node, false, 0), Err(QueryError::TreeTooDeep));
    }

    #[test]
    fn a_tree_that_only_negates_at_the_root_subtracts_from_every_doc() {
        // Igual que un `where mustnot`: todo menos lo negado. Un `not` anidado bajo un `or`
        // sigue sin aportar nada propio (docs/API.md), y un `and`/`or` vacío no matchea nada.
        // corpus: doc0 "vpn guide", doc1 "vpn setup", doc2 "mysql notes".
        let not = |p: &str| BoolNode::Not(Box::new(term(p)));
        let idx = corpus();
        for (node, expected) in [
            (not("vpn"), vec![2]),
            (BoolNode::And(vec![not("guide"), not("mysql")]), vec![1]),
            (BoolNode::Or(vec![not("vpn")]), vec![]),
            (BoolNode::And(vec![]), vec![]),
            (BoolNode::Or(vec![]), vec![]),
        ] {
            let mut p = params("");
            p.boolean_tree = Some(node.clone());
            let q = build_query(&p).unwrap();
            assert_eq!(ids(&search(&idx, &q, 0.0, 10)), expected, "{node:?}");
        }
    }

    #[test]
    fn a_term_leaf_that_analyzes_to_zero_tokens_matches_nothing() {
        // A `term` leaf whose phrase the analyzer reduces to zero tokens (no letters/digits/
        // the punctuation the tokenizer keeps — see analysis.rs's TOKEN_RE) must behave like
        // "match nothing", never "match everything": this is guaranteed today only by
        // phrase_scores' `terms.is_empty()` early return, not by any check here.
        //
        // NB: not every punctuation-only string qualifies — `analyze("...")`/`analyze("---")`
        // actually yield a ONE-token phrase (`.`/`-` are inside TOKEN_RE's char class, see
        // `emits_punctuation_only_tokens` in analysis.rs), so those are non-empty leaves, not
        // this case. `!` and `?` are outside the class, so they do reduce to zero tokens.
        let idx = corpus();
        for phrase in ["!!!", "???"] {
            assert!(
                crate::analysis::analyze(phrase).is_empty(),
                "test premise: {phrase:?} must analyze to zero tokens"
            );
            let q = map_bool_node(&term(phrase), false, 0).unwrap();
            let hits = search(&idx, &q, 0.0, 10);
            assert!(
                hits.is_empty(),
                "an empty-token leaf {phrase:?} must match zero docs, not everything: {:?}",
                ids(&hits)
            );
        }
    }

    #[test]
    fn a_zero_token_leaf_does_not_widen_an_or() {
        // An empty-token leaf next to a real leaf inside an `or` must contribute nothing to
        // the union — the sibling's result set must stay exactly as narrow as it would alone.
        let idx = corpus();
        let with_empty_sibling =
            map_bool_node(&BoolNode::Or(vec![term("!!!"), term("vpn")]), false, 0).unwrap();
        let alone = map_bool_node(&term("vpn"), false, 0).unwrap();
        assert_eq!(
            ids(&search(&idx, &with_empty_sibling, 0.0, 10)),
            ids(&search(&idx, &alone, 0.0, 10)),
            "a zero-token sibling must not widen the or's result"
        );
    }

    #[test]
    fn a_tree_replaces_the_free_text_subquery() {
        let mut p = params("texto que se ignora");
        p.boolean_tree = Some(BoolNode::And(vec![term("quick"), term("brown")]));
        let q = build_query(&p).unwrap();
        let dump = format!("{q:?}");
        assert!(dump.contains("Phrase"), "el árbol tiene que estar: {dump}");
        assert!(
            !dump.contains("Fuzzy"),
            "con árbol, el sub-query de texto libre NO se arma: {dump}"
        );
    }

    #[test]
    fn a_tree_with_empty_text_is_not_an_empty_query() {
        let mut p = params("");
        p.boolean_tree = Some(BoolNode::Or(vec![term("quick")]));
        assert!(
            build_query(&p).is_ok(),
            "un árbol cuenta como algo que buscar aunque text esté vacío"
        );
    }

    #[test]
    fn a_tree_still_ands_with_where_and_in() {
        let mut p = params("");
        p.boolean_tree = Some(BoolNode::Or(vec![term("quick")]));
        p.where_groups = vec![WhereGroup {
            field: "status".into(),
            values: vec!["open".into()],
            occur: Occur::Must,
        }];
        p.in_groups = vec![InGroup {
            field: "category".into(),
            values: vec!["10".into()],
        }];
        let Query::Boolean { clauses } = build_query(&p).unwrap() else {
            panic!("build_query devuelve un Boolean");
        };
        // árbol (1 Must) + where (1 Must) + in (todos los grupos IN colapsan en UNA sola
        // cláusula Must, ver el comentario de build_query junto a `in_clauses`) = 3.
        assert_eq!(
            clauses.len(),
            3,
            "árbol + where + in, las 3 en Must: {clauses:?}"
        );
        assert!(clauses.iter().all(|(o, _)| *o == Occur::Must));
    }

    #[test]
    fn the_tree_inherits_accent_insensitive_from_the_params() {
        let mut p = params("");
        p.accent_insensitive = true;
        p.boolean_tree = Some(BoolNode::Or(vec![term("impresion")]));
        let dump = format!("{:?}", build_query(&p).unwrap());
        assert!(
            !dump.contains("accent_insensitive: false"),
            "el flag de la query tiene que bajar a las hojas: {dump}"
        );
    }

    #[test]
    fn a_too_deep_tree_surfaces_as_an_error_not_a_crash() {
        let mut node = term("hoja");
        for _ in 0..40 {
            node = BoolNode::And(vec![node]);
        }
        let mut p = params("");
        p.boolean_tree = Some(node);
        assert_eq!(build_query(&p), Err(QueryError::TreeTooDeep));
    }
}
