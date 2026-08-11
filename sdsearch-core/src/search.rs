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
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct Hit {
    pub id: usize,
    pub score: f32,
    pub fields: HashMap<String, String>,
}

/// Result of a paged search: the requested page of hits plus the (optionally capped) total
/// match count. `total_capped` is true when the real match count exceeded the cap.
pub struct SearchOutcome {
    pub hits: Vec<Hit>,
    pub total: usize,
    pub total_capped: bool,
}

/// Field sort selector. `field` is used VERBATIM (the caller passes the `_key`-suffixed name,
/// consistent with range/matchAll). Relevance order is the ABSENCE of a `SortSpec`, not a
/// special value here.
pub struct SortSpec {
    pub field: String,
    pub ascending: bool,
}

/// A `sort_dir` token the caller spelled in a way this engine does not accept.
#[derive(Debug, PartialEq, Eq)]
pub struct InvalidSortDir(pub String);

impl std::fmt::Display for InvalidSortDir {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "unknown sort_dir {:?} (expected \"asc\" or \"desc\")",
            self.0
        )
    }
}
impl std::error::Error for InvalidSortDir {}

impl SortSpec {
    /// Builds a spec from the caller-facing direction token. Omitted = descending, matching the
    /// adapter default.
    ///
    /// Anything other than `"asc"`/`"desc"` is REJECTED rather than silently read as descending:
    /// `"ASC"`, `"ascending"` or a plain typo would otherwise reverse the caller's intended order
    /// with no signal anywhere — a wrong answer that looks like a working query. Parsing lives
    /// here, not at the FFI boundary, so the accepted set is covered by core's test suite
    /// (`sdsearch-php` is a `cdylib` and runs no tests of its own).
    pub fn new(field: String, dir: Option<&str>) -> Result<Self, InvalidSortDir> {
        let ascending = match dir {
            None | Some("desc") => false,
            Some("asc") => true,
            Some(other) => return Err(InvalidSortDir(other.to_string())),
        };
        Ok(Self { field, ascending })
    }
}

/// A doc's sort value, resolved at READ time from its stored field — never by re-encoding what
/// the index holds, so this works on indexes the legacy PHP engine wrote, with no reindex.
///
/// The variant is decided PER VALUE, not per field. That matters: deciding per field would mean
/// inspecting the field's whole vocabulary first, which is measured at 4.2 s on a 500k-doc index
/// (see the sort plan) — the exact cost this design exists to avoid. Per value it is a `parse`.
///
/// The derived `Ord` gives `Num < Text < Missing` and orders each variant by its payload, which
/// is the ascending output order. Two consequences worth stating:
/// - numeric fields order NUMERICALLY over raw, un-padded values (`"3" < "20" < "100"`), so the
///   feed never has to zero-pad;
/// - text fields fall back to byte order, which for ISO-8601 timestamps IS chronological order.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
enum SortKey {
    Num(i64),
    Text(String),
    Missing,
}

impl SortKey {
    /// Classifies one stored value. `None` (no such field on this doc) becomes `Missing`.
    fn from_stored(value: Option<String>) -> SortKey {
        match value {
            None => SortKey::Missing,
            Some(v) => match v.parse::<i64>() {
                Ok(n) => SortKey::Num(n),
                Err(_) => SortKey::Text(v),
            },
        }
    }
}

/// One candidate in the bounded top-K heap.
///
/// `Ord` is defined so that GREATER means "later in the final output". `BinaryHeap` is a max-heap,
/// so its root is then the WORST entry currently kept — exactly what a bounded top-K needs to
/// evict. `ascending` rides along per entry (it lands in padding the struct already had, so it
/// costs nothing) because every entry in one heap shares the same direction.
struct SortEntry {
    key: SortKey,
    id: usize,
    score: f32,
    ascending: bool,
}

impl Ord for SortEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Missing sorts LAST in both directions, so the missing flag is compared ascending
        // ALWAYS — outside the direction flip. Flipping it with the rest would float
        // value-less docs to the top of a descending page.
        let a_missing = matches!(self.key, SortKey::Missing);
        let b_missing = matches!(other.key, SortKey::Missing);
        a_missing
            .cmp(&b_missing)
            .then_with(|| {
                let by_key = self.key.cmp(&other.key);
                if self.ascending {
                    by_key
                } else {
                    by_key.reverse()
                }
            })
            // same tiebreak as the relevance path: score desc, then id asc (deterministic).
            .then_with(|| {
                other
                    .score
                    .partial_cmp(&self.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for SortEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Required by `Eq`, which `Ord` requires, which `BinaryHeap` requires — nothing ever calls it,
/// so it will always show as uncovered. Do NOT "fix" that with a test asserting it agrees with
/// `cmp`: it is *defined* as `cmp`, so such a test is tautological and buys nothing.
///
/// It is hand-written rather than derived because a derived `PartialEq` would compare fields
/// structurally and disagree with `cmp` — which treats a NaN score as `Equal` — breaking the
/// total-order contract `BinaryHeap` and `sort_unstable` rely on.
impl PartialEq for SortEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}

impl Eq for SortEntry {}

/// raw scores (doc_id, score) of a term in a field. No sort/filter/hydration.
/// The idf is computed ONCE (constant over the posting list), not per doc.
/// `restrict`: `None` scores all matching docs; `Some(set)` scores only docs in `set`. It filters WHICH docs are scored, never the score — idf uses collection-wide stats, so a scored doc's value is identical either way.
pub(crate) fn term_scores(
    index: &impl IndexReader,
    sim: Similarity,
    field: &str,
    term: &str,
    restrict: Option<&HashSet<usize>>,
) -> Vec<(usize, f32)> {
    let idf = sim.idf(
        index.total_docs() as f32,
        index.doc_freq(field, term) as f32,
    );
    let avg = index.avg_field_len(field);
    index
        .postings_for(field, term)
        .into_iter()
        .filter(|(doc_id, _)| restrict.is_none_or(|r| r.contains(doc_id)))
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
/// `restrict`: `None` scores all matching docs; `Some(set)` scores only docs in `set`. It filters WHICH docs are scored, never the score — idf uses collection-wide stats, so a scored doc's value is identical either way.
pub(crate) fn union_scores(
    index: &impl IndexReader,
    sim: Similarity,
    field: &str,
    terms: &[&str],
    restrict: Option<&HashSet<usize>>,
) -> HashMap<usize, f32> {
    let mut scored: HashMap<usize, f32> = HashMap::new();
    let avg = index.avg_field_len(field);
    for term in terms {
        let idf = sim.idf(
            index.total_docs() as f32,
            index.doc_freq(field, term) as f32,
        );
        for (doc_id, tf) in index.postings_for(field, term) {
            if restrict.is_some_and(|r| !r.contains(&doc_id)) {
                continue;
            }
            *scored.entry(doc_id).or_insert(0.0) +=
                sim.score(idf, tf, index.field_len(doc_id, field), avg);
        }
    }
    scored
}

/// Safety backstop: max terms a single wildcard leaf expands to, per field. With the default
/// `wildcard_min_prefix` this essentially never fires; it exists so a pathological broad
/// prefix (or an opted-in zero-prefix wildcard) cannot melt down. 4096 = 4× Zend's 1024
/// terms-per-query default, for recall headroom.
const MAX_WILDCARD_TERMS: usize = 4096;

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
        .terms_with_prefix_limited(field, prefix, MAX_WILDCARD_TERMS)
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
/// `restrict`: `None` scores all matching docs; `Some(set)` scores only docs in `set`. It filters WHICH docs are scored, never the score — idf uses collection-wide stats, so a scored doc's value is identical either way.
pub(crate) fn phrase_scores(
    index: &impl IndexReader,
    sim: Similarity,
    field: &str,
    terms: &[&str],
    restrict: Option<&HashSet<usize>>,
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
    let mut candidates: Vec<usize> = per_term[0]
        .0
        .keys()
        .copied()
        .filter(|d| restrict.is_none_or(|r| r.contains(d)))
        .collect();
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
/// `stored_fields` ONLY for the surviving hits. Thin wrapper over `finalize_paged` with
/// offset 0 and no total cap.
pub(crate) fn finalize(
    index: &impl IndexReader,
    scored: impl IntoIterator<Item = (usize, f32)>,
    min_score: f32,
    limit: usize,
) -> Vec<Hit> {
    finalize_paged(index, scored, min_score, 0, limit, None).hits
}

/// Like `finalize`, but returns the page `[offset, offset+limit)` of the ranking plus the
/// total match count. `total_cap`: `None` = exact count; `Some(cap)` saturates `total` at
/// `cap` and sets `total_capped` when the real count exceeds it. Only the top `offset+limit`
/// docs are selected before the sort; stored fields hydrate for the returned page only.
pub(crate) fn finalize_paged(
    index: &impl IndexReader,
    scored: impl IntoIterator<Item = (usize, f32)>,
    min_score: f32,
    offset: usize,
    limit: usize,
    total_cap: Option<usize>,
) -> SearchOutcome {
    let mut ranked: Vec<(usize, f32)> = scored
        .into_iter()
        .filter(|(_, s)| *s >= min_score)
        .collect();
    let count = ranked.len();
    let total = total_cap.map_or(count, |cap| count.min(cap));
    let total_capped = total_cap.is_some_and(|cap| count > cap);

    // score desc, id asc — the single comparator used by both the partition and the sort.
    let cmp = |a: &(usize, f32), b: &(usize, f32)| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    };
    // Materialize only the top `offset + limit` docs before sorting the page. `saturating_add`
    // guards `limit == usize::MAX` (the runner's "unlimited"), which then takes the full sort.
    let need = offset.saturating_add(limit);
    if need < ranked.len() {
        ranked.select_nth_unstable_by(need, cmp);
        ranked.truncate(need);
    }
    ranked.sort_by(cmp);
    let hits = ranked
        .into_iter()
        .skip(offset)
        .take(limit)
        .map(|(id, score)| Hit {
            id,
            score,
            fields: index.stored_fields(id),
        })
        .collect();
    SearchOutcome {
        hits,
        total,
        total_capped,
    }
}

/// Field-sorted counterpart of `finalize_paged`: same filtering, paging and total semantics, but
/// ordered by a field's value instead of by score.
///
/// Walks the MATCHED docs (never the sort field's term dictionary) and resolves each one's value
/// with `stored_value`, feeding a heap bounded to `offset + limit`. So:
/// - the sort path performs ZERO term-dictionary lookups — no `terms_in_range`, no `doc_freq`,
///   no per-term pre-pass. That is deliberate: an ordered term walk over a near-unique field
///   measured 4.2 s at 500k docs, and it is paid up front regardless of how few docs match;
/// - retained memory is `K' = offset + limit` entries, not one per match;
/// - `min_score` is applied BEFORE the value lookup, so filtered-out docs never touch the `.fdt`.
///
/// `limit == usize::MAX` (the runner's "unlimited") makes the bound vacuous and the heap retains
/// every match — inherent to an unbounded request, and the same O(matches) the relevance path
/// already pays there.
pub(crate) fn finalize_sorted(
    index: &impl IndexReader,
    scored: impl IntoIterator<Item = (usize, f32)>,
    min_score: f32,
    sort: &SortSpec,
    offset: usize,
    limit: usize,
    total_cap: Option<usize>,
) -> SearchOutcome {
    // `saturating_add` guards `limit == usize::MAX`. Note this is offset PLUS limit, not times:
    // paging to offset 10000 keeps 10020 entries, not 200000.
    let want = offset.saturating_add(limit);
    let mut heap: BinaryHeap<SortEntry> = BinaryHeap::new();
    let mut count = 0usize;

    for (id, score) in scored {
        if score < min_score {
            continue;
        }
        // every surviving match counts toward `total`, even when it cannot reach the page
        count += 1;
        if want == 0 {
            continue;
        }
        let entry = SortEntry {
            key: SortKey::from_stored(index.stored_value(id, &sort.field)),
            id,
            score,
            ascending: sort.ascending,
        };
        if heap.len() < want {
            heap.push(entry);
        } else if heap.peek().is_some_and(|worst| entry < *worst) {
            // better than the worst entry kept: evict it. Otherwise drop `entry` on the floor —
            // this is the branch the overwhelming majority of matches take.
            heap.pop();
            heap.push(entry);
        }
    }

    let total = total_cap.map_or(count, |cap| count.min(cap));
    let total_capped = total_cap.is_some_and(|cap| count > cap);

    // a heap only orders its root, so the retained K' entries still need a sort — O(K' log K'),
    // dominated by the O(matches) walk above.
    let mut ordered = heap.into_vec();
    ordered.sort_unstable();
    let hits = ordered
        .into_iter()
        .skip(offset)
        .take(limit)
        .map(|e| Hit {
            id: e.id,
            score: e.score,
            fields: index.stored_fields(e.id),
        })
        .collect();
    SearchOutcome {
        hits,
        total,
        total_capped,
    }
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
        term_scores(index, Similarity::Bm25, field, term, None),
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
        union_scores(index, Similarity::Bm25, field, terms, None),
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
        union_scores(index, Similarity::Bm25, field, &refs, None),
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
        union_scores(index, Similarity::Bm25, field, &refs, None),
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
        phrase_scores(index, Similarity::Bm25, field, terms, None),
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
    fn finalize_paged_offset_total_and_cap() {
        // 6 docs so stored_fields hydrate; scores with ties (ids 1,2,5 = 0.9) to exercise
        // the id-asc tiebreak that select_nth must preserve across the offset boundary.
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
        // reference ranking: score desc, id asc => [1, 2, 5, 4, 0, 3]
        let reference = [1usize, 2, 5, 4, 0, 3];

        // page (offset=2, limit=2) => 3rd and 4th of the ranking => [5, 4]
        let out = finalize_paged(&idx, scored.clone(), 0.0, 2, 2, None);
        let got: Vec<usize> = out.hits.iter().map(|h| h.id).collect();
        assert_eq!(got, reference[2..4].to_vec(), "offset+limit page");
        assert_eq!(out.total, 6, "no cap => exact count");
        assert!(!out.total_capped);

        // cap below the match count => total saturates and total_capped is true
        let out = finalize_paged(&idx, scored.clone(), 0.0, 0, 2, Some(3));
        assert_eq!(out.total, 3);
        assert!(out.total_capped);
        assert_eq!(out.hits.len(), 2);

        // min_score filter reduces the counted total (only scores >= 0.7 => ids 1,2,5,4)
        let out = finalize_paged(&idx, scored.clone(), 0.7, 0, 10, None);
        assert_eq!(out.total, 4);
        let got: Vec<usize> = out.hits.iter().map(|h| h.id).collect();
        assert_eq!(got, vec![1, 2, 5, 4]);

        // offset beyond the result set => empty page, total still reported
        let out = finalize_paged(&idx, scored, 0.0, 99, 5, None);
        assert!(out.hits.is_empty());
        assert_eq!(out.total, 6);
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

    // ---- field sort ----

    /// 6 docs whose `d_key` values are VARIABLE-WIDTH numbers, so numeric order
    /// ("3" < "20" < "100") disagrees with byte order ("100" < "20" < "3"). Doc 3 has no
    /// `d_key` at all. Scores are chosen so two docs share each value, exercising the tiebreak.
    fn sortable() -> (MemoryIndex, Vec<(usize, f32)>) {
        let mut idx = MemoryIndex::new();
        let rows = [
            (Some("20"), 0.5f32),
            (Some("3"), 0.9),
            (Some("20"), 0.7),
            (None, 0.4),
            (Some("3"), 0.2),
            (Some("100"), 0.6),
        ];
        for (v, _) in rows {
            let mut d = Document::new();
            d.add("body", "x", FieldKind::Text);
            if let Some(v) = v {
                d.add("d_key", v, FieldKind::Keyword);
            }
            idx.add_document(d);
        }
        let scored = rows.iter().enumerate().map(|(i, (_, s))| (i, *s)).collect();
        (idx, scored)
    }

    fn ids(out: &SearchOutcome) -> Vec<usize> {
        out.hits.iter().map(|h| h.id).collect()
    }

    fn spec(field: &str, ascending: bool) -> SortSpec {
        SortSpec {
            field: field.into(),
            ascending,
        }
    }

    #[test]
    fn sort_spec_new_accepts_only_asc_and_desc() {
        // omitted => descending, the adapter default
        assert_eq!(
            SortSpec::new("d_key".into(), None).map(|s| s.ascending),
            Ok(false)
        );
        assert_eq!(
            SortSpec::new("d_key".into(), Some("desc")).map(|s| s.ascending),
            Ok(false)
        );
        assert_eq!(
            SortSpec::new("d_key".into(), Some("asc")).map(|s| s.ascending),
            Ok(true)
        );

        // the field is carried through untouched (used verbatim, no `_key` inference)
        assert_eq!(
            SortSpec::new("created_at_key".into(), Some("asc")).map(|s| s.field),
            Ok("created_at_key".to_string())
        );

        // Everything else is an ERROR, not a silent fall back to descending. These three are the
        // realistic typos: wrong case, the long spelling, and an OpenSearch-ism. Each one would
        // otherwise hand back the exact reverse of what the caller asked for.
        for bad in ["ASC", "ascending", "descending", "", "up"] {
            assert_eq!(
                SortSpec::new("d_key".into(), Some(bad)).map(|s| s.ascending),
                Err(InvalidSortDir(bad.to_string())),
                "sort_dir {bad:?} must be rejected"
            );
        }

        assert_eq!(
            InvalidSortDir("ASC".into()).to_string(),
            r#"unknown sort_dir "ASC" (expected "asc" or "desc")"#
        );
    }

    #[test]
    fn finalize_sorted_orders_numerically_over_unpadded_values() {
        let (idx, scored) = sortable();
        // ascending: 3 (docs 1,4 -> score desc), 20 (docs 2,0), 100 (doc 5), then missing (doc 3).
        // Byte order would lead with "100" (doc 5) — asserting doc 1 first is what proves the
        // read-time numeric ordering, i.e. that the feed does NOT need to zero-pad.
        let out = finalize_sorted(&idx, scored.clone(), 0.0, &spec("d_key", true), 0, 10, None);
        assert_eq!(ids(&out), vec![1, 4, 2, 0, 5, 3]);
        assert_eq!(out.total, 6);
        assert!(!out.total_capped);

        // descending flips the VALUES but not the missing tail: 100, 20, 3, then doc 3 last.
        let out = finalize_sorted(&idx, scored, 0.0, &spec("d_key", false), 0, 10, None);
        assert_eq!(ids(&out), vec![5, 2, 0, 1, 4, 3]);
    }

    #[test]
    fn finalize_sorted_pages_totals_and_caps() {
        let (idx, scored) = sortable();
        let asc = spec("d_key", true);
        // full ascending order is [1, 4, 2, 0, 5, 3]
        let out = finalize_sorted(&idx, scored.clone(), 0.0, &asc, 1, 2, None);
        assert_eq!(ids(&out), vec![4, 2], "offset+limit page");
        assert_eq!(out.total, 6, "total is independent of the page");

        // the bounded heap must not disturb the offset boundary: every page of every size has to
        // agree with the full ordering, since the heap keeps exactly offset+limit entries.
        let full = ids(&finalize_sorted(
            &idx,
            scored.clone(),
            0.0,
            &asc,
            0,
            usize::MAX,
            None,
        ));
        for offset in 0..full.len() {
            for limit in 1..=full.len() {
                let page = ids(&finalize_sorted(
                    &idx,
                    scored.clone(),
                    0.0,
                    &asc,
                    offset,
                    limit,
                    None,
                ));
                let want: Vec<usize> = full.iter().skip(offset).take(limit).copied().collect();
                assert_eq!(page, want, "offset={offset} limit={limit}");
            }
        }

        // cap saturates `total` and flags it; the page itself is unaffected
        let out = finalize_sorted(&idx, scored.clone(), 0.0, &asc, 0, 2, Some(3));
        assert_eq!(out.total, 3);
        assert!(out.total_capped);
        assert_eq!(ids(&out), vec![1, 4]);

        // min_score filters BEFORE ordering and shrinks the total (keeps 1=0.9, 2=0.7, 5=0.6)
        let out = finalize_sorted(&idx, scored, 0.6, &asc, 0, 10, None);
        assert_eq!(ids(&out), vec![1, 2, 5]);
        assert_eq!(out.total, 3);
    }

    #[test]
    fn finalize_sorted_missing_values_sort_last_in_both_directions() {
        // every doc lacks the sort field => all Missing => the tiebreak alone decides,
        // and nothing panics on a field the index has never seen.
        let (idx, scored) = sortable();
        for ascending in [true, false] {
            let out = finalize_sorted(
                &idx,
                scored.clone(),
                0.0,
                &spec("absent_key", ascending),
                0,
                10,
                None,
            );
            // score desc, id asc: 1(.9), 2(.7), 5(.6), 0(.5), 3(.4), 4(.2)
            assert_eq!(ids(&out), vec![1, 2, 5, 0, 3, 4], "ascending={ascending}");
            assert_eq!(out.total, 6);
        }
    }

    #[test]
    fn finalize_sorted_with_a_zero_size_page_still_counts() {
        // offset+limit == 0 makes the heap pointless, but `total` must still be right, and the
        // loop must keep counting while skipping the value lookup entirely.
        //
        // Not reachable through the PHP surface: `search_index_paged` maps `limit: 0` to
        // `usize::MAX` ("unlimited") before this is called. It IS reachable by a Rust consumer
        // calling the public `search_with_weights_paged` directly, which is why it is a
        // supported case rather than dead code.
        let (inner, scored) = sortable();
        let idx = CountingIndex::new(inner);
        let out = finalize_sorted(&idx, scored, 0.0, &spec("d_key", true), 0, 0, None);
        assert!(out.hits.is_empty());
        assert_eq!(out.total, 6, "an empty page still reports the match count");
        assert!(!out.total_capped);
        // The short-circuit is the point: with no page to fill there is nothing to compare, so
        // not one doc should be read from the `.fdt`. Asserting only `total` here would pass
        // even with the short-circuit deleted — the result is the same, just N reads slower.
        assert!(
            idx.stored_value_calls.borrow().is_empty(),
            "a zero-size page must not read any stored value"
        );
        assert_eq!(idx.stored_fields_calls.get(), 0, "and hydrate nothing");

        // min_score and the cap still apply on that path
        let (inner, scored2) = sortable();
        let idx = CountingIndex::new(inner);
        let out = finalize_sorted(&idx, scored2, 0.6, &spec("d_key", true), 0, 0, Some(2));
        assert!(out.hits.is_empty());
        assert_eq!(out.total, 2);
        assert!(out.total_capped);
        assert!(idx.stored_value_calls.borrow().is_empty());
    }

    #[test]
    fn finalize_sorted_text_falls_back_to_byte_order() {
        let mut idx = MemoryIndex::new();
        for v in ["open", "closed", "pending"] {
            let mut d = Document::new();
            d.add("s_key", v, FieldKind::Keyword);
            idx.add_document(d);
        }
        let scored: Vec<(usize, f32)> = (0..3).map(|i| (i, 1.0)).collect();
        let out = finalize_sorted(&idx, scored, 0.0, &spec("s_key", true), 0, 10, None);
        // alphabetical: "closed"(1), "open"(0), "pending"(2)
        assert_eq!(ids(&out), vec![1, 0, 2]);
    }

    #[test]
    fn finalize_sorted_iso_timestamps_order_chronologically() {
        // A datetime string does not parse as i64, so it takes the Text branch — where byte
        // order IS chronological order for ISO-8601. This is why a text fallback is correct
        // rather than merely safe.
        let mut idx = MemoryIndex::new();
        for v in [
            "2026-07-27 10:00:00",
            "2025-01-02 23:59:59",
            "2026-07-27 09:59:59",
        ] {
            let mut d = Document::new();
            d.add("t_key", v, FieldKind::Keyword);
            idx.add_document(d);
        }
        let scored: Vec<(usize, f32)> = (0..3).map(|i| (i, 1.0)).collect();
        let out = finalize_sorted(&idx, scored, 0.0, &spec("t_key", true), 0, 10, None);
        assert_eq!(ids(&out), vec![1, 2, 0]);
    }

    #[test]
    fn finalize_sorted_places_a_multi_valued_doc_once() {
        let mut idx = MemoryIndex::new();
        let mut d = Document::new();
        d.add("m_key", "50", FieldKind::Keyword);
        d.add("m_key", "10", FieldKind::Keyword);
        idx.add_document(d);
        let mut d = Document::new();
        d.add("m_key", "30", FieldKind::Keyword);
        idx.add_document(d);

        let out = finalize_sorted(
            &idx,
            vec![(0, 1.0), (1, 1.0)],
            0.0,
            &spec("m_key", true),
            0,
            10,
            None,
        );
        // The point asserted here is placement: a doc with several values occupies ONE slot,
        // it does not appear once per value. WHICH of its values decides the slot is a reader
        // property, not a `finalize_sorted` one — `MemoryIndex` stores fields in a `HashMap`
        // and so cannot even hold two, while the on-disk reader walks the `.fdt` in write order.
        // That contract is pinned in `zsl::stored::read_stored_field_returns_the_first_of_a_repeated_field`.
        assert_eq!(out.hits.len(), 2, "doc 0 must occupy exactly one slot");
        assert_eq!(out.total, 2);
        let mut got = ids(&out);
        got.sort_unstable();
        assert_eq!(got, vec![0, 1]);
    }

    /// Pins the SHAPE of `finalize_sorted`'s reader access, so a refactor cannot quietly change
    /// the cost model. Two regressions this guards against are not hypothetical: an ordered term
    /// walk over the sort field measured 4.2 s at 500k docs, and a per-term `doc_freq` pre-pass
    /// cost a 1.9x regression in the range work.
    ///
    /// Every method sort must NEVER reach is `unreachable!` rather than counted. That is a
    /// stronger assertion than comparing a counter to zero — it covers the whole surface instead
    /// of the handful of methods someone remembered to count, and it names the violated invariant
    /// at the point of failure. It also means these arms are legitimately never executed: their
    /// absence from a coverage report is the property being asserted, not a gap in testing.
    struct CountingIndex {
        inner: MemoryIndex,
        stored_value_calls: std::cell::RefCell<Vec<usize>>,
        stored_fields_calls: std::cell::Cell<usize>,
    }

    impl CountingIndex {
        fn new(inner: MemoryIndex) -> CountingIndex {
            CountingIndex {
                inner,
                stored_value_calls: std::cell::RefCell::new(Vec::new()),
                stored_fields_calls: std::cell::Cell::new(0),
            }
        }
    }

    impl IndexReader for CountingIndex {
        // --- the only two the sort path may use ---
        fn stored_value(&self, doc_id: usize, field: &str) -> Option<String> {
            self.stored_value_calls.borrow_mut().push(doc_id);
            self.inner.stored_value(doc_id, field)
        }
        fn stored_fields(&self, doc_id: usize) -> HashMap<String, String> {
            self.stored_fields_calls
                .set(self.stored_fields_calls.get() + 1);
            self.inner.stored_fields(doc_id)
        }

        // --- term dictionary: reaching any of these is the regression ---
        fn doc_freq(&self, _field: &str, _term: &str) -> usize {
            unreachable!("field sort must not call doc_freq (dict.info() is a .tis seek+scan)")
        }
        fn postings_for(&self, _field: &str, _term: &str) -> Vec<(usize, u32)> {
            unreachable!("field sort must not read postings of the sort field")
        }
        fn terms_with_prefix(&self, _field: &str, _prefix: &str) -> Vec<String> {
            unreachable!("field sort must not enumerate the sort field's vocabulary")
        }
        fn terms_in_range(
            &self,
            _field: &str,
            _lower: Option<&str>,
            _upper: Option<&str>,
        ) -> Vec<String> {
            unreachable!("field sort must not walk the term dictionary (4.2 s at 500k docs)")
        }
        fn positions_for(&self, _field: &str, _term: &str, _doc_id: usize) -> Vec<u32> {
            unreachable!("field sort must not read positions")
        }

        // --- collection metadata: not needed either, and cheap to keep honest ---
        fn num_docs(&self) -> usize {
            unreachable!("field sort must not need collection statistics")
        }
        fn field_len(&self, _doc_id: usize, _field: &str) -> u32 {
            unreachable!("field sort must not need field lengths")
        }
        fn indexed_fields(&self) -> Vec<String> {
            unreachable!("field sort must not enumerate indexed fields")
        }
    }

    #[test]
    fn finalize_sorted_touches_each_matched_doc_once_and_the_dictionary_never() {
        let (inner, scored) = sortable();
        let idx = CountingIndex::new(inner);
        let out = finalize_sorted(&idx, scored, 0.0, &spec("d_key", true), 0, 2, None);
        assert_eq!(ids(&out), vec![1, 4]);

        // one value lookup per matched doc — never O(matches x K), never a second pass
        let mut looked_up = idx.stored_value_calls.borrow().clone();
        looked_up.sort_unstable();
        assert_eq!(looked_up, vec![0, 1, 2, 3, 4, 5]);

        // the expensive whole-doc hydration happens ONLY for the returned page
        assert_eq!(idx.stored_fields_calls.get(), 2, "hydrate the page, not N");

        // That this test reached its end at all is the dictionary assertion: every term-dictionary
        // method of `CountingIndex` is `unreachable!`, so an ordered walk or a doc_freq pre-pass
        // would have panicked above rather than merely bumped a counter.
    }

    #[test]
    fn finalize_sorted_skips_the_value_lookup_for_docs_below_min_score() {
        let (inner, scored) = sortable();
        let idx = CountingIndex::new(inner);
        // keeps only docs 1 (0.9), 2 (0.7) and 5 (0.6)
        let out = finalize_sorted(&idx, scored, 0.6, &spec("d_key", true), 0, 10, None);
        assert_eq!(ids(&out), vec![1, 2, 5]);

        let mut looked_up = idx.stored_value_calls.borrow().clone();
        looked_up.sort_unstable();
        assert_eq!(
            looked_up,
            vec![1, 2, 5],
            "docs filtered by min_score must never reach the .fdt"
        );
    }
}
