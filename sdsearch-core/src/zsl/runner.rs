//! runner: orchestrates ZslIndex + build_query + executor, reproducing the host
//! application's Zend Lucene search adapter (empty-result fallback, min_score,
//! limit==0 = unlimited).

use crate::analysis::analyze;
use crate::hybrid::{HybridParams, fuse_rrf};
use crate::index::IndexReader;
use crate::mlt::{MltParams, more_like_this};
use crate::prf::{PrfParams, search_prf};
use crate::query::{
    InGroup, Occur, Query, QueryError, QueryParams, build_query, search, search_with_weights,
};
use crate::search::Hit;
use crate::zsl::index::ZslIndex;
use std::collections::HashSet;
use std::path::Path;

/// Lexical retriever shared by `search_index` and `search_hybrid_index`: builds the query,
/// runs it, and falls back to an all-fields Boolean over the text when the primary result is
/// empty. `limit == 0` = unlimited. An invalid query propagates as `QueryError`.
fn lexical_search(
    index: &impl IndexReader,
    params: &QueryParams,
    min_score: f32,
    limit: usize,
) -> Result<Vec<Hit>, QueryError> {
    let query = build_query(params)?;
    let lim = if limit == 0 { usize::MAX } else { limit };
    let mut hits = search_with_weights(
        index,
        &query,
        &params.field_weights,
        params.similarity,
        min_score,
        lim,
    );
    if hits.is_empty() {
        if let Some(fb) = fallback_query(&params.text) {
            hits = search_with_weights(
                index,
                &fb,
                &params.field_weights,
                params.similarity,
                min_score,
                lim,
            );
        }
    }
    Ok(hits)
}

/// Searches a ZSL index reproducing the host application's Zend Lucene search adapter:
/// build_query -> executor -> if empty, all-fields fallback; filters min_score (`>=`),
/// limit==0 = unlimited.
pub fn search_index(
    index_dir: &Path,
    params: &QueryParams,
    min_score: f32,
    limit: usize,
) -> Result<Vec<Hit>, Box<dyn std::error::Error>> {
    let index = ZslIndex::open(index_dir)?;
    Ok(lexical_search(&index, params, min_score, limit)?)
}

/// Opens a ZSL index and runs a two-pass PRF (semantic) search. `limit == 0` = unlimited.
/// Degrades to a plain search internally when PRF cannot contribute (see `search_prf`).
///
/// In the active two-pass path the result is a RERANK, not a strict superset of a plain
/// `search_index` call: the boolean coord factor can reduce an original-only match's score
/// relative to plain search, so with a nonzero `min_score` or a binding `limit` this may
/// omit a hit that `search_index` would return. Only at `min_score == 0.0` and an
/// unlimited `limit` is the result guaranteed to be a superset of plain search.
pub fn search_prf_index(
    index_dir: &Path,
    params: &QueryParams,
    prf: &PrfParams,
    min_score: f32,
    limit: usize,
) -> Result<Vec<Hit>, Box<dyn std::error::Error>> {
    let index = ZslIndex::open(index_dir)?;
    Ok(search_prf(&index, params, prf, min_score, limit)?)
}

/// Opens a ZSL index once and runs a HYBRID search: the lexical retriever (`search_index`'s
/// logic, fallback included) and the semantic retriever (`search_prf`) are run as two
/// independent rankers and fused by Reciprocal Rank Fusion (`fuse_rrf`).
///
/// `min_score` filters each leg on its own native score scale BEFORE fusion. Each leg fetches
/// up to `hybrid.depth` candidates (0 = unlimited; raised to `limit` when a larger final
/// `limit` binds). `limit == 0` = unlimited final result. The returned `Hit.score` is the RRF
/// fused score (scale ~0.01-0.03 per matching leg), NOT comparable to a plain `search` score.
///
/// Because the lexical leg enters fusion in full, at `min_score == 0.0` and an unlimited
/// `limit` the result is a superset of a plain `search_index` call — this fixes the PRF wart
/// where a binding `min_score`/`limit` could drop a hit that plain search returned.
pub fn search_hybrid_index(
    index_dir: &Path,
    params: &QueryParams,
    prf: &PrfParams,
    hybrid: &HybridParams,
    min_score: f32,
    limit: usize,
) -> Result<Vec<Hit>, Box<dyn std::error::Error>> {
    let index = ZslIndex::open(index_dir)?;
    // Candidate depth per retriever: unlimited if either the pool or the final limit is
    // unlimited; otherwise the larger of the two so fusion has room to reorder.
    let depth = match (hybrid.depth, limit) {
        (0, _) | (_, 0) => 0,
        (d, l) => d.max(l),
    };
    let k = hybrid.k.max(1);
    let lexical = lexical_search(&index, params, min_score, depth)?;
    let semantic = search_prf(&index, params, prf, min_score, depth)?;
    Ok(fuse_rrf(&[lexical, semantic], k, limit))
}

/// Search-adapter fallback: an all-fields Boolean of terms (Should) over the UNIQUE
/// tokens of the text (dedup by text, like the re-parsed MultiTerm in the search adapter).
fn fallback_query(text: &str) -> Option<Query> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut clauses: Vec<(Occur, Query)> = Vec::new();
    for tok in analyze(text) {
        if seen.insert(tok.clone()) {
            clauses.push((
                Occur::Should,
                Query::Term {
                    field: None,
                    text: tok,
                },
            ));
        }
    }
    if clauses.is_empty() {
        None
    } else {
        Some(Query::Boolean { clauses })
    }
}

/// Resolves an id-field value to an internal doc id via an `InGroup` over
/// `<id_field>_key` (build_query adds the suffix), like the writer's resolver.
fn resolve_reference_doc(
    index: &ZslIndex,
    id_field: &str,
    id_value: &str,
) -> Result<Option<usize>, Box<dyn std::error::Error>> {
    let params = QueryParams {
        text: String::new(),
        where_groups: Vec::new(),
        in_groups: vec![InGroup {
            field: id_field.to_string(),
            values: vec![id_value.to_string()],
        }],
        fuzzy_similarity: 0.5,
        fuzzy_prefix_len: 3,
        wildcard_min_prefix: 0,
        accent_insensitive: false,
        field_weights: std::collections::HashMap::new(),
        similarity: crate::score::Similarity::Bm25,
    };
    let query = build_query(&params)?;
    let hits = search(index, &query, 0.0, 1);
    Ok(hits.first().map(|h| h.id))
}

/// Opens a ZSL index, resolves the reference id-field value to an internal doc id,
/// and runs a More Like This query. Returns an empty vec if the reference doc is not found.
pub fn more_like_this_index(
    index_dir: &Path,
    id_field: &str,
    id_value: &str,
    params: &MltParams,
) -> Result<Vec<Hit>, Box<dyn std::error::Error>> {
    let index = ZslIndex::open(index_dir)?;
    match resolve_reference_doc(&index, id_field, id_value)? {
        Some(doc_id) => Ok(more_like_this(&index, doc_id, params)),
        None => Ok(Vec::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::InGroup;
    use std::path::PathBuf;

    fn multiseg() -> PathBuf {
        PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/zsl_index_multiseg"
        ))
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
            field_weights: std::collections::HashMap::new(),
            similarity: crate::score::Similarity::Bm25,
        }
    }
    fn ids(hits: &[Hit]) -> Vec<usize> {
        let mut v: Vec<usize> = hits.iter().map(|h| h.id).collect();
        v.sort_unstable();
        v
    }

    #[test]
    fn search_prf_index_off_matches_plain() {
        // top_k = 0 (PRF disabled) must return exactly what search_index returns over the
        // same fixture — proves the runner wires PrfParams through and the plain path is intact.
        // ("vpn" is used, not "the": no title in this fixture contains "the", and
        // text_only_matches_across_segments below already proves "vpn" yields ids [0, 2].)
        use crate::prf::PrfParams;
        let dir = multiseg();
        let p = params("vpn");
        let off = PrfParams {
            top_k: 0,
            ..PrfParams::default()
        };
        let plain = search_index(&dir, &p, 0.0, 100).unwrap();
        let prf = search_prf_index(&dir, &p, &off, 0.0, 100).unwrap();
        let plain_ids: Vec<usize> = plain.iter().map(|h| h.id).collect();
        let prf_ids: Vec<usize> = prf.iter().map(|h| h.id).collect();
        assert_eq!(prf_ids, plain_ids);
    }

    #[test]
    fn search_prf_index_propagates_query_error() {
        // An invalid query (empty text) must propagate as an Err through the runner's
        // Box<dyn Error> boundary, not get swallowed into an empty Ok(vec![]) — mirrors
        // search_prf's own invalid_query_propagates_err test, one layer up the stack.
        use crate::prf::PrfParams;
        let dir = multiseg();
        let result = search_prf_index(&dir, &params(""), &PrfParams::default(), 0.0, 0);
        assert!(
            result.is_err(),
            "empty-text query must propagate an error through search_prf_index"
        );
    }

    #[test]
    fn search_prf_index_with_feedback_returns_hits_for_known_token() {
        // Real two-pass PRF (top_k>0, the default) over the multiseg fixture, driving
        // actual feedback-term harvesting through search_prf_index — the
        // search_prf_index_off_matches_plain test above only exercises the DISABLED
        // (top_k=0) path, which never invokes select_terms at all.
        //
        // "vpn" is known present (text_only_matches_across_segments proves plain search
        // yields ids [0,2]; the fixture's stored titles are "alpha vpn guide" (id 0) and
        // "gamma vpn tutorial" (id 2), so pass 1 harvests real feedback terms from them,
        // e.g. "alpha"/"guide"/"gamma"/"tutorial").
        //
        // At min_score=0.0 and an unlimited limit (limit=0), search_prf's doc comment
        // guarantees the augmented Should-union can only ever ADD matches relative to
        // plain search, never drop one (nothing is filtered by score or truncated by
        // limit) — so plain's ids must be a subset of PRF's ids.
        let dir = multiseg();
        let p = params("vpn");
        let plain = search_index(&dir, &p, 0.0, 0).unwrap();
        let prf = search_prf_index(&dir, &p, &PrfParams::default(), 0.0, 0).unwrap();

        assert!(
            !prf.is_empty(),
            "PRF must return hits for a known-present token: {:?}",
            ids(&prf)
        );
        let plain_ids: HashSet<usize> = plain.iter().map(|h| h.id).collect();
        let prf_ids: HashSet<usize> = prf.iter().map(|h| h.id).collect();
        assert!(
            plain_ids.is_subset(&prf_ids),
            "at min_score=0/unlimited limit, PRF must be a superset of plain: plain={plain_ids:?} prf={prf_ids:?}"
        );
    }

    #[test]
    fn text_only_matches_across_segments() {
        // "vpn" crosses segments -> [0,2] (same doc-set as the text-only boolean oracle).
        let hits = search_index(&multiseg(), &params("vpn"), 0.0, 0).unwrap();
        assert_eq!(ids(&hits), vec![0, 2]);
    }

    #[test]
    fn empty_primary_triggers_fallback() {
        // text "vpn" (Must) + in cat=999 (Must, no doc) => empty primary;
        // the all-fields fallback "vpn" recovers [0,2].
        let mut p = params("vpn");
        p.in_groups = vec![InGroup {
            field: "cat".into(),
            values: vec!["999".into()],
        }];
        let hits = search_index(&multiseg(), &p, 0.0, 0).unwrap();
        assert_eq!(ids(&hits), vec![0, 2]);
    }

    #[test]
    fn limit_zero_is_unlimited() {
        // "how" matches the two "how to ..." docs even with limit=0.
        let hits = search_index(&multiseg(), &params("how"), 0.0, 0).unwrap();
        assert!(hits.len() >= 2, "limit=0 must return all matches");
    }

    use crate::mlt::MltParams;
    use crate::zsl::writer::{IndexWriter, WriterDoc, WriterField, WriterOpts};
    use std::collections::HashMap as StdHashMap;

    fn mlt_params(fields: &[&str]) -> MltParams {
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
            min_should_match: None,
            field_weights: StdHashMap::new(),
            size: 10,
            min_score: 0.0,
        }
    }

    // The writer only appends to an EXISTING index, so bootstrap by copying the KB
    // fixture to a temp dir, then append 3 docs carrying an `id_key` keyword and a
    // `body` text field that shares a rare term ("zebra" in A and B only). The KB
    // docs use a `title` field, so they never collide with our `body` postings.
    fn temp_index_with_mlt_docs() -> PathBuf {
        let src = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/zsl_index_kb"
        ));
        let dir = std::env::temp_dir().join(format!("sdsearch_mlt_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for entry in std::fs::read_dir(&src).unwrap() {
            let p = entry.unwrap().path();
            if p.is_file() {
                std::fs::copy(&p, dir.join(p.file_name().unwrap())).unwrap();
            }
        }
        let mut w = IndexWriter::open(&dir, WriterOpts::default()).unwrap();
        for (id, body) in [
            ("A", "zebra alpha"),
            ("B", "zebra beta"),
            ("C", "cat gamma"),
        ] {
            w.add_document(WriterDoc {
                fields: vec![
                    WriterField::keyword("id_key", id),
                    WriterField::text("body", body),
                ],
            })
            .unwrap();
        }
        w.commit().unwrap();
        dir
    }

    #[test]
    fn more_like_this_index_finds_similar_and_excludes_source() {
        let dir = temp_index_with_mlt_docs();

        // reference doc "A" (resolved via id -> id_key) shares "zebra" with "B" only.
        let hits = more_like_this_index(&dir, "id", "A", &mlt_params(&["body"])).unwrap();
        let ids: Vec<String> = hits
            .iter()
            .filter_map(|h| h.fields.get("id_key").cloned())
            .collect();
        assert!(ids.contains(&"B".to_string()), "expected B among {ids:?}");
        assert!(
            !ids.contains(&"A".to_string()),
            "source A must be excluded: {ids:?}"
        );

        // unknown reference id -> empty, not an error.
        let none = more_like_this_index(&dir, "id", "ZZZ", &mlt_params(&["body"])).unwrap();
        assert!(none.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn search_hybrid_index_is_superset_of_lexical() {
        // At min_score=0 / unlimited, the lexical leg enters fusion in full, so every id a
        // plain search returns must appear in the fused output (the PRF wart-fix).
        use crate::hybrid::HybridParams;
        use crate::prf::PrfParams;
        let dir = multiseg();
        let p = params("vpn");
        let lexical = search_index(&dir, &p, 0.0, 0).unwrap();
        let hybrid = search_hybrid_index(
            &dir,
            &p,
            &PrfParams::default(),
            &HybridParams::default(),
            0.0,
            0,
        )
        .unwrap();
        assert!(
            !hybrid.is_empty(),
            "hybrid must return hits for a known token"
        );
        let lex_ids: HashSet<usize> = lexical.iter().map(|h| h.id).collect();
        let hyb_ids: HashSet<usize> = hybrid.iter().map(|h| h.id).collect();
        assert!(
            lex_ids.is_subset(&hyb_ids),
            "hybrid must represent every lexical hit: lex={lex_ids:?} hyb={hyb_ids:?}"
        );
    }

    #[test]
    fn search_hybrid_index_propagates_query_error() {
        // Empty-text query is invalid and must propagate as Err, not an empty Ok.
        use crate::hybrid::HybridParams;
        use crate::prf::PrfParams;
        let dir = multiseg();
        let r = search_hybrid_index(
            &dir,
            &params(""),
            &PrfParams::default(),
            &HybridParams::default(),
            0.0,
            0,
        );
        assert!(r.is_err(), "empty-text query must propagate an error");
    }

    #[test]
    fn search_hybrid_index_limit_truncates() {
        use crate::hybrid::HybridParams;
        use crate::prf::PrfParams;
        let dir = multiseg();
        let full = search_hybrid_index(
            &dir,
            &params("vpn"),
            &PrfParams::default(),
            &HybridParams::default(),
            0.0,
            0,
        )
        .unwrap();
        assert!(
            full.len() >= 2,
            "expected >=2 fused hits for vpn: {:?}",
            ids(&full)
        );
        let limited = search_hybrid_index(
            &dir,
            &params("vpn"),
            &PrfParams::default(),
            &HybridParams::default(),
            0.0,
            1,
        )
        .unwrap();
        assert_eq!(limited.len(), 1, "limit=1 must truncate the fused list");
    }

    #[test]
    fn search_hybrid_index_min_score_filters_each_leg() {
        // Scores are normalized to <=1.0 per leg; a min_score above that filters BOTH legs
        // before fusion => empty fused result. Proves min_score is applied inside the legs
        // (fusion itself has no min_score knob).
        use crate::hybrid::HybridParams;
        use crate::prf::PrfParams;
        let dir = multiseg();
        let r = search_hybrid_index(
            &dir,
            &params("vpn"),
            &PrfParams::default(),
            &HybridParams::default(),
            2.0,
            0,
        )
        .unwrap();
        assert!(
            r.is_empty(),
            "min_score above normalized max must empty both legs: {:?}",
            ids(&r)
        );
    }

    #[test]
    fn search_hybrid_index_prf_off_equals_lexical_ids() {
        // With PRF disabled (top_k=0) the semantic leg returns the plain base query, so both
        // legs cover the same docs; the fused id-set must equal the lexical id-set.
        use crate::hybrid::HybridParams;
        use crate::prf::PrfParams;
        let dir = multiseg();
        let p = params("vpn");
        let off = PrfParams {
            top_k: 0,
            ..PrfParams::default()
        };
        let lexical = search_index(&dir, &p, 0.0, 0).unwrap();
        let hybrid = search_hybrid_index(&dir, &p, &off, &HybridParams::default(), 0.0, 0).unwrap();
        assert_eq!(
            ids(&hybrid),
            ids(&lexical),
            "PRF-off fused id-set must equal lexical id-set"
        );
    }
}
