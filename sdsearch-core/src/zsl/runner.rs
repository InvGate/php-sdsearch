//! runner: orchestrates ZslIndex + build_query + executor, reproducing the host
//! application's Zend Lucene search adapter (min_score filtering, limit==0 = unlimited).

use crate::hybrid::{HybridParams, fuse_rrf};
use crate::index::IndexReader;
use crate::mlt::{MltParams, more_like_this};
use crate::prf::{PrfParams, search_prf};
use crate::query::{
    InGroup, QueryError, QueryParams, allow_list, build_query, search, search_with_weights_paged,
};
use crate::search::{Hit, SearchOutcome, SortSpec};
use crate::zsl::index::ZslIndex;
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
    let restrict = allow_list(index, params);
    Ok(search_with_weights_paged(
        index,
        &query,
        &params.field_weights,
        params.similarity,
        min_score,
        0,
        lim,
        None,
        restrict.as_ref(),
        None,
    )
    .hits)
}

/// Searches a ZSL index reproducing the host application's Zend Lucene search adapter:
/// build_query -> executor; filters min_score (`>=`), limit==0 = unlimited.
///
/// The legacy adapter had an empty-result fallback that re-parsed the query string. We do
/// NOT reproduce it: it kept the `where`/`in` filters required (an excluding filter is never
/// bypassed — see the oracle), and the only text it relaxed was already a subset of what
/// `text_subquery` matches in the primary pass. So the fallback could only ever return an
/// empty set here, while a naive text-only fallback would silently leak documents past an
/// excluding filter — which we must never do.
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
/// logic) and the semantic retriever (`search_prf`) are run as two independent rankers and
/// fused by Reciprocal Rank Fusion (`fuse_rrf`).
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

/// Paged variant of `search_index`: returns the page `[offset, offset+limit)` plus the
/// (optionally capped) total match count. `limit == 0` = unlimited, as in `search_index`.
/// `total_cap`: `None` = exact count; `Some(cap)` = saturated at `cap`.
pub fn search_index_paged(
    index_dir: &Path,
    params: &QueryParams,
    min_score: f32,
    offset: usize,
    limit: usize,
    total_cap: Option<usize>,
) -> Result<SearchOutcome, Box<dyn std::error::Error>> {
    let index = ZslIndex::open(index_dir)?;
    let query = build_query(params)?;
    reject_unknown_sort_field(&index, params.sort.as_ref())?;
    let lim = if limit == 0 { usize::MAX } else { limit };
    let restrict = allow_list(&index, params);
    Ok(search_with_weights_paged(
        &index,
        &query,
        &params.field_weights,
        params.similarity,
        min_score,
        offset,
        lim,
        total_cap,
        restrict.as_ref(),
        params.sort.as_ref(),
    ))
}

/// Fails a sort by a field this index does not have.
///
/// Without it a misspelled field name is not an error anywhere: every doc resolves to `Missing`,
/// they all tie, and the results come back in the tiebreak order (score desc, id asc) — which
/// reads as "sorting is broken" rather than "that field does not exist". `has_field` is a scan
/// over the `.fnm` names already in memory, run once per query, so the check is free relative to
/// the query it guards.
///
/// An index with no documents is exempt: it has no segments and therefore no field names, so
/// validating would turn every sorted query over an empty index into an error instead of the
/// empty result it should be.
fn reject_unknown_sort_field(
    index: &ZslIndex,
    sort: Option<&SortSpec>,
) -> Result<(), Box<dyn std::error::Error>> {
    let Some(spec) = sort else { return Ok(()) };
    if index.num_docs() == 0 || index.has_field(&spec.field) {
        return Ok(());
    }
    Err(format!(
        "unknown sort field {:?} (the index has no such field; the name is used verbatim, so pass the `_key` name)",
        spec.field
    )
    .into())
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
        synonyms: false,
        field_weights: std::collections::HashMap::new(),
        similarity: crate::score::Similarity::Bm25,
        range_filters: Vec::new(),
        match_all: Vec::new(),
        sort: None,
        boolean_tree: None,
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
    use std::collections::HashSet;
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
            synonyms: false,
            field_weights: std::collections::HashMap::new(),
            similarity: crate::score::Similarity::Bm25,
            range_filters: vec![],
            match_all: vec![],
            sort: None,
            boolean_tree: None,
        }
    }
    fn ids(hits: &[Hit]) -> Vec<usize> {
        let mut v: Vec<usize> = hits.iter().map(|h| h.id).collect();
        v.sort_unstable();
        v
    }

    // Bootstrap from the KB fixture (the writer only appends) and add two docs carrying a
    // `cat_key` keyword filter field and a `body` text field: cat 1 -> "alpha", cat 2 ->
    // "zebra". So "zebra" exists ONLY outside cat 1 -> a cat=1 filter must exclude it.
    fn temp_index_with_cat_docs(tag: &str) -> PathBuf {
        let src = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/zsl_index_kb"
        ));
        let dir =
            std::env::temp_dir().join(format!("sdsearch_catfilter_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for entry in std::fs::read_dir(&src).unwrap() {
            let p = entry.unwrap().path();
            if p.is_file() {
                std::fs::copy(&p, dir.join(p.file_name().unwrap())).unwrap();
            }
        }
        let mut w = IndexWriter::open(&dir, WriterOpts::default()).unwrap();
        for (cat, body) in [("1", "alpha"), ("2", "zebra")] {
            w.add_document(WriterDoc {
                fields: vec![
                    WriterField::keyword("cat_key", cat),
                    WriterField::text("body", body),
                ],
            })
            .unwrap();
        }
        w.commit().unwrap();
        dir
    }

    // Bootstrap from the KB fixture and add a doc whose body carries a colon-bearing token.
    // The analyzer keeps ':' inside a token, so "c:drive" is a SINGLE indexed term.
    fn temp_index_with_punct_doc(tag: &str) -> PathBuf {
        let src = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/zsl_index_kb"
        ));
        let dir = std::env::temp_dir().join(format!("sdsearch_punct_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for entry in std::fs::read_dir(&src).unwrap() {
            let p = entry.unwrap().path();
            if p.is_file() {
                std::fs::copy(&p, dir.join(p.file_name().unwrap())).unwrap();
            }
        }
        let mut w = IndexWriter::open(&dir, WriterOpts::default()).unwrap();
        w.add_document(WriterDoc {
            fields: vec![WriterField::text("body", "c:drive restore")],
        })
        .unwrap();
        w.commit().unwrap();
        dir
    }

    #[test]
    fn prefix_reaches_a_colon_bearing_token() {
        // Dropping the query-operator escaping lets a prefix like "c:dr" reach the indexed
        // token "c:drive" through the wildcard leaf. The old escaping (`c\:dr*`) suppressed
        // this because no indexed term carries a backslash.
        let dir = temp_index_with_punct_doc("colon");
        let hits = search_index(&dir, &params("c:dr"), 0.0, 0).unwrap();
        assert!(
            !hits.is_empty(),
            "prefix 'c:dr' should reach the 'c:drive' token via the wildcard leaf"
        );
        let _ = std::fs::remove_dir_all(&dir);
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

    /// A where group with no field name: the query `build_query` rejects.
    fn invalid_params() -> QueryParams {
        let mut p = params("vpn");
        p.where_groups = vec![crate::query::WhereGroup {
            field: String::new(),
            values: vec!["x".into()],
            occur: crate::query::Occur::Should,
        }];
        p
    }

    #[test]
    fn search_prf_index_propagates_query_error() {
        // An invalid query must propagate as an Err through the runner's Box<dyn Error>
        // boundary, not get swallowed into an empty Ok(vec![]) — mirrors search_prf's own
        // invalid_query_propagates_err test, one layer up the stack.
        use crate::prf::PrfParams;
        let dir = multiseg();
        let result = search_prf_index(&dir, &invalid_params(), &PrfParams::default(), 0.0, 0);
        assert!(
            result.is_err(),
            "an invalid query must propagate an error through search_prf_index"
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
    fn empty_primary_with_excluding_filter_stays_empty() {
        // text "vpn" (Must) + in cat=999 (Must, no doc) => empty primary. An excluding
        // filter must NOT be relaxed: the result stays empty rather than leaking the
        // text-only "vpn" matches [0,2] (which is what the removed fallback used to do).
        let mut p = params("vpn");
        p.in_groups = vec![InGroup {
            field: "cat".into(),
            values: vec!["999".into()],
        }];
        let hits = search_index(&multiseg(), &p, 0.0, 0).unwrap();
        assert!(
            hits.is_empty(),
            "excluding filter must not be bypassed: {:?}",
            ids(&hits)
        );
    }

    #[test]
    fn empty_primary_never_bypasses_an_in_filter() {
        // "zebra" lives ONLY in the cat=2 doc; a cat=1 filter must exclude it.
        // The empty-primary path must NOT relax the filter and leak the cat=2 doc
        // (parity with the legacy Zend adapter, whose fallback keeps the filter required).
        let dir = temp_index_with_cat_docs("infilter");
        let mut p = params("zebra");
        p.in_groups = vec![InGroup {
            field: "cat".into(),
            values: vec!["1".into()],
        }];
        let hits = search_index(&dir, &p, 0.0, 0).unwrap();
        assert!(
            hits.is_empty(),
            "cat=1 filter must exclude the cat=2 'zebra' doc, got {} hit(s)",
            hits.len()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn text_matches_the_cat_doc_when_unfiltered() {
        // Guard for the test above: proves "zebra" really does match a doc, so the
        // empty result there comes from the filter, not from the term being absent.
        let dir = temp_index_with_cat_docs("unfiltered");
        let hits = search_index(&dir, &params("zebra"), 0.0, 0).unwrap();
        assert!(
            !hits.is_empty(),
            "zebra should match the cat=2 doc unfiltered"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn limit_zero_is_unlimited() {
        // "how" matches the two "how to ..." docs even with limit=0.
        let hits = search_index(&multiseg(), &params("how"), 0.0, 0).unwrap();
        assert!(hits.len() >= 2, "limit=0 must return all matches");
    }

    #[test]
    fn search_index_paged_reports_total_and_offset() {
        // "how" matches the two "how to ..." docs in the multiseg fixture.
        let full = search_index_paged(&multiseg(), &params("how"), 0.0, 0, 0, None).unwrap();
        assert!(
            full.total >= 2,
            "total counts all matches, got {}",
            full.total
        );
        assert!(!full.total_capped);
        let full_ids = ids(&full.hits);

        // offset 1 with a large limit drops exactly the first hit of the ranking.
        let paged = search_index_paged(&multiseg(), &params("how"), 0.0, 1, 100, None).unwrap();
        assert_eq!(paged.hits.len(), full.hits.len() - 1);
        assert_eq!(paged.total, full.total, "total is independent of the page");

        // a cap of 1 saturates the total and flags it, without changing the page size.
        let capped = search_index_paged(&multiseg(), &params("how"), 0.0, 0, 100, Some(1)).unwrap();
        assert_eq!(capped.total, 1);
        assert!(capped.total_capped);
        assert_eq!(ids(&capped.hits), full_ids, "cap bounds total, not hits");
    }

    // Live docs of the multiseg fixture: doc 3 ("delta backup notes") is deleted by `_1_1.del`.
    const MULTISEG_LIVE: [usize; 5] = [0, 1, 2, 4, 5];

    fn in_only(field: &str, values: &[&str]) -> QueryParams {
        let mut p = params("");
        p.in_groups = vec![InGroup {
            field: field.into(),
            values: values.iter().map(|v| (*v).to_string()).collect(),
        }];
        p
    }

    #[test]
    fn search_index_paged_with_nothing_to_match_returns_every_live_doc() {
        // No text, tree, where or in is "everything", as OpenSearch answers an empty text.
        // It used to throw "empty query", which the host adapter turned into zero rows.
        let out = search_index_paged(&multiseg(), &params(""), 0.0, 0, 0, None)
            .expect("an empty query is not an error on the paged path");
        assert_eq!(ids(&out.hits), MULTISEG_LIVE, "the deleted doc stays out");
        assert_eq!(out.total, MULTISEG_LIVE.len());
        let flat = search_index(&multiseg(), &params(""), 0.0, 0).expect("nor on the flat one");
        assert_eq!(ids(&flat), MULTISEG_LIVE);
    }

    #[test]
    fn search_index_paged_with_only_a_range_narrows_every_live_doc() {
        // The "only a date filter" screen: `range` is an allow-list, not a clause, so it never
        // counted as something to match.
        let mut p = params("");
        p.range_filters = cat_1_range();
        let expected = search_index(&multiseg(), &in_only("cat", &["1"]), 0.0, 0).unwrap();
        assert!(
            !expected.is_empty() && expected.len() < MULTISEG_LIVE.len(),
            "fixture sanity: the range must keep some live docs and drop others"
        );

        let out = search_index_paged(&multiseg(), &p, 0.0, 0, 0, None)
            .expect("a range-only query is not an error on the paged path");
        assert_eq!(ids(&out.hits), ids(&expected));
    }

    #[test]
    fn search_index_paged_with_only_a_mustnot_where_returns_everything_but_the_excluded() {
        // A lone `mustnot` used to have nothing to subtract from and returned zero hits.
        let mut p = params("");
        p.where_groups = mustnot_lang_en();
        let english: HashSet<usize> = search_index(&multiseg(), &in_only("lang", &["en"]), 0.0, 0)
            .unwrap()
            .iter()
            .map(|h| h.id)
            .collect();
        assert!(!english.is_empty(), "fixture sanity: some doc is lang=en");
        let expected: Vec<usize> = MULTISEG_LIVE
            .into_iter()
            .filter(|d| !english.contains(d))
            .collect();

        let out = search_index_paged(&multiseg(), &p, 0.0, 0, 0, None).unwrap();
        assert_eq!(ids(&out.hits), expected);
    }

    fn cat_1_range() -> Vec<crate::query::RangeFilter> {
        vec![crate::query::RangeFilter {
            field: "cat_key".into(),
            lower: Some("1".into()),
            upper: Some("1".into()),
        }]
    }

    fn mustnot_lang_en() -> Vec<crate::query::WhereGroup> {
        vec![crate::query::WhereGroup {
            field: "lang".into(),
            values: vec!["en".into()],
            occur: crate::query::Occur::MustNot,
        }]
    }

    #[test]
    fn every_retriever_applies_the_range_like_the_paged_one() {
        // `search_index` and the semantic/hybrid paths used to skip the `range`/`match_all`
        // allow-list: with text, a date filter was silently dropped; with only a `mustnot`, the
        // `MatchAll` base leaked every live doc past it.
        let allowed: HashSet<usize> =
            ids(&search_index(&multiseg(), &in_only("cat", &["1"]), 0.0, 0).unwrap())
                .into_iter()
                .collect();
        let unfiltered = ids(&search_index(&multiseg(), &params("how"), 0.0, 0).unwrap());
        assert!(
            unfiltered.iter().any(|d| !allowed.contains(d)),
            "fixture sanity: the range must drop a text match"
        );

        let mut text = params("how");
        text.range_filters = cat_1_range();
        let mut mustnot_only = params("");
        mustnot_only.range_filters = cat_1_range();
        mustnot_only.where_groups = mustnot_lang_en();

        for p in [text, mustnot_only] {
            let paged = search_index_paged(&multiseg(), &p, 0.0, 0, 0, None).unwrap();
            let flat = search_index(&multiseg(), &p, 0.0, 0).unwrap();
            assert_eq!(ids(&flat), ids(&paged.hits), "text={:?}", p.text);

            let prf = PrfParams::default();
            let semantic = search_prf_index(&multiseg(), &p, &prf, 0.0, 0).unwrap();
            let hybrid =
                search_hybrid_index(&multiseg(), &p, &prf, &HybridParams::default(), 0.0, 0)
                    .unwrap();
            for (name, hits) in [("semantic", semantic), ("hybrid", hybrid)] {
                assert!(
                    ids(&hits).iter().all(|d| allowed.contains(d)),
                    "{name} leaked past the range (text={:?}): {:?}",
                    p.text,
                    ids(&hits)
                );
            }
        }
    }

    #[test]
    fn match_all_under_a_restrict_drops_deleted_and_unknown_ids() {
        // `restrict` is caller input on the pub paged API: `MatchAll` must narrow it to live
        // docs, not echo it back. Doc 3 is deleted in this fixture; 99 does not exist.
        let index = ZslIndex::open(&multiseg()).unwrap();
        let allow: HashSet<usize> = [0, 3, 99].into();
        let out = search_with_weights_paged(
            &index,
            &build_query(&params("")).unwrap(),
            &std::collections::HashMap::new(),
            crate::score::Similarity::Bm25,
            0.0,
            0,
            usize::MAX,
            None,
            Some(&allow),
            None,
        );
        assert_eq!(ids(&out.hits), vec![0]);
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
        // An invalid query must propagate as Err, not an empty Ok.
        use crate::hybrid::HybridParams;
        use crate::prf::PrfParams;
        let dir = multiseg();
        let r = search_hybrid_index(
            &dir,
            &invalid_params(),
            &PrfParams::default(),
            &HybridParams::default(),
            0.0,
            0,
        );
        assert!(r.is_err(), "an invalid query must propagate an error");
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
