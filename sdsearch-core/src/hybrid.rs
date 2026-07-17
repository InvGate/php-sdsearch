//! Reciprocal Rank Fusion (RRF): fuse several independently-ranked hit lists into one
//! ranking by summing `1/(k+rank)` across the lists a doc appears in. Rank-based, so the
//! retrievers' raw score scales never need to be comparable. Pure and index-free.

use crate::search::Hit;
use std::collections::HashMap;

/// Tuning for hybrid RRF fusion. `k` damps the weight of top ranks (canonical default 60);
/// `depth` is the candidate pool fetched per retriever before fusion (0 = unlimited).
#[derive(Debug, Clone)]
pub struct HybridParams {
    pub k: usize,
    pub depth: usize,
}

impl Default for HybridParams {
    fn default() -> Self {
        Self { k: 60, depth: 100 }
    }
}

/// Fuses ranked hit lists by Reciprocal Rank Fusion. For each list, a hit at 1-based
/// position `rank` contributes `1.0 / (k + rank)` to its doc's fused score. Docs are sorted
/// by fused score descending, ties broken by ascending `id` (deterministic). Each returned
/// `Hit` carries its fused score and the `fields` from the first list it appeared in.
/// `limit == 0` returns the full fused list.
pub fn fuse_rrf(rankings: &[Vec<Hit>], k: usize, limit: usize) -> Vec<Hit> {
    let mut scores: HashMap<usize, f32> = HashMap::new();
    let mut repr: HashMap<usize, Hit> = HashMap::new();
    for list in rankings {
        debug_assert!(
            {
                let mut seen = std::collections::HashSet::new();
                list.iter().all(|h| seen.insert(h.id))
            },
            "fuse_rrf: each ranking list must have unique doc ids (got a duplicate)"
        );
        for (i, hit) in list.iter().enumerate() {
            let rank = i + 1;
            let contribution = 1.0 / (k as f32 + rank as f32);
            *scores.entry(hit.id).or_insert(0.0) += contribution;
            repr.entry(hit.id).or_insert_with(|| hit.clone());
        }
    }
    let mut fused: Vec<Hit> = repr
        .into_values()
        .map(|mut hit| {
            hit.score = scores[&hit.id];
            hit
        })
        .collect();
    fused.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.id.cmp(&b.id)));
    if limit != 0 && fused.len() > limit {
        fused.truncate(limit);
    }
    fused
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(id: usize) -> Hit {
        Hit {
            id,
            score: 0.0,
            fields: HashMap::new(),
        }
    }

    fn ids(hits: &[Hit]) -> Vec<usize> {
        hits.iter().map(|h| h.id).collect()
    }

    #[test]
    fn doc_in_both_lists_beats_doc_in_one() {
        // list A ranks: 1->r1, 2->r2, 3->r3 ; list B ranks: 2->r1, 4->r2. k=60.
        // scores: id2 = 1/62 + 1/61 (highest), id1 = 1/61, id4 = 1/62, id3 = 1/63.
        // => order [2, 1, 4, 3].
        let a = vec![hit(1), hit(2), hit(3)];
        let b = vec![hit(2), hit(4)];
        let fused = fuse_rrf(&[a, b], 60, 0);
        assert_eq!(ids(&fused), vec![2, 1, 4, 3]);
    }

    #[test]
    fn exact_rrf_arithmetic() {
        // single list, single doc at rank 1, k=1 => score = 1/(1+1) = 0.5.
        let fused = fuse_rrf(&[vec![hit(7)]], 1, 0);
        assert_eq!(fused.len(), 1);
        assert!(
            (fused[0].score - 0.5).abs() < 1e-6,
            "score was {}",
            fused[0].score
        );
    }

    #[test]
    fn ties_break_by_ascending_id() {
        // id5 and id3 each appear once at rank 1 in separate lists => equal fused score.
        // deterministic tie-break => id3 before id5.
        let fused = fuse_rrf(&[vec![hit(5)], vec![hit(3)]], 60, 0);
        assert_eq!(ids(&fused), vec![3, 5]);
    }

    #[test]
    fn limit_truncates_and_zero_is_unlimited() {
        let a = vec![hit(1), hit(2), hit(3)];
        let b = vec![hit(2), hit(4)];
        let full = fuse_rrf(&[a.clone(), b.clone()], 60, 0);
        assert_eq!(full.len(), 4);
        let limited = fuse_rrf(&[a, b], 60, 2);
        assert_eq!(ids(&limited), vec![2, 1]); // top 2 of [2,1,4,3]
    }

    #[test]
    fn empty_lists_are_handled() {
        // one empty list: returns the other, RRF-ranked by position (1 before 2).
        let one = fuse_rrf(&[vec![hit(1), hit(2)], vec![]], 60, 0);
        assert_eq!(ids(&one), vec![1, 2]);
        // both empty: empty result.
        let none = fuse_rrf(&[Vec::new(), Vec::new()], 60, 0);
        assert!(none.is_empty());
    }

    #[test]
    #[should_panic(expected = "unique doc ids")]
    fn duplicate_id_in_one_list_panics_debug_assert() {
        // a single list with a duplicate id violates fuse_rrf's per-list uniqueness
        // precondition; the debug_assert should fire (only active with debug_assertions,
        // which cargo test enables by default).
        let dup = vec![hit(1), hit(1)];
        fuse_rrf(&[dup], 60, 0);
    }

    #[test]
    fn fields_come_from_first_occurrence() {
        let mut h = hit(1);
        h.fields.insert("title".to_string(), "hello".to_string());
        let fused = fuse_rrf(&[vec![h], vec![hit(1)]], 60, 0);
        assert_eq!(
            fused[0].fields.get("title").map(String::as_str),
            Some("hello")
        );
    }
}
