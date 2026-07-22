//! ZslIndex: aggregates N ZslSegment behind the IndexReader trait (global ids + unified stats).

use crate::index::IndexReader;
use crate::zsl::segment::ZslSegment;
use crate::zsl::segments::read_segment_infos;
use std::collections::HashMap;
use std::path::Path;

struct Entry {
    base: usize,    // global doc id of the segment's first doc
    max_doc: usize, // size of the id space (incl. deletes)
    seg: ZslSegment,
}

pub struct ZslIndex {
    entries: Vec<Entry>,
    /// total live docs, precomputed (num_docs() is O(1), called per doc in the scorer).
    num_docs: usize,
    /// total docs incl. deletes (Σ maxDoc), precomputed — the idf denominator (ZSL parity).
    total_docs: usize,
}

impl ZslIndex {
    pub fn open(index_dir: &Path) -> std::io::Result<ZslIndex> {
        let infos = read_segment_infos(index_dir)?;
        let mut entries = Vec::with_capacity(infos.len());
        let mut base = 0usize;
        for info in infos {
            let seg = ZslSegment::open_named(index_dir, &info.name, info.del_gen)?;
            let max_doc = seg.max_doc();
            entries.push(Entry { base, max_doc, seg });
            base += max_doc;
        }
        let num_docs = entries.iter().map(|e| e.seg.num_docs()).sum();
        let total_docs = entries.iter().map(|e| e.max_doc).sum();
        Ok(ZslIndex {
            entries,
            num_docs,
            total_docs,
        })
    }

    /// (entry, local id) that owns the global id, or None if out of range.
    fn locate(&self, global_id: usize) -> Option<(&Entry, usize)> {
        self.entries
            .iter()
            .find(|e| global_id >= e.base && global_id < e.base + e.max_doc)
            .map(|e| (e, global_id - e.base))
    }
}

impl IndexReader for ZslIndex {
    fn num_docs(&self) -> usize {
        self.num_docs
    }

    fn total_docs(&self) -> usize {
        self.total_docs
    }

    fn doc_freq(&self, field: &str, term: &str) -> usize {
        self.entries
            .iter()
            .map(|e| e.seg.doc_freq(field, term))
            .sum()
    }

    fn postings_for(&self, field: &str, term: &str) -> Vec<(usize, u32)> {
        let mut out = Vec::new();
        for e in &self.entries {
            for (local, tf) in e.seg.postings_for(field, term) {
                out.push((e.base + local, tf));
            }
        }
        out
    }

    fn field_len(&self, doc_id: usize, field: &str) -> u32 {
        match self.locate(doc_id) {
            Some((e, local)) => e.seg.field_len(local, field),
            None => 1,
        }
    }

    fn avg_field_len(&self, field: &str) -> f32 {
        // weighted mean: each segment's avg * its doc count, over the total doc count.
        // (seg.avg_field_len is over the segment's max_doc, so weight by max_doc.)
        let mut total_len = 0.0f64;
        let mut total_docs = 0usize;
        for e in &self.entries {
            total_len += f64::from(e.seg.avg_field_len(field)) * e.max_doc as f64;
            total_docs += e.max_doc;
        }
        if total_docs == 0 || total_len == 0.0 {
            1.0
        } else {
            (total_len / total_docs as f64) as f32
        }
    }

    fn stored_fields(&self, doc_id: usize) -> HashMap<String, String> {
        match self.locate(doc_id) {
            Some((e, local)) => e.seg.stored_fields(local),
            None => HashMap::new(),
        }
    }

    fn terms_with_prefix(&self, field: &str, prefix: &str) -> Vec<String> {
        let mut set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for e in &self.entries {
            set.extend(e.seg.terms_with_prefix(field, prefix));
        }
        set.into_iter().collect()
    }

    fn terms_with_prefix_limited(&self, field: &str, prefix: &str, limit: usize) -> Vec<String> {
        // Each segment yields its lexicographic-first `limit`; their union contains the global
        // first `limit` (a global i-th term, i<limit, is ≤ the i-th in its own segment), so the
        // merged+sorted+truncated result is the correct global first `limit`.
        let mut set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for e in &self.entries {
            set.extend(e.seg.terms_with_prefix_limited(field, prefix, limit));
        }
        set.into_iter().take(limit).collect()
    }

    fn terms_in_range(&self, field: &str, lower: Option<&str>, upper: Option<&str>) -> Vec<String> {
        let mut set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for e in &self.entries {
            set.extend(e.seg.terms_in_range(field, lower, upper));
        }
        set.into_iter().collect()
    }

    fn positions_for(&self, field: &str, term: &str, doc_id: usize) -> Vec<u32> {
        match self.locate(doc_id) {
            Some((e, local)) => e.seg.positions_for(field, term, local),
            None => Vec::new(),
        }
    }

    fn positions_all(&self, field: &str, term: &str) -> HashMap<usize, Vec<u32>> {
        let mut out = HashMap::new();
        for e in &self.entries {
            for (local, positions) in e.seg.positions_all(field, term) {
                out.insert(e.base + local, positions);
            }
        }
        out
    }

    fn indexed_fields(&self) -> Vec<String> {
        let mut set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for e in &self.entries {
            set.extend(e.seg.indexed_fields());
        }
        set.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::IndexReader;
    use crate::zsl::segment::ZslSegment;
    use std::path::PathBuf;

    fn kb() -> PathBuf {
        PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/zsl_index_kb"
        ))
    }

    #[test]
    fn single_segment_index_equals_segment() {
        let idx = ZslIndex::open(&kb()).unwrap();
        let seg = ZslSegment::open(&kb()).unwrap();
        assert_eq!(idx.num_docs(), seg.num_docs());
        assert_eq!(idx.doc_freq("title", "vpn"), seg.doc_freq("title", "vpn"));
        assert_eq!(
            idx.postings_for("title", "vpn"),
            seg.postings_for("title", "vpn")
        );
        // stored routing: the same doc returns the same id_key
        let d0 = idx.postings_for("title", "vpn")[0].0;
        assert_eq!(idx.stored_fields(d0), seg.stored_fields(d0));
    }

    #[test]
    fn zsl_index_avg_field_len_positive_and_matches_manual() {
        let idx = ZslIndex::open(&kb()).unwrap();
        let avg = idx.avg_field_len("title");
        assert!(avg >= 1.0, "avg must be at least 1.0, got {avg}");

        // manual average over every live+deleted doc's field_len must match the reader.
        let n = idx.total_docs();
        let manual: f64 = (0..n)
            .map(|d| f64::from(idx.field_len(d, "title")))
            .sum::<f64>()
            / n as f64;
        assert!(
            (f64::from(avg) - manual).abs() < 1e-3,
            "avg={avg} manual={manual}"
        );
    }

    #[test]
    fn zsl_index_avg_field_len_aggregates_across_segments() {
        let dir = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/zsl_index_multiseg"
        ));
        let idx = ZslIndex::open(&dir).unwrap();
        // must be multi-segment, or the weighted-mean path isn't exercised
        assert!(
            idx.entries.len() > 1,
            "fixture must be multi-segment, got {}",
            idx.entries.len()
        );
        let field = idx
            .indexed_fields()
            .into_iter()
            .next()
            .expect("an indexed field");
        let avg = idx.avg_field_len(&field);
        assert!(avg >= 1.0, "avg must be >= 1.0, got {avg}");
        // the weighted mean must equal the manual average of field_len over all docs
        let n = idx.total_docs();
        let manual: f64 = (0..n)
            .map(|d| f64::from(idx.field_len(d, &field)))
            .sum::<f64>()
            / n as f64;
        assert!(
            (f64::from(avg) - manual).abs() < 1e-3,
            "avg={avg} manual={manual} field={field}"
        );
    }
}
