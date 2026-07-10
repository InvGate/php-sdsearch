//! ZslIndex: aggregates N ZslSegment behind the IndexReader trait (global ids + unified stats).

use crate::index::IndexReader;
use crate::zsl::segment::ZslSegment;
use crate::zsl::segments::read_segment_infos;
use std::collections::HashMap;
use std::path::Path;

struct Entry {
    base: usize,     // global doc id of the segment's first doc
    max_doc: usize,  // size of the id space (incl. deletes)
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
        Ok(ZslIndex { entries, num_docs, total_docs })
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
        self.entries.iter().map(|e| e.seg.doc_freq(field, term)).sum()
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
        PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/zsl_index_kb"))
    }

    #[test]
    fn single_segment_index_equals_segment() {
        let idx = ZslIndex::open(&kb()).unwrap();
        let seg = ZslSegment::open(&kb()).unwrap();
        assert_eq!(idx.num_docs(), seg.num_docs());
        assert_eq!(idx.doc_freq("title", "vpn"), seg.doc_freq("title", "vpn"));
        assert_eq!(idx.postings_for("title", "vpn"), seg.postings_for("title", "vpn"));
        // stored routing: the same doc returns the same id_key
        let d0 = idx.postings_for("title", "vpn")[0].0;
        assert_eq!(idx.stored_fields(d0), seg.stored_fields(d0));
    }
}
