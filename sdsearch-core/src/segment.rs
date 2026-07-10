//! read-only reader of an on-disk index (format v2, build-once).

use crate::index::{IndexReader, SegmentMeta};
use crate::serialize::read_vint;
use crate::zsl::bytes::checked_capacity;
use fst::{IntoStreamer, Streamer};
use std::collections::HashMap;
use std::path::Path;

pub struct Segment {
    num_docs: usize,
    fsts: HashMap<String, fst::Map<Vec<u8>>>,
    postings: HashMap<String, Vec<u8>>,
    lengths: HashMap<String, Vec<u32>>,
    stored: Vec<HashMap<String, String>>,
}

impl Segment {
    pub fn open(dir: &Path) -> std::io::Result<Segment> {
        let io = |e: fst::Error| std::io::Error::other(e);
        let js = |e: serde_json::Error| std::io::Error::other(e);

        let meta: SegmentMeta =
            serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?).map_err(js)?;
        let lengths: HashMap<String, Vec<u32>> =
            serde_json::from_slice(&std::fs::read(dir.join("lengths.json"))?).map_err(js)?;
        let stored: Vec<HashMap<String, String>> =
            serde_json::from_slice(&std::fs::read(dir.join("stored.json"))?).map_err(js)?;

        if meta.format_version != 2 {
            return Err(std::io::Error::other(format!(
                "unsupported segment format version {}",
                meta.format_version
            )));
        }

        let mut fsts = HashMap::new();
        let mut postings = HashMap::new();
        for field in &meta.fields {
            let bytes = std::fs::read(dir.join(format!("terms.{field}.fst")))?;
            fsts.insert(field.clone(), fst::Map::new(bytes).map_err(io)?);
            postings.insert(field.clone(), std::fs::read(dir.join(format!("postings.{field}.bin")))?);
        }
        Ok(Segment { num_docs: meta.num_docs, fsts, postings, lengths, stored })
    }

    fn offset_of(&self, field: &str, term: &str) -> Option<u64> {
        self.fsts.get(field).and_then(|m| m.get(term.as_bytes()))
    }

    /// Fallible core of `postings_for`; the trait method degrades an `Err`
    /// (corrupt/truncated postings) to an empty result instead of panicking.
    fn try_postings_for(&self, field: &str, term: &str) -> std::io::Result<Vec<(usize, u32)>> {
        let Some(off) = self.offset_of(field, term) else {
            return Ok(Vec::new());
        };
        let data = &self.postings[field];
        let mut pos = off as usize;
        let doc_freq = read_vint(data, &mut pos)? as usize;
        let mut out = Vec::with_capacity(checked_capacity(doc_freq, data.len().saturating_sub(pos)));
        let mut prev = 0usize;
        for _ in 0..doc_freq {
            let delta = read_vint(data, &mut pos)? as usize;
            let freq = read_vint(data, &mut pos)? as u32;
            for _ in 0..freq {
                read_vint(data, &mut pos)?; // skip positions
            }
            let doc_id = prev + delta;
            out.push((doc_id, freq));
            prev = doc_id;
        }
        Ok(out)
    }

    /// Fallible core of `positions_for` (see `try_postings_for`).
    fn try_positions_for(&self, field: &str, term: &str, doc_id: usize) -> std::io::Result<Vec<u32>> {
        let Some(off) = self.offset_of(field, term) else {
            return Ok(Vec::new());
        };
        let data = &self.postings[field];
        let mut pos = off as usize;
        let doc_freq = read_vint(data, &mut pos)? as usize;
        let mut prev = 0usize;
        for _ in 0..doc_freq {
            let delta = read_vint(data, &mut pos)? as usize;
            let freq = read_vint(data, &mut pos)? as usize;
            let d = prev + delta;
            let mut positions = Vec::with_capacity(checked_capacity(freq, data.len().saturating_sub(pos)));
            for _ in 0..freq {
                positions.push(read_vint(data, &mut pos)? as u32);
            }
            if d == doc_id {
                return Ok(positions);
            }
            prev = d;
        }
        Ok(Vec::new())
    }
}

impl IndexReader for Segment {
    fn num_docs(&self) -> usize {
        self.num_docs
    }

    fn doc_freq(&self, field: &str, term: &str) -> usize {
        match self.offset_of(field, term) {
            Some(off) => {
                let data = &self.postings[field];
                let mut pos = off as usize;
                // degrade: a corrupt docFreq varint reports 0 rather than panicking
                read_vint(data, &mut pos).unwrap_or(0) as usize
            }
            None => 0,
        }
    }

    fn postings_for(&self, field: &str, term: &str) -> Vec<(usize, u32)> {
        // degrade: corrupt postings yield no results rather than a panic across FFI
        self.try_postings_for(field, term).unwrap_or_default()
    }

    fn field_len(&self, doc_id: usize, field: &str) -> u32 {
        self.lengths.get(field).and_then(|v| v.get(doc_id)).copied().unwrap_or(0)
    }

    fn stored_fields(&self, doc_id: usize) -> HashMap<String, String> {
        self.stored.get(doc_id).cloned().unwrap_or_default()
    }

    fn terms_with_prefix(&self, field: &str, prefix: &str) -> Vec<String> {
        let Some(map) = self.fsts.get(field) else {
            return Vec::new();
        };
        let pbytes = prefix.as_bytes();
        let mut out = Vec::new();
        // the range >= prefix walks in lexicographic order; terms starting with
        // `prefix` are contiguous, so we stop at the first one that does not.
        let mut stream = map.range().ge(pbytes).into_stream();
        while let Some((key, _)) = stream.next() {
            if !key.starts_with(pbytes) {
                break;
            }
            out.push(String::from_utf8_lossy(key).into_owned());
        }
        out
    }

    /// positions of `term` in `field` for `doc_id`, ascending order.
    fn positions_for(&self, field: &str, term: &str, doc_id: usize) -> Vec<u32> {
        // degrade: corrupt postings yield no positions rather than a panic across FFI
        self.try_positions_for(field, term, doc_id).unwrap_or_default()
    }

    fn indexed_fields(&self) -> Vec<String> {
        let mut v: Vec<String> = self.fsts.keys().cloned().collect();
        v.sort();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doc::{Document, FieldKind};
    use crate::index::MemoryIndex;

    fn build() -> MemoryIndex {
        let mut idx = MemoryIndex::new();
        let mut d0 = Document::new();
        d0.add("title", "hello hello world", FieldKind::Text);
        d0.add("status_key", "In Progress", FieldKind::Keyword);
        idx.add_document(d0);
        let mut d1 = Document::new();
        d1.add("title", "world peace", FieldKind::Text);
        idx.add_document(d1);
        idx
    }

    #[test]
    fn accessors_match_memory_index_after_roundtrip() {
        let mem = build();
        let dir = std::env::temp_dir().join(format!("sdsearch_seg_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        mem.write_to(&dir).unwrap();
        let seg = Segment::open(&dir).unwrap();

        assert_eq!(seg.num_docs(), mem.num_docs());
        assert_eq!(seg.doc_freq("title", "hello"), 1);
        assert_eq!(seg.postings_for("title", "hello"), vec![(0, 2)]);
        assert_eq!(seg.postings_for("title", "world"), vec![(0, 1), (1, 1)]);
        assert_eq!(seg.field_len(0, "title"), 3);
        assert_eq!(seg.doc_freq("status_key", "In Progress"), 1);
        assert_eq!(
            seg.stored_fields(0).get("title").map(String::as_str),
            Some("hello hello world")
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn terms_with_prefix_matches_between_memory_and_disk() {
        let mut mem = MemoryIndex::new();
        for text in ["test testing tested text team", "testable tester"] {
            let mut d = Document::new();
            d.add("body", text, FieldKind::Text);
            mem.add_document(d);
        }
        let dir = std::env::temp_dir().join(format!("sdsearch_prefix_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        mem.write_to(&dir).unwrap();
        let seg = Segment::open(&dir).unwrap();

        for prefix in ["te", "test", "tea", "zzz"] {
            assert_eq!(
                mem.terms_with_prefix("body", prefix),
                seg.terms_with_prefix("body", prefix),
                "prefix {prefix:?}"
            );
        }
        // spot-check content
        assert_eq!(
            mem.terms_with_prefix("body", "test"),
            vec!["test", "testable", "tested", "tester", "testing"]
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn positions_roundtrip_memory_to_disk() {
        let mut mem = MemoryIndex::new();
        let mut d = Document::new();
        d.add("body", "quick brown fox quick", FieldKind::Text);
        mem.add_document(d);
        let dir = std::env::temp_dir().join(format!("sdsearch_pos_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        mem.write_to(&dir).unwrap();
        let seg = Segment::open(&dir).unwrap();

        assert_eq!(seg.positions_for("body", "quick", 0), vec![0, 3]);
        assert_eq!(seg.positions_for("body", "brown", 0), vec![1]);
        assert_eq!(seg.positions_for("body", "quick", 0), mem.positions_for("body", "quick", 0));
        // freq is still correct after skipping positions
        assert_eq!(seg.postings_for("body", "quick"), vec![(0, 2)]);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
