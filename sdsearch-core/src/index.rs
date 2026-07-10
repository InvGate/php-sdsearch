//! in-memory inverted index (initial slice, no persistence).

use crate::analysis::analyze;
use crate::doc::{Document, FieldKind};
use crate::serialize::write_vint;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::path::Path;

/// postings of a term in a field: doc_id -> token positions (ascending); freq = positions.len()
type Postings = HashMap<usize, Vec<u32>>;

/// terms grouped by field: field -> [(term, postings)]
type TermPostingsByField<'a> = BTreeMap<&'a str, Vec<(&'a str, &'a HashMap<usize, Vec<u32>>)>>;

/// read contract shared by MemoryIndex (builder) and Segment (on disk).
pub trait IndexReader {
    fn num_docs(&self) -> usize;
    /// total docs for the idf denominator (includes deletes, like ZSL's `count()`,
    /// which is what ZSL feeds to `Similarity::idf`). With no deletes it equals `num_docs`;
    /// in ZSL it is maxDoc (incl. deletes). Default = num_docs (correct for readers without deletes).
    fn total_docs(&self) -> usize {
        self.num_docs()
    }
    fn doc_freq(&self, field: &str, term: &str) -> usize;
    fn postings_for(&self, field: &str, term: &str) -> Vec<(usize, u32)>;
    fn field_len(&self, doc_id: usize, field: &str) -> u32;
    fn stored_fields(&self, doc_id: usize) -> HashMap<String, String>;
    fn terms_with_prefix(&self, field: &str, prefix: &str) -> Vec<String>;
    fn positions_for(&self, field: &str, term: &str, doc_id: usize) -> Vec<u32>;
    /// names of indexed fields (unique, ascending order); used for all-fields queries.
    fn indexed_fields(&self) -> Vec<String>;

    /// ALL positions of a term, doc -> positions, in a single pass.
    /// The default (correct but O(docs·decode)) serves readers where `positions_for`
    /// is cheap; the on-disk segment reader overrides it with a single-pass decode (phrase at scale).
    fn positions_all(&self, field: &str, term: &str) -> HashMap<usize, Vec<u32>> {
        self.postings_for(field, term)
            .into_iter()
            .map(|(doc_id, _)| (doc_id, self.positions_for(field, term, doc_id)))
            .collect()
    }
}

/// on-disk segment metadata (format v2).
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct SegmentMeta {
    pub format_version: u32,
    pub num_docs: usize,
    pub fields: Vec<String>,
}

#[derive(Default)]
pub struct MemoryIndex {
    num_docs: usize,
    /// (field, term) -> postings
    postings: HashMap<(String, String), Postings>,
    /// doc_id -> (field -> token count) for length norm
    field_lengths: HashMap<usize, HashMap<String, u32>>,
    /// doc_id -> (field -> stored value) for stored fields
    stored: HashMap<usize, HashMap<String, String>>,
}

impl MemoryIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn num_docs(&self) -> usize {
        self.num_docs
    }

    pub fn add_document(&mut self, doc: Document) {
        let doc_id = self.num_docs;
        let mut lengths = HashMap::new();
        let mut stored_fields = HashMap::new();

        for field in doc.fields() {
            match field.kind {
                FieldKind::Text => {
                    let tokens = analyze(&field.value);
                    lengths.insert(field.name.clone(), tokens.len() as u32);
                    for (pos, term) in tokens.iter().enumerate() {
                        self.postings
                            .entry((field.name.clone(), term.clone()))
                            .or_default()
                            .entry(doc_id)
                            .or_default()
                            .push(pos as u32);
                    }
                    // text fields are also stored so they can be returned
                    stored_fields.insert(field.name.clone(), field.value.clone());
                }
                FieldKind::Keyword => {
                    // not tokenized: one term = the value, a single position
                    self.postings
                        .entry((field.name.clone(), field.value.clone()))
                        .or_default()
                        .insert(doc_id, vec![0]);
                    lengths.insert(field.name.clone(), 1);
                    stored_fields.insert(field.name.clone(), field.value.clone());
                }
                FieldKind::Stored => {
                    stored_fields.insert(field.name.clone(), field.value.clone());
                }
            }
        }

        self.field_lengths.insert(doc_id, lengths);
        self.stored.insert(doc_id, stored_fields);
        self.num_docs += 1;
    }

    /// serializes the index to directory `dir` in format v2 (build-once).
    pub fn write_to(&self, dir: &Path) -> std::io::Result<()> {
        let io_err = |e: fst::Error| std::io::Error::other(e);
        std::fs::create_dir_all(dir)?;

        // group terms by field, in lexicographic field order
        let mut by_field: TermPostingsByField = BTreeMap::new();
        for ((field, term), postings) in &self.postings {
            by_field
                .entry(field.as_str())
                .or_default()
                .push((term.as_str(), postings));
        }

        let mut field_names: Vec<String> = Vec::new();
        for (field, mut terms) in by_field {
            field_names.push(field.to_string());
            // fst requires insertion in lexicographic byte order; str Ord == byte order in UTF-8
            terms.sort_by(|a, b| a.0.cmp(b.0));

            let mut postings_bin: Vec<u8> = Vec::new();
            let mut builder = fst::MapBuilder::memory();
            for (term, postings) in terms {
                let offset = postings_bin.len() as u64;
                let mut docs: Vec<(usize, &Vec<u32>)> = postings
                    .iter()
                    .map(|(d, positions)| (*d, positions))
                    .collect();
                docs.sort_by_key(|(d, _)| *d);
                write_vint(&mut postings_bin, docs.len() as u64);
                let mut prev = 0usize;
                for (doc_id, positions) in docs {
                    write_vint(&mut postings_bin, (doc_id - prev) as u64);
                    write_vint(&mut postings_bin, positions.len() as u64);
                    let mut sorted = positions.clone();
                    sorted.sort_unstable();
                    for p in sorted {
                        write_vint(&mut postings_bin, p as u64);
                    }
                    prev = doc_id;
                }
                builder.insert(term.as_bytes(), offset).map_err(io_err)?;
            }
            let fst_bytes = builder.into_inner().map_err(io_err)?;
            std::fs::write(dir.join(format!("terms.{field}.fst")), fst_bytes)?;
            std::fs::write(dir.join(format!("postings.{field}.bin")), postings_bin)?;
        }

        // per-field lengths (by doc_id) and per-doc stored, via the trait accessors
        let mut lengths: HashMap<String, Vec<u32>> = HashMap::new();
        for field in &field_names {
            let mut v = vec![0u32; self.num_docs];
            for (doc_id, slot) in v.iter_mut().enumerate() {
                *slot = IndexReader::field_len(self, doc_id, field);
            }
            lengths.insert(field.clone(), v);
        }
        let stored: Vec<HashMap<String, String>> = (0..self.num_docs)
            .map(|d| IndexReader::stored_fields(self, d))
            .collect();

        let meta = SegmentMeta {
            format_version: 2,
            num_docs: self.num_docs,
            fields: field_names,
        };
        let json_err = |e: serde_json::Error| std::io::Error::other(e);
        std::fs::write(
            dir.join("meta.json"),
            serde_json::to_vec(&meta).map_err(json_err)?,
        )?;
        std::fs::write(
            dir.join("lengths.json"),
            serde_json::to_vec(&lengths).map_err(json_err)?,
        )?;
        std::fs::write(
            dir.join("stored.json"),
            serde_json::to_vec(&stored).map_err(json_err)?,
        )?;
        Ok(())
    }
}

impl IndexReader for MemoryIndex {
    fn num_docs(&self) -> usize {
        self.num_docs
    }

    /// doc_freq: how many docs the term appears in for the field
    fn doc_freq(&self, field: &str, term: &str) -> usize {
        self.postings
            .get(&(field.to_string(), term.to_string()))
            .map_or(0, |p| p.len())
    }

    /// postings of (field, term): iterator of (doc_id, term_freq); freq = number of positions
    fn postings_for(&self, field: &str, term: &str) -> Vec<(usize, u32)> {
        self.postings
            .get(&(field.to_string(), term.to_string()))
            .map(|p| {
                let mut v: Vec<(usize, u32)> = p
                    .iter()
                    .map(|(d, positions)| (*d, positions.len() as u32))
                    .collect();
                v.sort_by_key(|(d, _)| *d);
                v
            })
            .unwrap_or_default()
    }

    fn field_len(&self, doc_id: usize, field: &str) -> u32 {
        self.field_lengths
            .get(&doc_id)
            .and_then(|m| m.get(field))
            .copied()
            .unwrap_or(0)
    }

    fn stored_fields(&self, doc_id: usize) -> HashMap<String, String> {
        self.stored.get(&doc_id).cloned().unwrap_or_default()
    }

    /// terms of `field` starting with `prefix`, ascending order.
    fn terms_with_prefix(&self, field: &str, prefix: &str) -> Vec<String> {
        let mut out: Vec<String> = self
            .postings
            .keys()
            .filter(|(f, t)| f == field && t.starts_with(prefix))
            .map(|(_, t)| t.clone())
            .collect();
        out.sort();
        out
    }

    /// positions of `term` in `field` for `doc_id`, ascending order.
    fn positions_for(&self, field: &str, term: &str, doc_id: usize) -> Vec<u32> {
        self.postings
            .get(&(field.to_string(), term.to_string()))
            .and_then(|p| p.get(&doc_id))
            .map(|v| {
                let mut v = v.clone();
                v.sort_unstable();
                v
            })
            .unwrap_or_default()
    }

    fn indexed_fields(&self) -> Vec<String> {
        let mut set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for (field, _term) in self.postings.keys() {
            set.insert(field.clone());
        }
        set.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexes_text_and_counts_docs() {
        let mut idx = MemoryIndex::new();
        let mut d = Document::new();
        d.add("title", "hello hello world", FieldKind::Text);
        idx.add_document(d);

        assert_eq!(idx.num_docs(), 1);
        assert_eq!(idx.doc_freq("title", "hello"), 1);
        assert_eq!(idx.postings_for("title", "hello"), vec![(0, 2)]);
        assert_eq!(idx.field_len(0, "title"), 3);
    }

    #[test]
    fn records_token_positions() {
        let mut idx = MemoryIndex::new();
        let mut d = Document::new();
        d.add("title", "hello hello world", FieldKind::Text);
        idx.add_document(d);
        assert_eq!(idx.positions_for("title", "hello", 0), vec![0, 1]);
        assert_eq!(idx.positions_for("title", "world", 0), vec![2]);
        // freq is still derived from the positions
        assert_eq!(idx.postings_for("title", "hello"), vec![(0, 2)]);
    }

    #[test]
    fn keyword_field_is_not_tokenized() {
        let mut idx = MemoryIndex::new();
        let mut d = Document::new();
        d.add("status_key", "In Progress", FieldKind::Keyword);
        idx.add_document(d);

        // full value as one term; not split into "in"/"progress"
        assert_eq!(idx.doc_freq("status_key", "In Progress"), 1);
        assert_eq!(idx.doc_freq("status_key", "in"), 0);
    }

    #[test]
    fn indexed_fields_lists_unique_sorted() {
        let mut idx = MemoryIndex::new();
        let mut d = Document::new();
        d.add("title", "hello world", FieldKind::Text);
        d.add("body", "hello", FieldKind::Text);
        idx.add_document(d);
        assert_eq!(
            idx.indexed_fields(),
            vec!["body".to_string(), "title".to_string()]
        );
    }
}
