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
    /// Collection-wide average field length (BM25 length normalization). Default
    /// `1.0` = "no length signal" for readers that do not track lengths; the real
    /// readers override it. The real (on-disk) readers precompute this at open;
    /// `MemoryIndex` computes it on demand.
    fn avg_field_len(&self, field: &str) -> f32 {
        let _ = field;
        1.0
    }
    fn stored_fields(&self, doc_id: usize) -> HashMap<String, String>;
    fn terms_with_prefix(&self, field: &str, prefix: &str) -> Vec<String>;
    /// Like `terms_with_prefix`, but returns at most `limit` terms (lexicographic-first).
    /// The default collects then truncates; the on-disk ZSL readers override it to stop the
    /// dictionary scan early, so a pathological prefix (an empty/1-char prefix over a huge
    /// keyword field) never materializes the whole vocabulary. `usize::MAX` ≈ unbounded.
    fn terms_with_prefix_limited(&self, field: &str, prefix: &str, limit: usize) -> Vec<String> {
        let mut v = self.terms_with_prefix(field, prefix);
        v.truncate(limit);
        v
    }
    /// Terms of `field` within the inclusive range `[lower, upper]` (either bound `None` =
    /// unbounded on that side), ascending. Bounds are compared as byte strings against the
    /// stored term form. The default filters `terms_with_prefix(field, "")`; the on-disk
    /// term dictionary overrides it with a bounded seek+scan so a narrow range does not scan
    /// the field's whole vocabulary.
    fn terms_in_range(&self, field: &str, lower: Option<&str>, upper: Option<&str>) -> Vec<String> {
        self.terms_with_prefix(field, "")
            .into_iter()
            .filter(|t| {
                lower.is_none_or(|lo| t.as_str() >= lo) && upper.is_none_or(|hi| t.as_str() <= hi)
            })
            .collect()
    }
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

/// Uniform-stride sampled mean of a field's per-doc lengths, bounded to at most
/// `AVG_SAMPLE_CAP` samples. Computing the collection's average field length at open
/// must stay O(1): the on-disk readers open per request (PHP is shared-nothing), so an
/// O(num_docs) fold there would tax every low-hit query. The average is a stable
/// statistic (and the ZSL norm byte is already an 8-bit approximation), so a bounded
/// sample estimates it with negligible error. `n` is the doc count; `value(i)` yields
/// the i-th length. Returns 1.0 for an empty population or an all-zero total.
pub(crate) fn sampled_avg_field_len(n: usize, value: impl Fn(usize) -> u32) -> f32 {
    const AVG_SAMPLE_CAP: usize = 8192;
    if n == 0 {
        return 1.0;
    }
    let stride = (n / AVG_SAMPLE_CAP).max(1);
    let mut total: u64 = 0;
    let mut count: u64 = 0;
    let mut i = 0;
    while i < n {
        total += u64::from(value(i));
        count += 1;
        i += stride;
    }
    if total == 0 {
        1.0
    } else {
        total as f32 / count as f32
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
                        write_vint(&mut postings_bin, u64::from(p));
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
            .map_or(0, std::collections::HashMap::len)
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

    fn avg_field_len(&self, field: &str) -> f32 {
        if self.num_docs == 0 {
            return 1.0;
        }
        let total: u64 = (0..self.num_docs)
            .map(|d| u64::from(self.field_len(d, field)))
            .sum();
        if total == 0 {
            1.0
        } else {
            total as f32 / self.num_docs as f32
        }
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

    #[test]
    fn avg_field_len_averages_over_all_docs() {
        let mut idx = MemoryIndex::new();
        for text in ["a b c", "a", "a b"] {
            // lengths: 3, 1, 2  => avg = 6 / 3 = 2.0
            let mut d = Document::new();
            d.add("body", text, FieldKind::Text);
            idx.add_document(d);
        }
        assert!(
            (idx.avg_field_len("body") - 2.0).abs() < 1e-6,
            "got {}",
            idx.avg_field_len("body")
        );
    }

    #[test]
    fn avg_field_len_unknown_field_is_one() {
        let idx = MemoryIndex::new();
        assert_eq!(idx.avg_field_len("nope"), 1.0);
    }

    #[test]
    fn sampled_avg_field_len_guards_and_estimates() {
        // empty population and all-zero total both fall back to 1.0
        assert_eq!(sampled_avg_field_len(0, |_| 0), 1.0);
        assert_eq!(sampled_avg_field_len(100, |_| 0), 1.0);
        // constant length: the sample equals the exact mean at any size (past the cap)
        assert_eq!(sampled_avg_field_len(1_000_000, |_| 7), 7.0);
        // two blocks past the cap: stride sampling recovers the true mean (6.0) closely
        let est = sampled_avg_field_len(1_000_000, |i| if i < 500_000 { 4 } else { 8 });
        assert!((est - 6.0).abs() < 0.1, "est={est}");
    }

    #[test]
    fn terms_with_prefix_limited_bounds_and_matches_prefix() {
        let mut idx = MemoryIndex::new();
        let mut d = Document::new();
        d.add("body", "aa ab ac ad ae bx", FieldKind::Text);
        idx.add_document(d);
        // lexicographic-first 3 of the "a" bucket
        assert_eq!(
            idx.terms_with_prefix_limited("body", "a", 3),
            vec!["aa".to_string(), "ab".to_string(), "ac".to_string()]
        );
        // limit above the bucket size returns the whole bucket, matching the unbounded call
        assert_eq!(
            idx.terms_with_prefix_limited("body", "a", 999),
            idx.terms_with_prefix("body", "a")
        );
    }

    #[test]
    fn terms_in_range_filters_inclusive_bounds() {
        let mut idx = MemoryIndex::new();
        for v in ["100", "200", "300", "400"] {
            let mut d = Document::new();
            d.add("created_at_key", v, FieldKind::Keyword);
            idx.add_document(d);
        }
        // inclusive both ends: [200,300] => 200,300
        assert_eq!(
            idx.terms_in_range("created_at_key", Some("200"), Some("300")),
            vec!["200".to_string(), "300".to_string()]
        );
        // open-ended upper: [300,None] => 300,400
        assert_eq!(
            idx.terms_in_range("created_at_key", Some("300"), None),
            vec!["300".to_string(), "400".to_string()]
        );
        // open-ended lower: [None,200] => 100,200
        assert_eq!(
            idx.terms_in_range("created_at_key", None, Some("200")),
            vec!["100".to_string(), "200".to_string()]
        );
        // bounds need not be actual terms: [150,350] => 200,300
        assert_eq!(
            idx.terms_in_range("created_at_key", Some("150"), Some("350")),
            vec!["200".to_string(), "300".to_string()]
        );
        // unknown field => empty
        assert!(idx.terms_in_range("nope", None, None).is_empty());
    }
}
