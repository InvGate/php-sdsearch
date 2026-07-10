//! Writer for the Zend_Search_Lucene on-disk format (byte-faithful).
//! Each submodule is the INVERSE of a `zsl::*` reader: write → read back with the
//! trusted reader → assert == input is each piece's round-trip test.
//! Add-only append of ONE segment to an existing ZSL index.
pub mod cfs;
pub mod deletes;
pub(crate) mod durability;
pub mod fnm;
pub mod index_writer;
pub mod invert;
pub mod lock;
pub mod merge;
pub mod norms;
pub mod postings;
pub mod segments;
pub mod stored;
pub mod terms;

pub use index_writer::{CommitReport, IndexWriter};

/// Field kind; mirrors the host's schema→document mapping.
/// - `Text`: indexed + tokenized (analyzer) — e.g. title/description.
/// - `Keyword`: indexed NOT tokenized (suffix `_key`) — term = whole value.
/// - `UnIndexed`: stored only (suffix `_attr`) — no term, no norm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldKind {
    Text,
    Keyword,
    UnIndexed,
}

impl FieldKind {
    /// Does the field contribute terms to the inverted index?
    pub fn is_indexed(self) -> bool {
        matches!(self, FieldKind::Text | FieldKind::Keyword)
    }
    /// Is the field tokenized by the analyzer? (only `Text`)
    pub fn is_tokenized(self) -> bool {
        matches!(self, FieldKind::Text)
    }
}

#[derive(Debug, Clone)]
pub struct WriterField {
    pub name: String,
    pub value: String,
    pub kind: FieldKind,
    pub stored: bool,
}

impl WriterField {
    pub fn text(name: &str, value: &str) -> Self {
        Self { name: name.into(), value: value.into(), kind: FieldKind::Text, stored: true }
    }
    pub fn keyword(name: &str, value: &str) -> Self {
        Self { name: name.into(), value: value.into(), kind: FieldKind::Keyword, stored: true }
    }
    pub fn unindexed(name: &str, value: &str) -> Self {
        Self { name: name.into(), value: value.into(), kind: FieldKind::UnIndexed, stored: true }
    }
}

#[derive(Debug, Clone, Default)]
pub struct WriterDoc {
    pub fields: Vec<WriterField>,
}

/// document/field boosts + the streaming writer's flush buffer size.
#[derive(Debug, Clone)]
pub struct WriterOpts {
    pub doc_boost: f32,
    /// docs buffered before flushing a segment. Default 1000 (vs ZSL's 10);
    /// bounds the streaming writer's RAM. Ignored by `append_documents` (batch).
    pub max_buffered_docs: usize,
}

impl Default for WriterOpts {
    fn default() -> Self {
        Self { doc_boost: 1.0, max_buffered_docs: 1000 }
    }
}

/// append result: the created segment and the resulting generation.
#[derive(Debug, Clone, PartialEq)]
pub struct AppendReport {
    pub segment_name: String,
    pub doc_count: usize,
    pub generation: u64,
}

/// Writes ONE byte-faithful segment `<seg_name>.cfs` from `docs` (invert + writers).
/// Returns the doc count. Does NOT touch the generation: the segment stays invisible until
/// a caller references it in segments_{N+1}. Primitive shared by the batch append
/// and the streaming writer's flush.
pub(crate) fn write_segment_cfs(
    index_dir: &std::path::Path,
    seg_name: &str,
    docs: &[WriterDoc],
    opts: &WriterOpts,
) -> std::io::Result<usize> {
    let inv = invert::invert(docs, opts);
    let fnm = fnm::write_fnm(&inv.fields);
    let (fdt, fdx) = stored::write_stored(&inv.stored);
    let nrm = norms::write_norms(&inv.norm_lengths, opts.doc_boost);
    let dict = terms::write_term_dict(&inv.terms);

    let cfs_bytes = assemble_cfs(seg_name, &fnm, &fdt, &fdx, &nrm, &dict);
    std::fs::write(index_dir.join(format!("{seg_name}.cfs")), &cfs_bytes)?;
    Ok(inv.doc_count)
}

/// Assembles the `.cfs` bytes in ZSL's canonical file order
/// (`.fdx .fdt .fnm .nrm .tis .tii .frq .prx`). The single place that order lives:
/// shared by `write_segment_cfs` and `merge::merge_segments`.
pub(crate) fn assemble_cfs(
    seg_name: &str,
    fnm: &[u8],
    fdt: &[u8],
    fdx: &[u8],
    nrm: &[u8],
    dict: &terms::DictFiles,
) -> Vec<u8> {
    let files: Vec<(&str, &[u8])> = vec![
        (".fdx", fdx),
        (".fdt", fdt),
        (".fnm", fnm),
        (".nrm", nrm),
        (".tis", &dict.tis),
        (".tii", &dict.tii),
        (".frq", &dict.frq),
        (".prx", &dict.prx),
    ];
    cfs::write_cfs(seg_name, &files)
}

/// Appends `docs` as ONE new byte-faithful segment to an EXISTING ZSL index (add-only).
/// Writes `<seg>.cfs` and then flips the generation (segments_{N+1} + segments.gen).
/// Leaves the index readable by both the native reader and ZSL (sequential handoff).
pub fn append_documents(
    index_dir: &std::path::Path,
    docs: &[WriterDoc],
    opts: &WriterOpts,
) -> std::io::Result<AppendReport> {
    let gen = segments::read_generation(index_dir)?;

    // empty batch = no-op: ZSL never creates a 0-doc segment (DocumentWriter::close
    // returns null). We do not advance the generation or write a .cfs.
    if docs.is_empty() {
        return Ok(AppendReport {
            segment_name: String::new(),
            doc_count: 0,
            generation: gen.generation,
        });
    }

    let seg_name = segments::segment_name(gen.name_counter);
    let doc_count = write_segment_cfs(index_dir, &seg_name, docs, opts)?;

    // only once the .cfs is on disk do we flip the generation.
    let new_gen = segments::write_appended_generation(index_dir, &gen, &seg_name, doc_count as u32)?;

    Ok(AppendReport { segment_name: seg_name, doc_count, generation: new_gen })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::IndexReader;
    use crate::zsl::index::ZslIndex;
    use std::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    /// copies the ENTIRE KB fixture (incl. `_2.cfs`) to a fresh temp dir.
    fn temp_kb_full() -> std::path::PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("sdsearch_append_{}_{}", std::process::id(), n));
        std::fs::create_dir_all(&dir).unwrap();
        let src = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/zsl_index_kb"
        ));
        for entry in std::fs::read_dir(&src).unwrap() {
            let p = entry.unwrap().path();
            std::fs::copy(&p, dir.join(p.file_name().unwrap())).unwrap();
        }
        dir
    }

    #[test]
    fn empty_batch_is_a_noop_leaving_generation_untouched() {
        // ZSL never creates a 0-doc segment (DocumentWriter::close returns null).
        // An empty append must not advance the generation or write a .cfs.
        let dir = temp_kb_full();
        let report = append_documents(&dir, &[], &WriterOpts::default()).unwrap();
        assert_eq!(report.doc_count, 0);
        assert_eq!(report.segment_name, "");
        assert_eq!(report.generation, 6); // KB generation unchanged

        // no new segment created, generation not flipped
        assert!(!dir.join("_3.cfs").exists());
        assert!(!dir.join("segments_7").exists());
        let idx = ZslIndex::open(&dir).unwrap();
        assert_eq!(idx.num_docs(), 20);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn appends_batch_to_real_index_and_reopens_with_reader() {
        let dir = temp_kb_full();
        let before = ZslIndex::open(&dir).unwrap().num_docs();
        assert_eq!(before, 20);

        let docs = vec![WriterDoc {
            fields: vec![
                WriterField::text("title", "zqxterm alpha"),
                WriterField::keyword("id", "KB-9001"),
            ],
        }];
        let report = append_documents(&dir, &docs, &WriterOpts::default()).unwrap();
        assert_eq!(report.segment_name, "_3");
        assert_eq!(report.doc_count, 1);
        assert_eq!(report.generation, 7);

        // reopen with the native reader: old + new visible
        let idx = ZslIndex::open(&dir).unwrap();
        assert_eq!(idx.num_docs(), before + 1);
        assert_eq!(idx.doc_freq("title", "zqxterm"), 1);

        // stored of the new doc (base doc-id = _2's maxDoc = 20)
        let stored = idx.stored_fields(before);
        assert_eq!(stored.get("title").unwrap(), "zqxterm alpha");
        assert_eq!(stored.get("id").unwrap(), "KB-9001");

        std::fs::remove_dir_all(&dir).ok();
    }
}
