//! ZslSegment: implements IndexReader over a single-segment ZSL index.
use crate::index::IndexReader;
use crate::zsl::cfs::CompoundFile;
use crate::zsl::deletes::DeletedDocs;
use crate::zsl::fields::{FieldInfo, read_field_infos};
use crate::zsl::norms::{approx_field_len, read_norms};
use crate::zsl::postings::{for_each_posting, read_all_positions, read_freqs, read_positions};
use crate::zsl::stored::{StoredRaw, read_stored_fields, read_stored_raw};
use crate::zsl::terms::{TermCursor, TermDict, TermInfo};
use std::collections::HashMap;
use std::path::Path;

pub struct ZslSegment {
    num_docs_total: usize,
    /// live docs (num_docs_total minus deletes), precomputed at open (num_docs() is O(1)).
    num_docs_live: usize,
    fields: Vec<FieldInfo>,
    dict: TermDict,
    norms: HashMap<String, Vec<u8>>,
    /// per-field average length over num_docs_total, precomputed once at open from
    /// the norm bytes already in RAM (folded into the open path; no per-query cost).
    avg_field_len: HashMap<String, f32>,
    deletes: DeletedDocs,
    cfs: CompoundFile,
    fdx_name: String,
    fdt_name: String,
    frq_name: String,
    prx_name: String,
}

impl ZslSegment {
    /// total docs (includes deletes); size of the segment's id space.
    pub fn max_doc(&self) -> usize {
        self.num_docs_total
    }

    /// field infos in field-number order (for the field union in the merge).
    pub fn field_infos(&self) -> &[crate::zsl::fields::FieldInfo] {
        &self.fields
    }

    /// `true` if the segment has a positions file (`.prx`). A segment with indexed
    /// fields but NO `.prx` would silently lose positions during the merge.
    pub fn has_prx(&self) -> bool {
        !self.prx_name.is_empty()
    }

    /// Is the local doc deleted according to this segment's `.del`?
    pub fn is_deleted(&self, local_doc: usize) -> bool {
        self.deletes.is_deleted(local_doc)
    }

    /// column of the field's raw norm bytes (one per doc, incl. deletes), or `None`.
    /// The merge COPIES them verbatim (no re-encoding) into the merged segment.
    pub fn norm_bytes(&self, field: &str) -> Option<&[u8]> {
        self.norms.get(field).map(std::vec::Vec::as_slice)
    }

    /// all `(field, text)` terms of the segment (to walk them during the merge).
    pub fn all_terms(&self) -> Vec<(String, String)> {
        self.dict.iter_terms()
    }

    /// Streams a term's LIVE postings (deletes filtered) one local doc at a time, ascending.
    /// No-op if the term or `.prx` is absent (mirrors `positions_all`'s degradation).
    pub fn for_each_live_posting(&self, field: &str, term: &str, mut f: impl FnMut(usize, &[u32])) {
        if self.prx_name.is_empty() {
            return;
        }
        let Some(ti) = self.dict.info(field, term) else {
            return;
        };
        let (Some(frq), Some(prx)) = (self.cfs.sub(&self.frq_name), self.cfs.sub(&self.prx_name))
        else {
            return;
        };
        // degrade: a corrupt tail stops iteration rather than panicking across FFI
        let _ = for_each_posting(frq, prx, &ti, |d, pos| {
            if !self.deletes.is_deleted(d) {
                f(d, pos);
            }
        });
    }

    /// Like [`for_each_live_posting`](Self::for_each_live_posting) but takes a `TermInfo`
    /// the caller already has (captured from a [`TermCursor::peek_info`] during the merge)
    /// instead of resolving it via `self.dict.info(field, term)`. Semantics are otherwise
    /// identical: no-op if `.prx` is absent, deletes filtered, a corrupt tail stops iteration
    /// rather than panicking across FFI.
    ///
    /// This is the merge's fast path: the lazy `TermDict::info` forward-decodes up to
    /// `INDEX_INTERVAL` (128) `.tis` entries per call, so re-looking-up every term during a
    /// full-index merge is O(term_count × 128); the cursor already decoded each `TermInfo`
    /// while walking `.tis` once, so threading it through here avoids all the re-seeking.
    pub fn for_each_live_posting_ti(&self, term_info: &TermInfo, mut f: impl FnMut(usize, &[u32])) {
        if self.prx_name.is_empty() {
            return;
        }
        let (Some(frq), Some(prx)) = (self.cfs.sub(&self.frq_name), self.cfs.sub(&self.prx_name))
        else {
            return;
        };
        // degrade: a corrupt tail stops iteration rather than panicking across FFI
        let _ = for_each_posting(frq, prx, term_info, |d, pos| {
            if !self.deletes.is_deleted(d) {
                f(d, pos);
            }
        });
    }

    /// lazy cursor over every `(field, term)` pair in ZSL canonical order
    /// (field names ascending, terms ascending within each field), without
    /// materializing a `Vec` of all terms like `all_terms` does. Used by the
    /// bounded-memory k-way streaming merge.
    pub fn term_cursor(&self) -> TermCursor {
        self.dict.cursor()
    }

    /// stored fields of a doc in write order (LOCAL field_num + tokenized flag).
    pub fn stored_raw(&self, local_doc: usize) -> std::io::Result<Vec<StoredRaw>> {
        read_stored_raw(
            self.cfs.sub(&self.fdx_name).unwrap(),
            self.cfs.sub(&self.fdt_name).unwrap(),
            local_doc,
        )
    }

    /// opens the only segment in the directory (scans the first .cfs and any .del).
    pub fn open(index_dir: &Path) -> std::io::Result<ZslSegment> {
        let cfs_path = std::fs::read_dir(index_dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .find(|p| p.extension().is_some_and(|x| x == "cfs"))
            .ok_or_else(|| std::io::Error::other("no .cfs in index dir"))?;
        let del_path = std::fs::read_dir(index_dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .find(|p| p.extension().is_some_and(|x| x == "del"));
        Self::open_from(index_dir, &cfs_path, del_path)
    }

    /// opens a named segment (`<seg_name>.cfs`) and its `.del` per `del_gen`.
    pub fn open_named(
        index_dir: &Path,
        seg_name: &str,
        del_gen: i64,
    ) -> std::io::Result<ZslSegment> {
        let cfs_path = index_dir.join(format!("{seg_name}.cfs"));
        let del_path = match del_gen {
            -1 => None,
            0 => Some(index_dir.join(format!("{seg_name}.del"))),
            g => Some(index_dir.join(format!(
                "{seg_name}_{}.del",
                crate::zsl::segments::to_base36(g as u64)
            ))),
        };
        Self::open_from(index_dir, &cfs_path, del_path)
    }

    fn open_from(
        index_dir: &Path,
        cfs_path: &Path,
        del_path: Option<std::path::PathBuf>,
    ) -> std::io::Result<ZslSegment> {
        let cfs = CompoundFile::open(cfs_path)?;
        let name_ending = |ext: &str| cfs.names().into_iter().find(|n| n.ends_with(ext));
        let fnm = name_ending(".fnm").ok_or_else(|| std::io::Error::other("no .fnm"))?;
        let tis = name_ending(".tis").ok_or_else(|| std::io::Error::other("no .tis"))?;
        let tii_name = name_ending(".tii").ok_or_else(|| std::io::Error::other("no .tii"))?;

        let fields = read_field_infos(cfs.sub(&fnm).unwrap())?;
        let field_names: Vec<String> = fields.iter().map(|f| f.name.clone()).collect();
        let dict = TermDict::open(
            cfs.sub(&tis).unwrap(),
            cfs.sub(&tii_name).unwrap(),
            &field_names,
        )?;

        let fdx_name = name_ending(".fdx").ok_or_else(|| std::io::Error::other("no .fdx"))?;
        let num_docs_total = cfs.sub(&fdx_name).unwrap().len() / 8;

        let indexed: Vec<String> = fields
            .iter()
            .filter(|f| f.is_indexed)
            .map(|f| f.name.clone())
            .collect();
        let norms = match name_ending(".nrm") {
            Some(n) => read_norms(cfs.sub(&n).unwrap(), &indexed, num_docs_total),
            None => HashMap::new(),
        };
        let avg_field_len = norms
            .iter()
            .map(|(field, bytes)| {
                let total: u64 = bytes.iter().map(|&b| u64::from(approx_field_len(b))).sum();
                let avg = if bytes.is_empty() || total == 0 {
                    1.0
                } else {
                    total as f32 / bytes.len() as f32
                };
                (field.clone(), avg)
            })
            .collect::<HashMap<String, f32>>();

        // .del lives OUTSIDE the .cfs; we load it only if the file exists. A corrupt or
        // unsupported (sparse) .del surfaces as an error at open time rather than a crash.
        let deletes = match del_path
            .filter(|p| p.exists())
            .and_then(|p| std::fs::read(p).ok())
        {
            Some(b) => DeletedDocs::read(&b)?,
            None => DeletedDocs::none(),
        };

        let fdt_name = name_ending(".fdt").ok_or_else(|| std::io::Error::other("no .fdt"))?;
        let frq_name = name_ending(".frq").ok_or_else(|| std::io::Error::other("no .frq"))?;
        let prx_name = name_ending(".prx").unwrap_or_default();

        let _ = index_dir; // the .del was already resolved above
        let num_docs_live = (0..num_docs_total)
            .filter(|&d| !deletes.is_deleted(d))
            .count();
        Ok(ZslSegment {
            num_docs_total,
            num_docs_live,
            fields,
            dict,
            norms,
            avg_field_len,
            deletes,
            cfs,
            fdx_name,
            fdt_name,
            frq_name,
            prx_name,
        })
    }
}

impl IndexReader for ZslSegment {
    fn num_docs(&self) -> usize {
        self.num_docs_live
    }

    /// maxDoc (incl. deletes) — what ZSL uses as N in idf.
    fn total_docs(&self) -> usize {
        self.num_docs_total
    }

    fn doc_freq(&self, field: &str, term: &str) -> usize {
        self.dict
            .info(field, term)
            .map_or(0, |ti| ti.doc_freq as usize)
    }

    fn postings_for(&self, field: &str, term: &str) -> Vec<(usize, u32)> {
        match self.dict.info(field, term) {
            // degrade: a corrupt .frq yields no postings rather than a panic across FFI
            Some(ti) => read_freqs(self.cfs.sub(&self.frq_name).unwrap(), &ti)
                .unwrap_or_default()
                .into_iter()
                .filter(|(d, _)| !self.deletes.is_deleted(*d))
                .collect(),
            None => Vec::new(),
        }
    }

    fn field_len(&self, doc_id: usize, field: &str) -> u32 {
        self.norms
            .get(field)
            .and_then(|v| v.get(doc_id))
            .map_or(1, |b| approx_field_len(*b))
    }

    fn avg_field_len(&self, field: &str) -> f32 {
        self.avg_field_len.get(field).copied().unwrap_or(1.0)
    }

    fn stored_fields(&self, doc_id: usize) -> HashMap<String, String> {
        // degrade: a corrupt .fdt/.fdx yields no stored fields rather than a panic across FFI
        read_stored_fields(
            self.cfs.sub(&self.fdx_name).unwrap(),
            self.cfs.sub(&self.fdt_name).unwrap(),
            &self.fields,
            doc_id,
        )
        .unwrap_or_default()
    }

    fn terms_with_prefix(&self, field: &str, prefix: &str) -> Vec<String> {
        let mut out = self.dict.terms_with_prefix(field, prefix);
        out.sort();
        out
    }

    fn positions_for(&self, field: &str, term: &str, doc_id: usize) -> Vec<u32> {
        match self.dict.info(field, term) {
            Some(ti) if self.prx_name.is_empty() => {
                let _ = ti;
                Vec::new()
            }
            // degrade: corrupt .frq/.prx yields no positions rather than a panic across FFI
            Some(ti) => read_positions(
                self.cfs.sub(&self.frq_name).unwrap(),
                self.cfs.sub(&self.prx_name).unwrap(),
                &ti,
                doc_id,
            )
            .unwrap_or_default(),
            None => Vec::new(),
        }
    }

    /// a single pass over `.frq`/`.prx` for the whole term (vs O(docs) walks).
    fn positions_all(&self, field: &str, term: &str) -> HashMap<usize, Vec<u32>> {
        match self.dict.info(field, term) {
            // degrade: corrupt .frq/.prx yields no positions rather than a panic across FFI
            Some(ti) if !self.prx_name.is_empty() => read_all_positions(
                self.cfs.sub(&self.frq_name).unwrap(),
                self.cfs.sub(&self.prx_name).unwrap(),
                &ti,
            )
            .unwrap_or_default()
            .into_iter()
            .filter(|(d, _)| !self.deletes.is_deleted(*d))
            .collect(),
            _ => HashMap::new(),
        }
    }

    fn indexed_fields(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .fields
            .iter()
            .filter(|f| f.is_indexed)
            .map(|f| f.name.clone())
            .collect();
        v.sort();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn seg() -> ZslSegment {
        let dir = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/zsl_index"
        ));
        ZslSegment::open(&dir).unwrap()
    }

    #[test]
    fn num_docs_matches_oracle() {
        assert_eq!(seg().num_docs(), 4);
    }

    #[test]
    fn postings_and_docfreq_match_oracle() {
        let s = seg();
        // incidents index: "new" in title in all 4 docs, freq 1 each
        assert_eq!(s.doc_freq("title", "new"), 4);
        assert_eq!(
            s.postings_for("title", "new"),
            vec![(0, 1), (1, 1), (2, 1), (3, 1)]
        );
    }

    #[test]
    fn stored_fields_round_trip() {
        let s = seg();
        assert_eq!(
            s.stored_fields(0).get("id_key").map(String::as_str),
            Some("165")
        );
    }

    #[test]
    fn zsl_segment_avg_field_len_is_precomputed_positive() {
        use std::path::PathBuf;
        let dir = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/zsl_index_kb"
        ));
        let s = ZslSegment::open(&dir).unwrap();
        assert!(s.avg_field_len("title") >= 1.0);
    }

    #[test]
    fn open_named_matches_scanning_open() {
        let dir = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/zsl_index_kb"
        ));
        // KB has a single segment "_2" with no deletes (del_gen -1)
        let named = ZslSegment::open_named(&dir, "_2", -1).unwrap();
        let scanned = ZslSegment::open(&dir).unwrap();
        assert_eq!(named.max_doc(), 20);
        assert_eq!(named.num_docs(), scanned.num_docs());
        assert_eq!(
            named.postings_for("title", "vpn"),
            scanned.postings_for("title", "vpn")
        );
    }

    #[test]
    fn indexed_fields_includes_title() {
        assert!(seg().indexed_fields().contains(&"title".to_string()));
    }

    #[test]
    fn open_errors_on_corrupt_segment() {
        let dir = std::env::temp_dir().join("sdsearch_seg_open_bad");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("_x.cfs"), [0x80u8]).unwrap(); // 1-byte garbage .cfs
        assert!(ZslSegment::open(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn seg_kb() -> ZslSegment {
        let dir = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/zsl_index_kb"
        ));
        // KB has a single segment "_2" with no deletes (del_gen -1); it spans
        // several stored/indexed fields, unlike the tiny "zsl_index" fixture.
        ZslSegment::open_named(&dir, "_2", -1).unwrap()
    }

    #[test]
    fn term_cursor_yields_all_terms_in_zsl_canonical_order() {
        let s = seg_kb();
        let mut expected = s.all_terms();
        expected.sort();

        // sanity: the fixture must actually exercise field-name ordering, not
        // just within-field term ordering.
        let distinct_fields: std::collections::HashSet<&String> =
            expected.iter().map(|(f, _)| f).collect();
        assert!(
            distinct_fields.len() >= 2,
            "fixture has too few fields to exercise field ordering: {distinct_fields:?}"
        );

        let mut got: Vec<(String, String)> = Vec::new();
        let mut cur = s.term_cursor();
        while let Some((field, term)) = cur.peek() {
            got.push((field.to_string(), term.to_string()));
            cur.advance();
        }
        assert_eq!(got, expected);
    }

    #[test]
    fn for_each_live_posting_matches_positions_all() {
        let s = seg_kb();
        let (field, term) = s.all_terms().into_iter().next().unwrap();
        let mut expected: Vec<(usize, Vec<u32>)> =
            s.positions_all(&field, &term).into_iter().collect();
        expected.sort_by_key(|(d, _)| *d);

        let mut got: Vec<(usize, Vec<u32>)> = Vec::new();
        s.for_each_live_posting(&field, &term, |d, pos| got.push((d, pos.to_vec())));
        got.sort_by_key(|(d, _)| *d);

        assert_eq!(got, expected);
    }

    fn seg_multiseg_1() -> ZslSegment {
        let dir = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/zsl_index_multiseg"
        ));
        // segment "_1" is the fixture's 2nd commit batch ("gamma vpn tutorial" local doc 0,
        // "delta backup notes" local doc 1); its "_1_1.del" (del_gen 1) marks local doc 1
        // deleted. Unlike the KB fixture (no deletes at all), this segment lets a test
        // actually exercise the delete-filter branch of `for_each_live_posting`.
        ZslSegment::open_named(&dir, "_1", 1).unwrap()
    }

    #[test]
    fn for_each_live_posting_excludes_genuinely_deleted_doc() {
        let s = seg_multiseg_1();

        // sanity: the fixture really has a deletion here, or the rest of this test is vacuous.
        assert!(
            s.is_deleted(1),
            "fixture local doc 1 (\"delta backup notes\") must be deleted"
        );

        // "backup" only appears in the deleted doc. Fetch its postings straight from the
        // .frq via read_freqs (delete-AGNOSTIC — no `deletes.is_deleted` filtering), so this
        // is an independent source from `positions_all` (which is itself delete-filtered)
        // and can actually prove `for_each_live_posting` removes something.
        let field = "title";
        let term = "backup";
        let ti = s
            .dict
            .info(field, term)
            .expect("fixture must index title:backup");
        let raw = read_freqs(s.cfs.sub(&s.frq_name).unwrap(), &ti).unwrap();
        let raw_docs: Vec<usize> = raw.iter().map(|(d, _)| *d).collect();
        assert!(
            raw_docs.contains(&1),
            "delete-agnostic postings must include the deleted doc: {raw_docs:?}"
        );

        let mut live: Vec<usize> = Vec::new();
        s.for_each_live_posting(field, term, |d, _pos| live.push(d));

        // the chosen term must have a genuine gap between delete-agnostic and live counts,
        // otherwise a dropped/inverted filter would not be caught by this test.
        assert!(
            raw_docs.len() > live.len(),
            "term must have delete-agnostic count > live count to be non-vacuous: raw={raw_docs:?} live={live:?}"
        );
        assert!(
            !live.contains(&1),
            "for_each_live_posting must exclude the deleted doc: live={live:?}"
        );
        assert!(live.iter().all(|d| !s.is_deleted(*d)));

        // must also agree with positions_all's doc set (both are delete-filtered).
        let mut expected: Vec<usize> = s.positions_all(field, term).into_keys().collect();
        expected.sort_unstable();
        let mut got = live.clone();
        got.sort_unstable();
        assert_eq!(got, expected);
    }

    #[test]
    fn merge_accessors_expose_fields_deletes_norms_terms_stored() {
        let s = seg(); // incidents fixture, 4 docs, no deletes
        // field_infos: includes title (indexed)
        assert!(
            s.field_infos()
                .iter()
                .any(|f| f.name == "title" && f.is_indexed)
        );
        // is_deleted: nothing deleted in the fixture
        assert!(!s.is_deleted(0));
        // norm_bytes: title has a 4-byte column (one per doc)
        assert_eq!(s.norm_bytes("title").map(<[u8]>::len), Some(4));
        assert!(s.norm_bytes("campo_inexistente").is_none());
        // all_terms: title:new present
        assert!(
            s.all_terms()
                .contains(&("title".to_string(), "new".to_string()))
        );
        // stored_raw doc 0: contains id_key="165" (same value as stored_fields)
        let raw = s.stored_raw(0).unwrap();
        let names = s.field_infos();
        let idkey = raw.iter().find(|r| names[r.field_num].name == "id_key");
        assert_eq!(idkey.map(|r| r.value.as_str()), Some("165"));
    }
}
