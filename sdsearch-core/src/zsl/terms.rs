//! Term dictionary reader (.tis): sorted terms + docFreq + pointers to .frq/.prx.
use crate::zsl::bytes::{read_i32_be, read_modified_utf8, read_u64_be, read_vint};

#[derive(Debug, Clone, PartialEq)]
pub struct TermInfo {
    pub doc_freq: u32,
    pub freq_pointer: u64,
    pub prox_pointer: u64,
}

/// a field's terms in a COMPACT representation: instead of ~1M separate `String`s
/// (each with heap overhead), they are stored concatenated in a single byte buffer
/// with parallel arrays of offsets and infos. Cuts RAM sharply (less allocator
/// overhead) while keeping O(log N) binary search. ZSL writes terms grouped by field
/// and sorted by text, so the buffer stays sorted.
#[derive(Default)]
struct FieldTerms {
    /// term texts concatenated, in ascending order.
    text: Vec<u8>,
    /// n+1 offsets: term i is `text[offsets[i]..offsets[i+1]]`.
    offsets: Vec<u32>,
    /// n TermInfo, one per term.
    infos: Vec<TermInfo>,
}

impl FieldTerms {
    fn len(&self) -> usize {
        self.infos.len()
    }
    fn term(&self, i: usize) -> &[u8] {
        &self.text[self.offsets[i] as usize..self.offsets[i + 1] as usize]
    }
    /// zero-copy `&str` view of term `i`. `text` is built exclusively from
    /// `String`s (`read_modified_utf8` decodes into a real `String`, never raw
    /// bytes) and `offsets` only ever land on whole-term boundaries, so every
    /// slice is guaranteed valid UTF-8 — no lossy/alloc path needed here,
    /// unlike `iter_terms`'s owned `String::from_utf8_lossy`.
    fn term_str(&self, i: usize) -> &str {
        std::str::from_utf8(self.term(i))
            .expect("FieldTerms invariant violated: term bytes are valid UTF-8")
    }
}

pub struct TermDict {
    by_field: std::collections::HashMap<String, FieldTerms>,
}

impl TermDict {
    /// Decodes the `.tis` (format 2.1+, marker 0xFFFFFFFD).
    ///
    /// Each entry shares a prefix with the previous term ONLY over the
    /// text (never over the concatenated field name): `SegmentWriter::_dumpTermDictEntry`
    /// writes `prefixLength=0` whenever the field number changes, so
    /// applying the shared prefix always over the previous text (without
    /// distinguishing field) reproduces exactly what `DictionaryLoader::load` does.
    /// The freq/prox pointers are deltas accumulated across the WHOLE file,
    /// not per field.
    pub fn read(tis: &[u8], field_names: &[String]) -> std::io::Result<TermDict> {
        let mut pos = 0usize;
        let _marker = read_i32_be(tis, &mut pos)?;
        let term_count = read_u64_be(tis, &mut pos)?;
        let _index_interval = read_i32_be(tis, &mut pos)?;
        let _skip_interval = read_i32_be(tis, &mut pos)?;
        let _max_skip_levels = read_i32_be(tis, &mut pos)?;

        let mut by_field: std::collections::HashMap<String, FieldTerms> =
            std::collections::HashMap::new();
        let mut prev_text = String::new();
        let mut freq_ptr: u64 = 0;
        let mut prox_ptr: u64 = 0;
        for _ in 0..term_count {
            let shared = read_vint(tis, &mut pos)? as usize;
            let suffix = read_modified_utf8(tis, &mut pos)?;
            let field_num = read_vint(tis, &mut pos)? as usize;
            let doc_freq = read_vint(tis, &mut pos)? as u32;
            freq_ptr = freq_ptr.wrapping_add(read_vint(tis, &mut pos)?);
            prox_ptr = prox_ptr.wrapping_add(read_vint(tis, &mut pos)?);
            // skipOffset is always omitted: skips are disabled (docFreq < skipInterval).

            let prefix: String = prev_text.chars().take(shared).collect();
            let text = format!("{prefix}{suffix}");

            let field = field_names.get(field_num).cloned().unwrap_or_default();
            let ft = by_field.entry(field).or_default();
            ft.offsets.push(ft.text.len() as u32);
            ft.text.extend_from_slice(text.as_bytes());
            ft.infos.push(TermInfo {
                doc_freq,
                freq_pointer: freq_ptr,
                prox_pointer: prox_ptr,
            });

            prev_text = text;
        }
        // final sentinel offset per field so the last term can be sliced.
        for ft in by_field.values_mut() {
            ft.offsets.push(ft.text.len() as u32);
        }
        Ok(TermDict { by_field })
    }

    /// Looks up (field, term) by binary search over the field's compact buffer.
    pub fn info(&self, field: &str, term: &str) -> Option<&TermInfo> {
        let ft = self.by_field.get(field)?;
        let target = term.as_bytes();
        let (mut lo, mut hi) = (0usize, ft.len());
        while lo < hi {
            let mid = (lo + hi) / 2;
            match ft.term(mid).cmp(target) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(&ft.infos[mid]),
            }
        }
        None
    }

    /// Terms of `field` that start with `prefix`. Locates the range start by
    /// binary search and walks until a term stops matching the prefix.
    pub fn terms_with_prefix(&self, field: &str, prefix: &str) -> Vec<String> {
        let ft = match self.by_field.get(field) {
            Some(f) => f,
            None => return Vec::new(),
        };
        let pb = prefix.as_bytes();
        // partition_point: first index with term(i) >= prefix.
        let (mut lo, mut hi) = (0usize, ft.len());
        while lo < hi {
            let mid = (lo + hi) / 2;
            if ft.term(mid) < pb {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let mut out = Vec::new();
        let mut i = lo;
        while i < ft.len() {
            let t = ft.term(i);
            if t.starts_with(pb) {
                out.push(String::from_utf8_lossy(t).into_owned());
                i += 1;
            } else {
                break;
            }
        }
        out
    }

    /// lazy cursor over every `(field, term)` pair in ZSL canonical order
    /// (`fieldName · \0 · text`, i.e. field names ascending, terms ascending
    /// within each field). Unlike `iter_terms`, never materializes a `Vec` of
    /// all terms up front — the k-way streaming merge walks this instead.
    pub fn cursor(&self) -> TermCursor<'_> {
        TermCursor::new(self)
    }

    /// Enumerates ALL terms as `(field, text)`. Order not guaranteed (grouped by
    /// field, each field ascending). Used by the merge to walk each source segment's
    /// terms and copy their postings via `positions_all(field, text)`.
    pub fn iter_terms(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for (field, ft) in &self.by_field {
            for i in 0..ft.len() {
                out.push((
                    field.clone(),
                    String::from_utf8_lossy(ft.term(i)).into_owned(),
                ));
            }
        }
        out
    }
}

/// One sampled entry of the sparse `.tii` index: a full term (text + `TermInfo`)
/// plus the `.tis` byte offset from which sequential scanning must resume to
/// reach it (see `LazyTermDict::open`'s module-level accumulation note).
pub struct IndexEntry {
    pub field: String,
    pub text: String,
    pub info: TermInfo,
    pub tis_offset: usize,
}

/// Lazily-queryable term dictionary: holds the raw `.tis` bytes uninterpreted
/// and only the SPARSE `.tii` index parsed up front (a small fraction of the
/// terms). Unlike the eager `TermDict`, opening this does not walk every `.tis`
/// entry — full term lookups/iteration (Tasks 2-4) will scan `.tis` starting
/// from the nearest `IndexEntry`.
// `tis`/`by_field`/`field_names` are scaffolding for now: this task only builds
// and stores them (`open` + the test-only `index_by_field` accessor); the
// query/iteration ops that read `tis`/`field_names` land in Tasks 2-4.
#[allow(dead_code)]
pub struct LazyTermDict {
    tis: Vec<u8>,
    by_field: std::collections::HashMap<String, Vec<IndexEntry>>,
    field_names: Vec<String>,
}

impl LazyTermDict {
    /// Decodes the `.tii` sparse index (mirrors `TermDict::read`'s header handling,
    /// plus the synthetic first entry ZSL always writes — see `zsl/writer/terms.rs`)
    /// and stashes a copy of `.tis` for later on-demand decoding (Tasks 2-4).
    ///
    /// `.tii` layout: 24-byte header (same shape as `.tis`) + one synthetic entry
    /// (VInt prefix=0, empty String suffix, RAW Int32 field=0xFFFFFFFF, literal byte
    /// 0x0F, VInt docFreq=0, VInt freqDelta=0, VInt proxDelta=0, VInt IndexDelta=24 —
    /// the `.tis` offset of the first real term, right after its 24-byte header) +
    /// N real entries, each like a `.tis` entry but with an extra trailing
    /// VInt IndexDelta.
    ///
    /// The IndexDelta is a delta of the `.tis` byte offset from which sequential
    /// scanning must resume to reach the NEXT indexed term (not necessarily that
    /// term's own start offset when several un-indexed terms lie in between) —
    /// this mirrors the writer's `last_index_position`/`index_position` bookkeeping
    /// in `TermDictStreamWriter::dump_tis_entry`. The synthetic entry's IndexDelta
    /// (24) seeds this accumulator for the first real entry.
    pub fn open(tis: &[u8], tii: &[u8], field_names: &[String]) -> std::io::Result<LazyTermDict> {
        let mut pos = 0usize;
        // header (same 24 bytes as .tis)
        let _marker = read_i32_be(tii, &mut pos)?;
        let _index_count = read_u64_be(tii, &mut pos)?;
        let _index_interval = read_i32_be(tii, &mut pos)?;
        let _skip_interval = read_i32_be(tii, &mut pos)?;
        let _max_skip_levels = read_i32_be(tii, &mut pos)?;

        // synthetic first entry: VInt prefix, String suffix, RAW Int32 field (0xFFFFFFFF),
        // literal byte 0x0F, VInt docFreq, VInt freqDelta, VInt proxDelta, VInt IndexDelta.
        let _pfx = read_vint(tii, &mut pos)?;
        let _suf = read_modified_utf8(tii, &mut pos)?;
        let _raw_field = read_i32_be(tii, &mut pos)?; // raw Int, not VInt
                                                      // consume the literal 0x0F marker byte:
        if pos >= tii.len() {
            return Err(std::io::Error::other("tii: truncated synthetic entry"));
        }
        pos += 1;
        let _df = read_vint(tii, &mut pos)?;
        let _fd = read_vint(tii, &mut pos)?;
        let _pd = read_vint(tii, &mut pos)?;
        let mut tis_offset = read_vint(tii, &mut pos)? as usize; // = 24 (first .tis term)

        // real index entries, prefix/pointers accumulate across entries.
        let mut by_field: std::collections::HashMap<String, Vec<IndexEntry>> = Default::default();
        let mut prev_text = String::new();
        let (mut freq_ptr, mut prox_ptr) = (0u64, 0u64);
        while pos < tii.len() {
            let shared = read_vint(tii, &mut pos)? as usize;
            let suffix = read_modified_utf8(tii, &mut pos)?;
            let field_num = read_vint(tii, &mut pos)? as usize;
            let doc_freq = read_vint(tii, &mut pos)? as u32;
            freq_ptr = freq_ptr.wrapping_add(read_vint(tii, &mut pos)?);
            prox_ptr = prox_ptr.wrapping_add(read_vint(tii, &mut pos)?);
            let index_delta = read_vint(tii, &mut pos)? as usize;
            let prefix: String = prev_text.chars().take(shared).collect();
            let text = format!("{prefix}{suffix}");
            let field = field_names.get(field_num).cloned().unwrap_or_default();
            by_field.entry(field.clone()).or_default().push(IndexEntry {
                field,
                text: text.clone(),
                info: TermInfo {
                    doc_freq,
                    freq_pointer: freq_ptr,
                    prox_pointer: prox_ptr,
                },
                tis_offset,
            });
            prev_text = text;
            tis_offset = tis_offset.wrapping_add(index_delta);
        }
        // within a field, index entries are ascending by construction (.tis order); keep as-is.
        Ok(LazyTermDict {
            tis: tis.to_vec(),
            by_field,
            field_names: field_names.to_vec(),
        })
    }

    #[cfg(test)]
    pub fn index_by_field(&self) -> &std::collections::HashMap<String, Vec<IndexEntry>> {
        &self.by_field
    }
}

/// Lazy cursor over all `(field, term)` pairs of a `TermDict`, in ZSL
/// canonical order (field names sorted ascending, terms ascending within each
/// field — equivalent to sorting `(fieldName, text)` tuples since `\0` is the
/// minimum byte and never appears inside a field name or term). Backed
/// directly by each field's already-sorted `FieldTerms` buffer: no `Vec` of
/// all terms is built up front, so memory stays bounded regardless of how
/// many terms the segment holds.
pub struct TermCursor<'a> {
    dict: &'a TermDict,
    fields: Vec<&'a str>,
    field_idx: usize,
    term_idx: usize,
}

impl<'a> TermCursor<'a> {
    fn new(dict: &'a TermDict) -> TermCursor<'a> {
        let mut fields: Vec<&str> = dict.by_field.keys().map(String::as_str).collect();
        fields.sort_unstable();
        TermCursor {
            dict,
            fields,
            field_idx: 0,
            term_idx: 0,
        }
    }

    /// current `(field, term)` pair, or `None` once every field is exhausted.
    pub fn peek(&self) -> Option<(&str, &str)> {
        let field = *self.fields.get(self.field_idx)?;
        let ft = self
            .dict
            .by_field
            .get(field)
            .expect("field came from this dict's own key set");
        Some((field, ft.term_str(self.term_idx)))
    }

    /// moves to the next pair in canonical order. No-op once exhausted.
    pub fn advance(&mut self) {
        if self.field_idx >= self.fields.len() {
            return;
        }
        self.term_idx += 1;
        // `by_field` only ever holds fields with >=1 term (a field is inserted
        // lazily, on its first term, while reading `.tis`), so the next field
        // (if any) is guaranteed non-empty — no need to loop-skip empties.
        let current_len = self.dict.by_field[self.fields[self.field_idx]].len();
        if self.term_idx >= current_len {
            self.field_idx += 1;
            self.term_idx = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zsl::cfs::CompoundFile;
    use crate::zsl::fields::read_field_infos;
    use std::path::PathBuf;

    fn dict() -> TermDict {
        let dir = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/zsl_index"
        ));
        let path = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .find(|p| p.extension().map(|x| x == "cfs").unwrap_or(false))
            .unwrap();
        let cf = CompoundFile::open(&path).unwrap();
        let fnm = cf
            .names()
            .into_iter()
            .find(|n| n.ends_with(".fnm"))
            .unwrap();
        let tis = cf
            .names()
            .into_iter()
            .find(|n| n.ends_with(".tis"))
            .unwrap();
        let names: Vec<String> = read_field_infos(cf.sub(&fnm).unwrap())
            .unwrap()
            .into_iter()
            .map(|f| f.name)
            .collect();
        TermDict::read(cf.sub(&tis).unwrap(), &names).unwrap()
    }

    #[test]
    fn termdict_read_errors_on_truncation() {
        // marker(4) + termCount(8)=5 + intervals(12), but no term bodies follow
        let mut buf = vec![0xFF, 0xFF, 0xFF, 0xFD];
        buf.extend_from_slice(&5u64.to_be_bytes());
        buf.extend_from_slice(&[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]);
        assert!(TermDict::read(&buf, &[]).is_err());
    }

    #[test]
    fn finds_known_term_with_docfreq() {
        let d = dict();
        // incidents index: all 4 docs are titled "New workflow" => "new" in title, docFreq 4
        let ti = d.info("title", "new").expect("term new missing");
        assert_eq!(ti.doc_freq, 4);
    }

    #[test]
    fn prefix_scan_returns_sorted_terms() {
        let d = dict();
        // title terms = {new, workflow}; prefix "ne" => only "new"
        let mut ne = d.terms_with_prefix("title", "ne");
        ne.sort();
        assert_eq!(ne, vec!["new"]);
    }

    #[test]
    fn iter_terms_enumerates_all_field_text_pairs() {
        let d = dict(); // incidents fixture: title={new, workflow}
        let mut got = d.iter_terms();
        got.sort();
        // the incidents fixture has 4 docs "New workflow" + fields id_key/users/etc.
        // minimal stable assertion: title:new and title:workflow are present.
        assert!(
            got.contains(&("title".to_string(), "new".to_string())),
            "got={got:?}"
        );
        assert!(
            got.contains(&("title".to_string(), "workflow".to_string())),
            "got={got:?}"
        );
        // total count = sum of terms per field (non-empty)
        assert!(!got.is_empty());
    }

    #[cfg(test)]
    fn fixture_dict_bytes() -> (Vec<u8>, Vec<u8>, Vec<String>) {
        use crate::zsl::cfs::CompoundFile;
        use crate::zsl::fields::read_field_infos;
        let dir = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/zsl_index"
        ));
        let path = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .find(|p| p.extension().map(|x| x == "cfs").unwrap_or(false))
            .unwrap();
        let cf = CompoundFile::open(&path).unwrap();
        let find = |ext: &str| cf.names().into_iter().find(|n| n.ends_with(ext)).unwrap();
        let names: Vec<String> = read_field_infos(cf.sub(&find(".fnm")).unwrap())
            .unwrap()
            .into_iter()
            .map(|f| f.name)
            .collect();
        (
            cf.sub(&find(".tis")).unwrap().to_vec(),
            cf.sub(&find(".tii")).unwrap().to_vec(),
            names,
        )
    }

    #[test]
    fn tii_index_terms_are_a_correct_sparse_subset_of_tis() {
        // KB fixture has few terms → sparse index may hold only field-firsts; use the assertions
        // that hold regardless of size. (A larger differential is exercised in Task 2.)
        let (tis, tii, names) = fixture_dict_bytes(); // helper added below
        let lazy = LazyTermDict::open(&tis, &tii, &names).unwrap();
        let eager = TermDict::read(&tis, &names).unwrap();
        // every sparse index entry must be a real term of the eager dict with identical TermInfo,
        // and its recorded tis_offset must decode (in Task 2) to that same term.
        for (field, entries) in lazy.index_by_field() {
            for e in entries {
                let ti = eager
                    .info(field, &e.text)
                    .unwrap_or_else(|| panic!("index term {field}:{} absent in eager", e.text));
                assert_eq!(*ti, e.info, "TermInfo mismatch for {field}:{}", e.text);
            }
        }
        // the synthetic sentinel (empty text, field 0xFFFFFFFF) must NOT appear as an index entry.
        assert!(lazy
            .index_by_field()
            .values()
            .flatten()
            .all(|e| !e.text.is_empty()));
    }
}
