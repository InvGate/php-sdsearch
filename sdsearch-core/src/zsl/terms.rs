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
///
/// `#[cfg(test)]`: the eager full-parse dictionary is kept ONLY as the
/// differential oracle for the lazy production `TermDict` (see module tests);
/// production never compiles it.
#[cfg(test)]
#[derive(Default)]
struct EagerFieldTerms {
    /// term texts concatenated, in ascending order.
    text: Vec<u8>,
    /// n+1 offsets: term i is `text[offsets[i]..offsets[i+1]]`.
    offsets: Vec<u32>,
    /// n TermInfo, one per term.
    infos: Vec<TermInfo>,
}

#[cfg(test)]
impl EagerFieldTerms {
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
            .expect("EagerFieldTerms invariant violated: term bytes are valid UTF-8")
    }
}

/// Eager full-parse term dictionary, kept ONLY as a `#[cfg(test)]` differential
/// oracle: it walks the ENTIRE `.tis` up front into per-field compact buffers.
/// Production uses the lazy `.tii`-backed `TermDict` below; the module tests
/// assert the two agree on every query over the fixtures and a synthetic corpus.
#[cfg(test)]
pub struct EagerTermDict {
    by_field: std::collections::HashMap<String, EagerFieldTerms>,
}

#[cfg(test)]
impl EagerTermDict {
    /// Decodes the `.tis` (format 2.1+, marker 0xFFFFFFFD).
    ///
    /// Each entry shares a prefix with the previous term ONLY over the
    /// text (never over the concatenated field name): `SegmentWriter::_dumpTermDictEntry`
    /// writes `prefixLength=0` whenever the field number changes, so
    /// applying the shared prefix always over the previous text (without
    /// distinguishing field) reproduces exactly what `DictionaryLoader::load` does.
    /// The freq/prox pointers are deltas accumulated across the WHOLE file,
    /// not per field.
    pub fn read(tis: &[u8], field_names: &[String]) -> std::io::Result<EagerTermDict> {
        let mut pos = 0usize;
        let _marker = read_i32_be(tis, &mut pos)?;
        let term_count = read_u64_be(tis, &mut pos)?;
        let _index_interval = read_i32_be(tis, &mut pos)?;
        let _skip_interval = read_i32_be(tis, &mut pos)?;
        let _max_skip_levels = read_i32_be(tis, &mut pos)?;

        let mut by_field: std::collections::HashMap<String, EagerFieldTerms> =
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
        Ok(EagerTermDict { by_field })
    }

    /// Looks up (field, term) by binary search over the field's compact buffer.
    pub fn info(&self, field: &str, term: &str) -> Option<TermInfo> {
        let ft = self.by_field.get(field)?;
        let target = term.as_bytes();
        let (mut lo, mut hi) = (0usize, ft.len());
        while lo < hi {
            let mid = (lo + hi) / 2;
            match ft.term(mid).cmp(target) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(ft.infos[mid].clone()),
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
    pub fn cursor(&self) -> EagerTermCursor<'_> {
        EagerTermCursor::new(self)
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
/// reach it (see `TermDict::open`'s module-level accumulation note).
pub struct IndexEntry {
    pub field: String,
    pub text: String,
    pub info: TermInfo,
    pub tis_offset: usize,
}

/// Lazily-queryable term dictionary: holds the raw `.tis` bytes uninterpreted
/// and only the SPARSE `.tii` index parsed up front (a small fraction of the
/// terms), as a single GLOBAL list (not grouped by field). Unlike the eager
/// `EagerTermDict`, opening this does not walk every `.tis` entry — `info` seeks to
/// the nearest `IndexEntry` and scans forward from there.
///
/// `.tis`/`.tii` are physically ordered by the composite key `(field_name
/// ascending, text ascending)` (the eager `EagerTermCursor` / writer guarantee
/// this), so `index` is already sorted by that key and can be binary-searched
/// with `partition_point`. `index[0]` is a synthetic anchor `("", "")` — see
/// `open` — so every real target has a well-defined predecessor entry.
pub struct TermDict {
    tis: Vec<u8>,
    index: Vec<IndexEntry>,
    field_names: Vec<String>,
}

impl TermDict {
    /// Decodes the `.tii` sparse index (mirrors `EagerTermDict::read`'s header handling,
    /// plus the synthetic first entry ZSL always writes — see `zsl/writer/terms.rs`)
    /// and stashes a copy of `.tis` for later on-demand decoding by `info`.
    ///
    /// `.tii` layout: 24-byte header (same shape as `.tis`) + one synthetic entry
    /// (VInt prefix=0, empty String suffix, RAW Int32 field=0xFFFFFFFF, literal byte
    /// 0x0F, VInt docFreq=0, VInt freqDelta=0, VInt proxDelta=0, VInt IndexDelta=24 —
    /// the `.tis` offset of the first real term, right after its 24-byte header) +
    /// N real entries, each like a `.tis` entry but with an extra trailing
    /// VInt IndexDelta.
    ///
    /// CRITICAL: per `zsl/writer/terms.rs::dump_tis_entry`, each entry's IndexDelta
    /// is captured from `index_position = self.tis_len` AFTER the sampled term's
    /// `.tis` entry was written — so an accumulated `tis_offset` points to the
    /// `.tis` byte position IMMEDIATELY AFTER the indexed term (the start of
    /// whatever term follows it), not the indexed term's own start. `info` relies on
    /// this: after an exact hit on an index entry it can only resume decoding from
    /// `tis_offset`, seeded with that entry's own text/pointers as predecessor state.
    /// `INDEX_INTERVAL` also counts GLOBALLY across all fields, so a field's first
    /// term may have no preceding `.tii` sample of its own — the synthetic anchor
    /// (`tis_offset = 24`, the first real `.tis` term's start) covers that case.
    pub fn open(tis: &[u8], tii: &[u8], field_names: &[String]) -> std::io::Result<TermDict> {
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
        let _synthetic_index_delta = read_vint(tii, &mut pos)? as usize; // = 24

        // synthetic anchor: ("", "") sorts before every real (field, text), and its
        // tis_offset (24) is the .tis byte position of the first real term, right
        // after the 24-byte header.
        let mut index = vec![IndexEntry {
            field: String::new(),
            text: String::new(),
            info: TermInfo {
                doc_freq: 0,
                freq_pointer: 0,
                prox_pointer: 0,
            },
            tis_offset: 24,
        }];
        let mut prev_text = String::new();
        let (mut freq_ptr, mut prox_ptr) = (0u64, 0u64);
        let mut running = 24usize; // synthetic IndexDelta already consumed above == 24
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
            running = running.wrapping_add(index_delta); // ADD FIRST …
            index.push(IndexEntry {
                // … THEN assign tis_offset = running
                field,
                text: text.clone(),
                info: TermInfo {
                    doc_freq,
                    freq_pointer: freq_ptr,
                    prox_pointer: prox_ptr,
                },
                tis_offset: running,
            });
            prev_text = text;
        }
        Ok(TermDict {
            tis: tis.to_vec(),
            index,
            field_names: field_names.to_vec(),
        })
    }

    /// Test-only: number of parsed `.tii` index entries (including the synthetic
    /// `("","")` anchor). Lets tests assert a synthesized corpus actually produced
    /// more than the trivial single-anchor index, i.e. exercised the sparse seek
    /// path in `info` rather than degenerating to a full `.tis` scan from offset 24.
    #[cfg(test)]
    pub fn index_len(&self) -> usize {
        self.index.len()
    }

    /// Looks up `(field, term)` by seeking to the nearest `.tii` index entry
    /// at or before it (composite key `(field, text)`) and, unless that's an
    /// exact hit, forward-decoding `.tis` from there until the target is found,
    /// passed, or the file ends.
    pub fn info(&self, field: &str, term: &str) -> Option<TermInfo> {
        let key = (field, term);
        // largest index entry with (field, text) <= key. index[0] is the synthetic ("","") anchor,
        // so partition_point is always >= 1.
        let gt = self
            .index
            .partition_point(|e| (e.field.as_str(), e.text.as_str()) <= key);
        let anchor = &self.index[gt - 1];
        if !anchor.text.is_empty() && anchor.field == field && anchor.text == term {
            return Some(anchor.info.clone()); // exact hit on an index term (no .tis scan)
        }
        // scan forward from the anchor's tis_offset (already PAST the anchor's own term), seeded
        // with the anchor's text/pointers as the predecessor state.
        let mut pos = anchor.tis_offset;
        let mut prev = anchor.text.clone();
        let (mut fp, mut pp) = (anchor.info.freq_pointer, anchor.info.prox_pointer);
        while pos < self.tis.len() {
            let (f, t, ti) = decode_entry(
                &self.tis,
                &mut pos,
                &prev,
                &mut fp,
                &mut pp,
                &self.field_names,
            )
            .ok()?;
            match (f.as_str(), t.as_str()).cmp(&key) {
                std::cmp::Ordering::Equal => return Some(ti),
                std::cmp::Ordering::Greater => return None, // passed the key in canonical order ⇒ absent
                std::cmp::Ordering::Less => prev = t,
            }
        }
        None
    }

    /// Terms of `field` that start with `prefix`, matching the eager
    /// `EagerTermDict::terms_with_prefix` exactly. Finds the `.tii` anchor at or before
    /// `(field, prefix)` in canonical `(field_name, text)` order, seeks its
    /// `tis_offset`, and forward-decodes `.tis` from there — collecting matches of
    /// `field` until the field changes or the text passes the prefix range.
    ///
    /// The anchor may sit in an EARLIER field than `field` (e.g. when `field`'s own
    /// first term precedes any `.tii` sample taken from it): entries with `f < field`
    /// are skipped, `f == field` entries are collected/filtered by prefix, and the
    /// scan stops as soon as `f > field` (canonical order guarantees nothing further
    /// can belong to `field`).
    pub fn terms_with_prefix(&self, field: &str, prefix: &str) -> Vec<String> {
        let key = (field, prefix);
        // index[0] is the synthetic ("","") anchor ⇒ partition_point is always >= 1.
        let gt = self
            .index
            .partition_point(|e| (e.field.as_str(), e.text.as_str()) <= key);
        let anchor = &self.index[gt - 1];
        let mut out = Vec::new();
        // the anchor itself is a real term that may already match (e.g. prefix == an index term).
        if !anchor.text.is_empty() && anchor.field == field && anchor.text.starts_with(prefix) {
            out.push(anchor.text.clone());
        }
        let mut pos = anchor.tis_offset;
        let mut prev = anchor.text.clone();
        let (mut fp, mut pp) = (anchor.info.freq_pointer, anchor.info.prox_pointer);
        while pos < self.tis.len() {
            let Ok((f, t, _)) = decode_entry(
                &self.tis,
                &mut pos,
                &prev,
                &mut fp,
                &mut pp,
                &self.field_names,
            ) else {
                break;
            };
            // canonical order is (field_name, text); stop once we pass this field.
            if f.as_str() > field {
                break;
            }
            if f == field {
                if t.starts_with(prefix) {
                    out.push(t.clone());
                } else if t.as_str() > prefix {
                    break; // past the range in-field
                }
            }
            prev = t;
        }
        out
    }

    /// Sequentially decodes the WHOLE `.tis` from just past its 24-byte header.
    /// `.tis` is physically written in canonical `(field_name asc, text asc)`
    /// order (see `EagerTermCursor`'s doc comment and `zsl/writer/terms.rs`), so a
    /// plain top-to-bottom decode already yields that order — no sorting or
    /// per-field grouping needed, unlike the eager `EagerTermDict::iter_terms`
    /// (which enumerates a `HashMap` and so has no ordering guarantee of its
    /// own). Used by both `iter_terms` and `cursor`.
    fn decode_all(&self) -> Vec<(String, String)> {
        let mut pos = 24usize; // past the header
        let mut prev = String::new();
        let (mut fp, mut pp) = (0u64, 0u64);
        let mut out = Vec::new();
        while pos < self.tis.len() {
            match decode_entry(
                &self.tis,
                &mut pos,
                &prev,
                &mut fp,
                &mut pp,
                &self.field_names,
            ) {
                Ok((f, t, _)) => {
                    prev = t.clone();
                    out.push((f, t));
                }
                Err(_) => break,
            }
        }
        out
    }

    /// Enumerates ALL terms as `(field, text)`, in canonical `.tis` order (see
    /// `decode_all`). Used by the merge to walk each source segment's terms
    /// and copy their postings via `positions_all(field, text)`.
    pub fn iter_terms(&self) -> Vec<(String, String)> {
        self.decode_all()
    }

    /// Cursor over every `(field, term)` pair in ZSL canonical order,
    /// matching the eager `EagerTermCursor`'s semantics exactly (the merge relies
    /// on this order to interleave sources correctly). Unlike `EagerTermCursor`,
    /// this OWNS its decoded `Vec` instead of borrowing the dict — it's built
    /// from raw `.tis` bytes rather than an already-grouped `HashMap`, and the
    /// merge only ever needs `peek`/`advance`, so no lifetime is required.
    pub fn cursor(&self) -> TermCursor {
        TermCursor {
            terms: self.decode_all(),
            i: 0,
        }
    }
}

/// Owned cursor over a `TermDict`'s terms in canonical order (see
/// `TermDict::cursor`). Pre-decoded at construction time since the merge
/// reads every entry anyway, so there's no benefit to decoding lazily on
/// `advance`.
pub struct TermCursor {
    terms: Vec<(String, String)>,
    i: usize,
}

impl TermCursor {
    /// current `(field, term)` pair, or `None` once every pair is exhausted.
    pub fn peek(&self) -> Option<(&str, &str)> {
        self.terms
            .get(self.i)
            .map(|(f, t)| (f.as_str(), t.as_str()))
    }

    /// moves to the next pair in canonical order. No-op once exhausted.
    pub fn advance(&mut self) {
        if self.i < self.terms.len() {
            self.i += 1;
        }
    }
}

/// Decodes a single `.tis` entry starting at `*pos`, advancing it past the entry.
/// Mirrors `EagerTermDict::read`'s per-entry decoding (shared prefix taken over the
/// previous TEXT only, never across field boundaries — see that function's doc
/// comment) but for exactly one entry, so `TermDict::info` can resume a scan
/// from any `.tii`-indexed offset instead of decoding the whole `.tis`.
fn decode_entry(
    tis: &[u8],
    pos: &mut usize,
    prev_text: &str,
    freq_ptr: &mut u64,
    prox_ptr: &mut u64,
    field_names: &[String],
) -> std::io::Result<(String, String, TermInfo)> {
    let shared = read_vint(tis, pos)? as usize;
    let suffix = read_modified_utf8(tis, pos)?;
    let field_num = read_vint(tis, pos)? as usize;
    let doc_freq = read_vint(tis, pos)? as u32;
    *freq_ptr = freq_ptr.wrapping_add(read_vint(tis, pos)?);
    *prox_ptr = prox_ptr.wrapping_add(read_vint(tis, pos)?);
    let prefix: String = prev_text.chars().take(shared).collect();
    let text = format!("{prefix}{suffix}");
    let field = field_names.get(field_num).cloned().unwrap_or_default();
    Ok((
        field,
        text,
        TermInfo {
            doc_freq,
            freq_pointer: *freq_ptr,
            prox_pointer: *prox_ptr,
        },
    ))
}

/// Cursor over all `(field, term)` pairs of an `EagerTermDict`, in ZSL
/// canonical order (field names sorted ascending, terms ascending within each
/// field — equivalent to sorting `(fieldName, text)` tuples since `\0` is the
/// minimum byte and never appears inside a field name or term). Backed
/// directly by each field's already-sorted `EagerFieldTerms` buffer. Kept as a
/// `#[cfg(test)]` oracle for the production lazy `TermCursor`.
#[cfg(test)]
pub struct EagerTermCursor<'a> {
    dict: &'a EagerTermDict,
    fields: Vec<&'a str>,
    field_idx: usize,
    term_idx: usize,
}

#[cfg(test)]
impl<'a> EagerTermCursor<'a> {
    fn new(dict: &'a EagerTermDict) -> EagerTermCursor<'a> {
        let mut fields: Vec<&str> = dict.by_field.keys().map(String::as_str).collect();
        fields.sort_unstable();
        EagerTermCursor {
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

    fn dict() -> EagerTermDict {
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
        EagerTermDict::read(cf.sub(&tis).unwrap(), &names).unwrap()
    }

    #[test]
    fn termdict_read_errors_on_truncation() {
        // marker(4) + termCount(8)=5 + intervals(12), but no term bodies follow
        let mut buf = vec![0xFF, 0xFF, 0xFF, 0xFD];
        buf.extend_from_slice(&5u64.to_be_bytes());
        buf.extend_from_slice(&[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]);
        assert!(EagerTermDict::read(&buf, &[]).is_err());
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

    fn fixture_dict_bytes_multiseg() -> (Vec<u8>, Vec<u8>, Vec<String>) {
        use crate::zsl::cfs::CompoundFile;
        use crate::zsl::fields::read_field_infos;
        let dir = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/zsl_index_multiseg"
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
    fn lazy_info_matches_eager_for_every_term_and_absentees() {
        for (tis, tii, names) in [fixture_dict_bytes(), fixture_dict_bytes_multiseg()] {
            let eager = EagerTermDict::read(&tis, &names).unwrap();
            let lazy = TermDict::open(&tis, &tii, &names).unwrap();
            for (field, text) in eager.iter_terms() {
                assert_eq!(
                    lazy.info(&field, &text),
                    eager.info(&field, &text),
                    "mismatch at {field}:{text}"
                );
            }
            assert_eq!(lazy.info("title", "definitelymissing"), None);
            assert_eq!(lazy.info("no_such_field", "x"), None);
        }
    }

    /// Synthesizes a >128-term, 2-field term dictionary directly via the writer's
    /// batch `write_term_dict` (mirroring `tii_samples_every_index_interval_terms`
    /// / `multi_field_sample_terms` in `zsl/writer/terms.rs`, scaled up to 560 terms —
    /// the same order of magnitude the review used to demonstrate ~429/560 mismatches
    /// when the accumulation order is reverted). Guarantees `index_len() > 1`, i.e.
    /// the `.tii` actually got multiple real samples rather than just the synthetic
    /// `("","")` anchor — the committed fixtures (`zsl_index`, `zsl_index_multiseg`)
    /// both have far fewer than `INDEX_INTERVAL` (128) terms and can't exercise the
    /// sparse seek path on their own.
    ///
    /// field 0 ("body"): 400 terms, zero-padded so lexicographic order == insertion
    /// order, sharing the "shared" prefix (exercises prefix-sharing across many
    /// consecutive .tis/.tii entries). Every 10th term gets a second, far-away doc
    /// (exercises multi-doc freq/prox deltas).
    /// field 1 ("title"): 160 more terms, zero-padded ascending, disjoint doc-id
    /// range from field 0 (prefix sharing must not leak across fields — see
    /// `dump_entry`'s doc comment).
    fn large_dict_bytes() -> (Vec<u8>, Vec<u8>, Vec<String>) {
        use crate::zsl::writer::invert::TermPostings;
        use crate::zsl::writer::terms::write_term_dict;

        let mut terms: Vec<TermPostings> = Vec::new();

        for i in 0..400usize {
            let text = format!("shared{i:04}");
            let docs = if i % 10 == 0 {
                vec![(i, vec![0u32, 3]), (i + 10_000, vec![1u32])]
            } else {
                vec![(i, vec![0u32])]
            };
            terms.push(TermPostings {
                field_num: 0,
                text,
                docs,
            });
        }
        for i in 0..160usize {
            let text = format!("word{i:04}");
            let docs = if i % 7 == 0 {
                vec![(20_000 + i, vec![0u32, 2]), (20_000 + i + 500, vec![1u32])]
            } else {
                vec![(20_000 + i, vec![0u32])]
            };
            terms.push(TermPostings {
                field_num: 1,
                text,
                docs,
            });
        }
        assert_eq!(
            terms.len(),
            560,
            "corpus must comfortably exceed INDEX_INTERVAL (128)"
        );

        let field_names = vec!["body".to_string(), "title".to_string()];
        let dict_files = write_term_dict(&terms);
        (dict_files.tis, dict_files.tii, field_names)
    }

    #[test]
    fn lazy_seek_path_exercised_by_synthetic_multi_sample_tii() {
        let (tis, tii, field_names) = large_dict_bytes();

        let eager = EagerTermDict::read(&tis, &field_names).unwrap();
        let lazy = TermDict::open(&tis, &tii, &field_names).unwrap();

        // sanity: the synthesized corpus must actually produce multiple .tii samples,
        // i.e. more than just the synthetic ("","") anchor — otherwise this test would
        // silently degenerate back into the same full-scan-only coverage as the
        // fixture-based differential test above.
        assert!(
            lazy.index_len() > 1,
            "corpus too small to exercise the sparse .tii seek path: index_len={}",
            lazy.index_len()
        );

        for (field, text) in eager.iter_terms() {
            assert_eq!(
                lazy.info(&field, &text),
                eager.info(&field, &text),
                "mismatch at {field}:{text}"
            );
        }

        assert_eq!(lazy.info("body", "definitelymissing"), None);
        assert_eq!(lazy.info("no_such_field", "x"), None);
    }

    /// for each `(field, prefix)` asserts `lazy.terms_with_prefix` == `eager.terms_with_prefix`
    /// (both sorted, since neither API guarantees an order).
    fn assert_prefix_parity(
        tis: &[u8],
        tii: &[u8],
        names: &[String],
        fields: &[&str],
        prefixes: &[&str],
    ) {
        let eager = EagerTermDict::read(tis, names).unwrap();
        let lazy = TermDict::open(tis, tii, names).unwrap();
        for &field in fields {
            for &pfx in prefixes {
                let mut a = lazy.terms_with_prefix(field, pfx);
                a.sort();
                let mut b = eager.terms_with_prefix(field, pfx);
                b.sort();
                assert_eq!(a, b, "prefix mismatch {field}:{pfx:?}");
            }
        }
    }

    #[test]
    fn lazy_prefix_matches_eager() {
        // small fixtures (fields title/body/id_key) — only the synthetic anchor, no
        // real .tii samples.
        for (tis, tii, names) in [fixture_dict_bytes(), fixture_dict_bytes_multiseg()] {
            assert_prefix_parity(
                &tis,
                &tii,
                &names,
                &["title", "body", "id_key", "no_such_field"],
                &["", "a", "ne", "work", "zzz"],
            );
        }

        // large synthetic corpus (real, multi-sample .tii): "body" has shared0000..shared0399,
        // "title" has word0000..word0159 (INDEX_INTERVAL=128 globally, body written first, so
        // .tii samples land inside "body" at ~terms 128/256/384 and spill into "title" after
        // term 400). Prefixes chosen to hit all three terms_with_prefix branches:
        // "" / "a" precede the first real sample (anchor may be an earlier field or the
        // synthetic ("","") one); "shared" matches broadly within one field; "shared01"
        // (shared0100..shared0199) straddles the ~128 sample boundary; "shared05" is past
        // the last real "body" term (max shared0399) so it matches nothing; "zzz" matches
        // nothing in any field.
        let (tis, tii, names) = large_dict_bytes();
        assert_prefix_parity(
            &tis,
            &tii,
            &names,
            &names.iter().map(String::as_str).collect::<Vec<_>>(),
            &["", "a", "shared", "shared01", "shared05", "zzz"],
        );
    }

    #[test]
    fn lazy_iter_and_cursor_match_eager() {
        for (tis, tii, names) in [
            fixture_dict_bytes(),
            fixture_dict_bytes_multiseg(),
            large_dict_bytes(),
        ] {
            let eager = EagerTermDict::read(&tis, &names).unwrap();
            let lazy = TermDict::open(&tis, &tii, &names).unwrap();

            // iter_terms: same multiset AND, critically for the merge, cursor gives the same ORDER.
            let mut a = lazy.iter_terms();
            a.sort();
            let mut b = eager.iter_terms();
            b.sort();
            assert_eq!(a, b);

            // cursor sequence identical, element by element (canonical .tis order — the merge relies on it)
            let (mut ec, mut lc) = (eager.cursor(), lazy.cursor());
            loop {
                let (e, l) = (
                    ec.peek().map(|(f, t)| (f.to_string(), t.to_string())),
                    lc.peek().map(|(f, t)| (f.to_string(), t.to_string())),
                );
                assert_eq!(e, l);
                if e.is_none() {
                    break;
                }
                ec.advance();
                lc.advance();
            }
        }
    }
}
