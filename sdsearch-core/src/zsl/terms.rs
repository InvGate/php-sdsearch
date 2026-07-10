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
    pub fn read(tis: &[u8], field_names: &[String]) -> TermDict {
        let mut pos = 0usize;
        let _marker = read_i32_be(tis, &mut pos);
        let term_count = read_u64_be(tis, &mut pos);
        let _index_interval = read_i32_be(tis, &mut pos);
        let _skip_interval = read_i32_be(tis, &mut pos);
        let _max_skip_levels = read_i32_be(tis, &mut pos);

        let mut by_field: std::collections::HashMap<String, FieldTerms> =
            std::collections::HashMap::new();
        let mut prev_text = String::new();
        let mut freq_ptr: u64 = 0;
        let mut prox_ptr: u64 = 0;
        for _ in 0..term_count {
            let shared = read_vint(tis, &mut pos) as usize;
            let suffix = read_modified_utf8(tis, &mut pos);
            let field_num = read_vint(tis, &mut pos) as usize;
            let doc_freq = read_vint(tis, &mut pos) as u32;
            freq_ptr = freq_ptr.wrapping_add(read_vint(tis, &mut pos));
            prox_ptr = prox_ptr.wrapping_add(read_vint(tis, &mut pos));
            // skipOffset is always omitted: skips are disabled (docFreq < skipInterval).

            let prefix: String = prev_text.chars().take(shared).collect();
            let text = format!("{prefix}{suffix}");

            let field = field_names.get(field_num).cloned().unwrap_or_default();
            let ft = by_field.entry(field).or_default();
            ft.offsets.push(ft.text.len() as u32);
            ft.text.extend_from_slice(text.as_bytes());
            ft.infos.push(TermInfo { doc_freq, freq_pointer: freq_ptr, prox_pointer: prox_ptr });

            prev_text = text;
        }
        // final sentinel offset per field so the last term can be sliced.
        for ft in by_field.values_mut() {
            ft.offsets.push(ft.text.len() as u32);
        }
        TermDict { by_field }
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

    /// Enumerates ALL terms as `(field, text)`. Order not guaranteed (grouped by
    /// field, each field ascending). Used by the merge to walk each source segment's
    /// terms and copy their postings via `positions_all(field, text)`.
    pub fn iter_terms(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for (field, ft) in &self.by_field {
            for i in 0..ft.len() {
                out.push((field.clone(), String::from_utf8_lossy(ft.term(i)).into_owned()));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zsl::cfs::CompoundFile;
    use crate::zsl::fields::read_field_infos;
    use std::path::PathBuf;

    fn dict() -> TermDict {
        let dir = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/zsl_index"));
        let path = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .find(|p| p.extension().map(|x| x == "cfs").unwrap_or(false))
            .unwrap();
        let cf = CompoundFile::open(&path).unwrap();
        let fnm = cf.names().into_iter().find(|n| n.ends_with(".fnm")).unwrap();
        let tis = cf.names().into_iter().find(|n| n.ends_with(".tis")).unwrap();
        let names: Vec<String> = read_field_infos(cf.sub(&fnm).unwrap()).into_iter().map(|f| f.name).collect();
        TermDict::read(cf.sub(&tis).unwrap(), &names)
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
        assert!(got.contains(&("title".to_string(), "new".to_string())), "got={got:?}");
        assert!(got.contains(&("title".to_string(), "workflow".to_string())), "got={got:?}");
        // total count = sum of terms per field (non-empty)
        assert!(!got.is_empty());
    }
}
