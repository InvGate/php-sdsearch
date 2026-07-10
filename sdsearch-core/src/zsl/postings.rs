//! ZSL postings decode: .frq (doc deltas + freqs) and .prx (positions).
use crate::zsl::bytes::read_vint;
use crate::zsl::terms::TermInfo;

/// Decodes `.frq` for a term: docId (accumulated delta) + freq per doc.
/// `VInt(v)`: `docDelta = v >> 1`; if `v & 1 == 1` the freq is 1 (implicit),
/// otherwise a second `VInt(freq)` is read.
pub fn read_freqs(frq: &[u8], info: &TermInfo) -> Vec<(usize, u32)> {
    let mut pos = info.freq_pointer as usize;
    let mut out = Vec::with_capacity(info.doc_freq as usize);
    let mut prev = 0usize;
    for _ in 0..info.doc_freq {
        let v = read_vint(frq, &mut pos);
        let doc_delta = (v >> 1) as usize;
        let freq = if v & 1 == 1 { 1 } else { read_vint(frq, &mut pos) as u32 };
        let doc_id = prev + doc_delta;
        out.push((doc_id, freq));
        prev = doc_id;
    }
    out
}

/// Decodes the `.prx` positions for `doc_id`, walking `.frq` in parallel
/// to know how many positions to consume per doc.
pub fn read_positions(frq: &[u8], prx: &[u8], info: &TermInfo, doc_id: usize) -> Vec<u32> {
    let mut fpos = info.freq_pointer as usize;
    let mut ppos = info.prox_pointer as usize;
    let mut prev = 0usize;
    for _ in 0..info.doc_freq {
        let v = read_vint(frq, &mut fpos);
        let doc_delta = (v >> 1) as usize;
        let freq = if v & 1 == 1 { 1 } else { read_vint(frq, &mut fpos) as u32 };
        let d = prev + doc_delta;
        let mut positions = Vec::with_capacity(freq as usize);
        let mut prev_pos = 0u32;
        for _ in 0..freq {
            prev_pos += read_vint(prx, &mut ppos) as u32;
            positions.push(prev_pos);
        }
        if d == doc_id {
            return positions;
        }
        prev = d;
    }
    Vec::new()
}

/// Decodes ALL positions of a term in a single pass: doc -> positions.
/// Avoids re-walking `.frq`/`.prx` from the pointer for each doc (which made phrase O(C·docFreq)).
pub fn read_all_positions(frq: &[u8], prx: &[u8], info: &TermInfo) -> Vec<(usize, Vec<u32>)> {
    let mut fpos = info.freq_pointer as usize;
    let mut ppos = info.prox_pointer as usize;
    let mut prev = 0usize;
    let mut out = Vec::with_capacity(info.doc_freq as usize);
    for _ in 0..info.doc_freq {
        let v = read_vint(frq, &mut fpos);
        let doc_delta = (v >> 1) as usize;
        let freq = if v & 1 == 1 { 1 } else { read_vint(frq, &mut fpos) as u32 };
        let d = prev + doc_delta;
        let mut positions = Vec::with_capacity(freq as usize);
        let mut prev_pos = 0u32;
        for _ in 0..freq {
            prev_pos += read_vint(prx, &mut ppos) as u32;
            positions.push(prev_pos);
        }
        out.push((d, positions));
        prev = d;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zsl::cfs::CompoundFile;
    use crate::zsl::fields::read_field_infos;
    use crate::zsl::terms::TermDict;
    use std::path::PathBuf;

    fn cfs() -> CompoundFile {
        let dir = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/zsl_index"));
        let path = std::fs::read_dir(&dir).unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .find(|p| p.extension().map(|x| x == "cfs").unwrap_or(false)).unwrap();
        CompoundFile::open(&path).unwrap()
    }

    #[test]
    fn freqs_match_oracle_for_new() {
        let cf = cfs();
        let sub = |ext: &str| cf.sub(&cf.names().into_iter().find(|n| n.ends_with(ext)).unwrap()).unwrap().to_vec();
        let names: Vec<String> = read_field_infos(&sub(".fnm")).into_iter().map(|f| f.name).collect();
        let dict = TermDict::read(&sub(".tis"), &names);
        let info = dict.info("title", "new").unwrap();
        let freqs = read_freqs(&sub(".frq"), info);
        // "new" is in all 4 docs (all "New workflow"), freq 1 each
        assert_eq!(freqs, vec![(0, 1), (1, 1), (2, 1), (3, 1)]);
    }
}
