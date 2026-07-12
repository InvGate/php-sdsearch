//! ZSL postings decode: .frq (doc deltas + freqs) and .prx (positions).
use crate::zsl::bytes::{checked_capacity, read_vint};
use crate::zsl::terms::TermInfo;

/// Decodes `.frq` for a term: docId (accumulated delta) + freq per doc.
/// `VInt(v)`: `docDelta = v >> 1`; if `v & 1 == 1` the freq is 1 (implicit),
/// otherwise a second `VInt(freq)` is read.
pub fn read_freqs(frq: &[u8], info: &TermInfo) -> std::io::Result<Vec<(usize, u32)>> {
    let mut pos = info.freq_pointer as usize;
    let mut out = Vec::with_capacity(checked_capacity(
        info.doc_freq as usize,
        frq.len().saturating_sub(pos),
    ));
    let mut prev = 0usize;
    for _ in 0..info.doc_freq {
        let v = read_vint(frq, &mut pos)?;
        let doc_delta = (v >> 1) as usize;
        let freq = if v & 1 == 1 {
            1
        } else {
            read_vint(frq, &mut pos)? as u32
        };
        let doc_id = prev + doc_delta;
        out.push((doc_id, freq));
        prev = doc_id;
    }
    Ok(out)
}

/// Decodes the `.prx` positions for `doc_id`, walking `.frq` in parallel
/// to know how many positions to consume per doc.
pub fn read_positions(
    frq: &[u8],
    prx: &[u8],
    info: &TermInfo,
    doc_id: usize,
) -> std::io::Result<Vec<u32>> {
    let mut fpos = info.freq_pointer as usize;
    let mut ppos = info.prox_pointer as usize;
    let mut prev = 0usize;
    for _ in 0..info.doc_freq {
        let v = read_vint(frq, &mut fpos)?;
        let doc_delta = (v >> 1) as usize;
        let freq = if v & 1 == 1 {
            1
        } else {
            read_vint(frq, &mut fpos)? as u32
        };
        let d = prev + doc_delta;
        let mut positions = Vec::with_capacity(checked_capacity(
            freq as usize,
            prx.len().saturating_sub(ppos),
        ));
        let mut prev_pos = 0u32;
        for _ in 0..freq {
            prev_pos += read_vint(prx, &mut ppos)? as u32;
            positions.push(prev_pos);
        }
        if d == doc_id {
            return Ok(positions);
        }
        prev = d;
    }
    Ok(Vec::new())
}

/// Decodes a term's postings one doc at a time, invoking `f(doc, &positions)` in ascending
/// doc order. A scratch `Vec<u32>` is reused across docs so peak extra RAM is one doc's
/// positions, not the whole term.
pub fn for_each_posting(
    frq: &[u8],
    prx: &[u8],
    info: &TermInfo,
    mut f: impl FnMut(usize, &[u32]),
) -> std::io::Result<()> {
    let mut fpos = info.freq_pointer as usize;
    let mut ppos = info.prox_pointer as usize;
    let mut prev = 0usize;
    let mut positions: Vec<u32> = Vec::new();
    for _ in 0..info.doc_freq {
        let v = read_vint(frq, &mut fpos)?;
        let doc_delta = (v >> 1) as usize;
        let freq = if v & 1 == 1 {
            1
        } else {
            read_vint(frq, &mut fpos)? as u32
        };
        let d = prev + doc_delta;
        positions.clear();
        let mut prev_pos = 0u32;
        for _ in 0..freq {
            prev_pos += read_vint(prx, &mut ppos)? as u32;
            positions.push(prev_pos);
        }
        f(d, &positions);
        prev = d;
    }
    Ok(())
}

/// Decodes ALL positions of a term in a single pass: doc -> positions.
/// Avoids re-walking `.frq`/`.prx` from the pointer for each doc (which made phrase O(C·docFreq)).
pub fn read_all_positions(
    frq: &[u8],
    prx: &[u8],
    info: &TermInfo,
) -> std::io::Result<Vec<(usize, Vec<u32>)>> {
    let mut out = Vec::with_capacity(checked_capacity(
        info.doc_freq as usize,
        frq.len().saturating_sub(info.freq_pointer as usize),
    ));
    for_each_posting(frq, prx, info, |d, pos| out.push((d, pos.to_vec())))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zsl::cfs::CompoundFile;
    use crate::zsl::fields::read_field_infos;
    use crate::zsl::terms::EagerTermDict;
    use std::path::PathBuf;

    fn cfs() -> CompoundFile {
        let dir = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/zsl_index"
        ));
        let path = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .find(|p| p.extension().is_some_and(|x| x == "cfs"))
            .unwrap();
        CompoundFile::open(&path).unwrap()
    }

    #[test]
    fn freqs_match_oracle_for_new() {
        let cf = cfs();
        let sub = |ext: &str| {
            cf.sub(&cf.names().into_iter().find(|n| n.ends_with(ext)).unwrap())
                .unwrap()
                .to_vec()
        };
        let names: Vec<String> = read_field_infos(&sub(".fnm"))
            .unwrap()
            .into_iter()
            .map(|f| f.name)
            .collect();
        let dict = EagerTermDict::read(&sub(".tis"), &names).unwrap();
        let info = dict.info("title", "new").unwrap();
        let freqs = read_freqs(&sub(".frq"), &info).unwrap();
        // "new" is in all 4 docs (all "New workflow"), freq 1 each
        assert_eq!(freqs, vec![(0, 1), (1, 1), (2, 1), (3, 1)]);
    }

    #[test]
    fn read_freqs_errors_on_truncated_frq() {
        let info = TermInfo {
            doc_freq: 3,
            freq_pointer: 0,
            prox_pointer: 0,
        };
        assert!(read_freqs(&[], &info).is_err());
    }

    #[test]
    fn for_each_posting_decodes_written_postings() {
        // Independent ground truth: encode known postings with the writer, then assert the
        // decoder reproduces them EXACTLY. Doc ids ascend and the per-doc position counts VARY
        // (3, 1, 4, 1) so a scratch buffer that is not cleared between docs would leak stale
        // positions into later docs and fail here. NOT compared against `read_all_positions`
        // (which now delegates to `for_each_posting`, so that comparison is tautological).
        let docs: Vec<(usize, Vec<u32>)> = vec![
            (0, vec![0, 3, 7]),
            (2, vec![5]),
            (5, vec![1, 2, 9, 40]),
            (6, vec![0]),
        ];

        let mut frq: Vec<u8> = Vec::new();
        let mut prx: Vec<u8> = Vec::new();
        let (freq_pointer, prox_pointer) =
            crate::zsl::writer::postings::write_term_postings(&mut frq, &mut prx, &docs);
        let info = TermInfo {
            doc_freq: docs.len() as u32,
            freq_pointer,
            prox_pointer,
        };

        let mut got: Vec<(usize, Vec<u32>)> = Vec::new();
        for_each_posting(&frq, &prx, &info, |doc, pos| got.push((doc, pos.to_vec()))).unwrap();
        assert_eq!(got, docs);
    }
}
