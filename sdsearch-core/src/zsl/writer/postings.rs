//! ZSL postings writer: `.frq` (doc deltas + freqs) and `.prx` (positions).
//! Inverse of `zsl::postings`. Mirrors `SegmentWriter::addTerm`: per doc it writes
//! `docDelta<<1` (+`1` with an implicit freq if freq==1, or `+ VInt(freq)`), and in `.prx`
//! the positions as deltas. Returns the start pointers in each file,
//! which the term dict stores as `freqPointer`/`proxPointer`.

use crate::zsl::bytes::write_vint;

/// Writes ONE term's postings (ascending docs, ascending positions)
/// to the end of `frq`/`prx`. Returns `(freq_pointer, prox_pointer)` = start offsets.
pub fn write_term_postings(
    frq: &mut Vec<u8>,
    prx: &mut Vec<u8>,
    docs: &[(usize, Vec<u32>)],
) -> (u64, u64) {
    let freq_pointer = frq.len() as u64;
    let prox_pointer = prx.len() as u64;

    let mut prev_doc = 0usize;
    for (doc_id, positions) in docs {
        let doc_delta = (doc_id - prev_doc) * 2;
        prev_doc = *doc_id;
        if positions.len() > 1 {
            write_vint(frq, doc_delta as u64);
            write_vint(frq, positions.len() as u64);
        } else {
            write_vint(frq, (doc_delta + 1) as u64);
        }

        let mut prev_pos = 0u32;
        for &pos in positions {
            write_vint(prx, (pos - prev_pos) as u64);
            prev_pos = pos;
        }
    }

    (freq_pointer, prox_pointer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zsl::postings::{read_all_positions, read_freqs};
    use crate::zsl::terms::TermInfo;

    #[test]
    fn roundtrips_freqs_and_positions_through_reader() {
        // doc0: freq 2 pos[1,2]; doc2: freq 1 pos[1]
        let docs = vec![(0usize, vec![1u32, 2]), (2, vec![1])];
        let mut frq = Vec::new();
        let mut prx = Vec::new();
        let (fp, pp) = write_term_postings(&mut frq, &mut prx, &docs);
        assert_eq!((fp, pp), (0, 0));

        let info = TermInfo {
            doc_freq: 2,
            freq_pointer: fp,
            prox_pointer: pp,
        };
        assert_eq!(read_freqs(&frq, &info).unwrap(), vec![(0, 2), (2, 1)]);
        assert_eq!(
            read_all_positions(&frq, &prx, &info).unwrap(),
            vec![(0, vec![1, 2]), (2, vec![1])]
        );
    }

    #[test]
    fn pointers_are_absolute_offsets_for_a_second_term() {
        let mut frq = Vec::new();
        let mut prx = Vec::new();
        // the first term consumes bytes; the second starts at offsets != 0
        write_term_postings(&mut frq, &mut prx, &[(0usize, vec![1u32])]);
        let (fp, pp) = write_term_postings(&mut frq, &mut prx, &[(1usize, vec![3u32, 7])]);
        assert!(fp > 0 && pp > 0);

        let info = TermInfo {
            doc_freq: 1,
            freq_pointer: fp,
            prox_pointer: pp,
        };
        assert_eq!(read_freqs(&frq, &info).unwrap(), vec![(1, 2)]);
        // 0-based position? no: we store them raw; deltas 3 then 4 -> 3,7
        assert_eq!(
            read_all_positions(&frq, &prx, &info).unwrap(),
            vec![(1, vec![3, 7])]
        );
    }

    #[test]
    fn keyword_position_zero_roundtrips() {
        // keyword: a single position 0 (delta 0)
        let mut frq = Vec::new();
        let mut prx = Vec::new();
        let (fp, pp) = write_term_postings(&mut frq, &mut prx, &[(5usize, vec![0u32])]);
        let info = TermInfo {
            doc_freq: 1,
            freq_pointer: fp,
            prox_pointer: pp,
        };
        assert_eq!(read_freqs(&frq, &info).unwrap(), vec![(5, 1)]);
        assert_eq!(
            read_all_positions(&frq, &prx, &info).unwrap(),
            vec![(5, vec![0])]
        );
    }
}
