//! ZSL term dictionary writer: `.tis` (all terms) and `.tii` (sparse index).
//! Inverse of `zsl::terms`. Mirrors `SegmentWriter::_dumpTermDictEntry` +
//! `initializeDictionaryFiles` + `closeDictionaryFiles`:
//! - header (24 bytes): marker `0xFFFFFFFD`, Long(termCount) [back-patch], indexInterval=128,
//!   skipInterval=0x7FFFFFFF, maxSkipLevels=0.
//! - `.tii` starts with ONE synthetic entry (empty term, field `0xFFFFFFFF` as a raw Int,
//!   byte `0x0F`, IndexDelta=24) and then one sample every `indexInterval` terms.
//! - each entry: VInt(prefixChars) + String(suffix) + VInt(field) + VInt(docFreq) +
//!   VInt(freqDelta) + VInt(proxDelta) (+ skipOffset if !=0, never here). Prefix shared
//!   only with the previous term of the SAME field.

use super::invert::TermPostings;
use super::postings::write_term_postings;
use crate::zsl::bytes::{write_i32_be, write_i64_be, write_modified_utf8, write_vint};

pub const INDEX_INTERVAL: u64 = 128;
const MARKER: i32 = -3; // 0xFFFFFFFD
const SKIP_INTERVAL: i32 = 0x7FFF_FFFF;
const MAX_SKIP_LEVELS: i32 = 0;

pub struct DictFiles {
    pub tis: Vec<u8>,
    pub tii: Vec<u8>,
    pub frq: Vec<u8>,
    pub prx: Vec<u8>,
}

pub fn write_term_dict(terms: &[TermPostings]) -> DictFiles {
    let mut tis = Vec::new();
    let mut tii = Vec::new();
    let mut frq = Vec::new();
    let mut prx = Vec::new();

    write_header(&mut tis);
    write_header(&mut tii);

    // initial synthetic .tii entry (hand-written, NOT via dump_entry):
    // the field number is a raw Int 0xFFFFFFFF, not a VInt.
    write_vint(&mut tii, 0); // prefixChars
    write_modified_utf8(&mut tii, ""); // empty suffix
    write_i32_be(&mut tii, -1); // 0xFFFFFFFF
    tii.push(0x0F);
    write_vint(&mut tii, 0); // docFreq
    write_vint(&mut tii, 0); // freqDelta
    write_vint(&mut tii, 0); // proxDelta
    write_vint(&mut tii, 24); // IndexDelta

    // state of the previous term in the .tis (borrows term text from `terms`, no clone)
    let mut prev: Option<(&str, usize, u64, u64)> = None; // (text, field, freqPtr, proxPtr)
    // state of the last sample in the .tii
    let mut idx_prev: Option<(&str, usize, u64, u64)> = None;
    let mut last_index_position: u64 = 24;

    for (i, term) in terms.iter().enumerate() {
        let (freq_ptr, prox_ptr) = write_term_postings(&mut frq, &mut prx, &term.docs);
        let doc_freq = term.doc_freq();

        dump_entry(&mut tis, prev, term, doc_freq, freq_ptr, prox_ptr);
        prev = Some((term.text.as_str(), term.field_num, freq_ptr, prox_ptr));

        // sample every indexInterval terms
        if (i as u64 + 1).is_multiple_of(INDEX_INTERVAL) {
            dump_entry(&mut tii, idx_prev, term, doc_freq, freq_ptr, prox_ptr);
            let index_position = tis.len() as u64;
            write_vint(&mut tii, index_position - last_index_position);
            last_index_position = index_position;
            idx_prev = Some((term.text.as_str(), term.field_num, freq_ptr, prox_ptr));
        }
    }

    let term_count = terms.len() as u64;
    patch_long(&mut tis, 4, term_count);
    let tii_count = (term_count - term_count % INDEX_INTERVAL) / INDEX_INTERVAL + 1;
    patch_long(&mut tii, 4, tii_count);

    DictFiles { tis, tii, frq, prx }
}

fn write_header(out: &mut Vec<u8>) {
    write_i32_be(out, MARKER);
    write_i64_be(out, 0); // placeholder termCount (back-patched at offset 4)
    write_i32_be(out, INDEX_INTERVAL as i32);
    write_i32_be(out, SKIP_INTERVAL);
    write_i32_be(out, MAX_SKIP_LEVELS);
}

fn patch_long(buf: &mut [u8], offset: usize, v: u64) {
    buf[offset..offset + 8].copy_from_slice(&v.to_be_bytes());
}

/// Writes a term dict entry. Shares a prefix with `prev` only if it is of the SAME
/// field; writes freq/prox as a delta relative to `prev`, or absolute if `prev` is None.
fn dump_entry(
    out: &mut Vec<u8>,
    prev: Option<(&str, usize, u64, u64)>,
    term: &TermPostings,
    doc_freq: u32,
    freq_ptr: u64,
    prox_ptr: u64,
) {
    let (prefix_chars, prefix_bytes) = match prev {
        Some((ptext, pfield, ..)) if pfield == term.field_num => common_prefix(ptext, &term.text),
        _ => (0, 0),
    };
    write_vint(out, prefix_chars as u64);
    write_modified_utf8(out, &term.text[prefix_bytes..]);
    write_vint(out, term.field_num as u64);
    write_vint(out, doc_freq as u64);
    match prev {
        Some((_, _, pf, pp)) => {
            write_vint(out, freq_ptr - pf);
            write_vint(out, prox_ptr - pp);
        }
        None => {
            write_vint(out, freq_ptr);
            write_vint(out, prox_ptr);
        }
    }
    // skipOffset is omitted: docFreq is always < skipInterval.
}

/// Common prefix in (chars, bytes). Matching chars ⟺ matching bytes (UTF-8),
/// so counting equal leading chars reproduces ZSL's byte-then-char calculation.
fn common_prefix(a: &str, b: &str) -> (usize, usize) {
    let mut chars = 0usize;
    let mut bytes = 0usize;
    for (ca, cb) in a.chars().zip(b.chars()) {
        if ca == cb {
            chars += 1;
            bytes += ca.len_utf8();
        } else {
            break;
        }
    }
    (chars, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zsl::bytes::read_u64_be;
    use crate::zsl::postings::{read_all_positions, read_freqs};
    use crate::zsl::terms::TermDict;
    use crate::zsl::writer::invert::invert;
    use crate::zsl::writer::{WriterDoc, WriterField, WriterOpts};

    fn field_names(inv: &crate::zsl::writer::invert::Inverted) -> Vec<String> {
        inv.fields.iter().map(|m| m.name.clone()).collect()
    }

    #[test]
    fn tis_roundtrips_docfreq_freqs_and_positions_through_reader() {
        let docs = vec![
            WriterDoc {
                fields: vec![
                    WriterField::text("title", "New workflow"),
                    WriterField::text("body", "new new"),
                ],
            },
            WriterDoc {
                fields: vec![
                    WriterField::text("title", "workflow done"),
                    WriterField::text("body", ""),
                ],
            },
        ];
        let inv = invert(&docs, &WriterOpts::default());
        let f = write_term_dict(&inv.terms);
        let names = field_names(&inv);
        let dict = TermDict::read(&f.tis, &names).unwrap();

        assert_eq!(dict.info("title", "workflow").unwrap().doc_freq, 2);
        assert_eq!(dict.info("title", "done").unwrap().doc_freq, 1);

        let body_new = dict.info("body", "new").unwrap();
        assert_eq!(read_freqs(&f.frq, body_new).unwrap(), vec![(0, 2)]);

        let title_wf = dict.info("title", "workflow").unwrap();
        assert_eq!(read_all_positions(&f.frq, &f.prx, title_wf).unwrap(), vec![(0, vec![2]), (1, vec![1])]);
    }

    #[test]
    fn tis_header_has_marker_and_backpatched_term_count() {
        let docs = vec![WriterDoc { fields: vec![WriterField::text("t", "a b c")] }];
        let inv = invert(&docs, &WriterOpts::default());
        let f = write_term_dict(&inv.terms);
        assert_eq!(&f.tis[0..4], &[0xFF, 0xFF, 0xFF, 0xFD]); // marker
        let mut pos = 4;
        assert_eq!(read_u64_be(&f.tis, &mut pos).unwrap(), 3); // 3 terms: a,b,c
    }

    #[test]
    fn tii_starts_with_synthetic_entry_and_count_one_for_small_batch() {
        let docs = vec![WriterDoc { fields: vec![WriterField::text("t", "a b c")] }];
        let inv = invert(&docs, &WriterOpts::default());
        let f = write_term_dict(&inv.terms);
        // header 24 bytes; count == 1 (fewer than indexInterval terms)
        let mut pos = 4;
        assert_eq!(read_u64_be(&f.tii, &mut pos).unwrap(), 1);
        // synthetic entry: VInt(0) VInt(0) Int(0xFFFFFFFF) byte(0x0F) VInt0 VInt0 VInt0 VInt(24)
        let expected = [0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x0F, 0x00, 0x00, 0x00, 0x18];
        assert_eq!(&f.tii[24..24 + expected.len()], &expected);
        assert_eq!(f.tii.len(), 24 + expected.len()); // only header + synthetic
    }

    #[test]
    fn tii_samples_every_index_interval_terms() {
        // 300 unique terms in one field -> count = (300 - 300%128)/128 + 1 = 3
        let value: String = (0..300).map(|i| format!("w{i:04}")).collect::<Vec<_>>().join(" ");
        let docs = vec![WriterDoc { fields: vec![WriterField::text("t", &value)] }];
        let inv = invert(&docs, &WriterOpts::default());
        assert_eq!(inv.terms.len(), 300);
        let f = write_term_dict(&inv.terms);

        let mut pos = 4;
        assert_eq!(read_u64_be(&f.tis, &mut pos).unwrap(), 300);
        let mut pos = 4;
        assert_eq!(read_u64_be(&f.tii, &mut pos).unwrap(), 3);

        // and the .tis is still readable
        let dict = TermDict::read(&f.tis, &field_names(&inv)).unwrap();
        assert_eq!(dict.info("t", "w0000").unwrap().doc_freq, 1);
        assert_eq!(dict.info("t", "w0299").unwrap().doc_freq, 1);
    }
}
