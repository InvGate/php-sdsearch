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
use std::io::{self, Seek, SeekFrom, Write};

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

/// prev-term state for prefix-sharing + freq/prox pointer deltas: (text, field_num, freq_ptr, prox_ptr).
/// Owned (unlike the batch loop's `&str` borrow) because a streaming writer only sees one
/// term at a time and must remember the previous one across calls.
type PrevTerm = (String, usize, u64, u64);

/// Streams the four dict sub-files. Terms MUST be added in ZSL canonical order
/// (fieldName·\0·text). docFreq is taken per term from `term.docs.len()`. `.tis`/`.tii`
/// need Seek to back-patch the 8-byte term-count in their headers on `finish`.
pub struct TermDictStreamWriter<Tis, Tii, Frq, Prx>
where
    Tis: Write + Seek,
    Tii: Write + Seek,
    Frq: Write,
    Prx: Write,
{
    tis: Tis,
    tii: Tii,
    frq: Frq,
    prx: Prx,
    tis_len: u64,
    frq_len: u64,
    prx_len: u64,
    prev: Option<PrevTerm>,
    idx_prev: Option<PrevTerm>,
    last_index_position: u64,
    term_count: u64,
}

impl<Tis, Tii, Frq, Prx> TermDictStreamWriter<Tis, Tii, Frq, Prx>
where
    Tis: Write + Seek,
    Tii: Write + Seek,
    Frq: Write,
    Prx: Write,
{
    pub fn new(mut tis: Tis, mut tii: Tii, frq: Frq, prx: Prx) -> io::Result<Self> {
        let mut header = Vec::new();
        write_header(&mut header);
        tis.write_all(&header)?;
        tii.write_all(&header)?;

        // initial synthetic .tii entry (hand-written, NOT via dump_entry):
        // the field number is a raw Int 0xFFFFFFFF, not a VInt.
        let mut synthetic = Vec::new();
        write_vint(&mut synthetic, 0); // prefixChars
        write_modified_utf8(&mut synthetic, ""); // empty suffix
        write_i32_be(&mut synthetic, -1); // 0xFFFFFFFF
        synthetic.push(0x0F);
        write_vint(&mut synthetic, 0); // docFreq
        write_vint(&mut synthetic, 0); // freqDelta
        write_vint(&mut synthetic, 0); // proxDelta
        write_vint(&mut synthetic, 24); // IndexDelta
        tii.write_all(&synthetic)?;

        Ok(Self {
            tis,
            tii,
            frq,
            prx,
            tis_len: header.len() as u64, // 24: header is already on disk
            frq_len: 0,
            prx_len: 0,
            prev: None,
            idx_prev: None,
            last_index_position: 24,
            term_count: 0,
        })
    }

    /// Appends one term's dictionary entry (`.tis`, and `.tii` every `INDEX_INTERVAL`th
    /// term) plus its postings (`.frq`/`.prx`). Terms must arrive in ZSL canonical order.
    pub fn add_term(&mut self, term: &TermPostings) -> io::Result<()> {
        // `write_term_postings` only depends on `term.docs` (it resets its own doc/position
        // deltas per call), so writing into fresh local buffers reproduces the exact same
        // bytes as appending into a running `.frq`/`.prx` buffer; the absolute pointers are
        // then our own running lengths rather than the (always-zero) buffer-local ones.
        let mut frq_buf = Vec::new();
        let mut prx_buf = Vec::new();
        write_term_postings(&mut frq_buf, &mut prx_buf, &term.docs);
        let freq_ptr = self.frq_len;
        let prox_ptr = self.prx_len;

        let doc_freq = term.doc_freq();

        let mut tis_entry = Vec::new();
        dump_entry(
            &mut tis_entry,
            self.prev
                .as_ref()
                .map(|(t, f, fp, pp)| (t.as_str(), *f, *fp, *pp)),
            term,
            doc_freq,
            freq_ptr,
            prox_ptr,
        );
        self.tis.write_all(&tis_entry)?;
        self.tis_len += tis_entry.len() as u64;
        self.prev = Some((term.text.clone(), term.field_num, freq_ptr, prox_ptr));

        self.frq.write_all(&frq_buf)?;
        self.frq_len += frq_buf.len() as u64;
        self.prx.write_all(&prx_buf)?;
        self.prx_len += prx_buf.len() as u64;

        self.term_count += 1;
        // sample every indexInterval terms
        if self.term_count.is_multiple_of(INDEX_INTERVAL) {
            let mut tii_entry = Vec::new();
            dump_entry(
                &mut tii_entry,
                self.idx_prev
                    .as_ref()
                    .map(|(t, f, fp, pp)| (t.as_str(), *f, *fp, *pp)),
                term,
                doc_freq,
                freq_ptr,
                prox_ptr,
            );
            let index_position = self.tis_len;
            write_vint(&mut tii_entry, index_position - self.last_index_position);
            self.last_index_position = index_position;
            self.tii.write_all(&tii_entry)?;
            self.idx_prev = Some((term.text.clone(), term.field_num, freq_ptr, prox_ptr));
        }

        Ok(())
    }

    /// Back-patches the 8-byte term counts at offset 4 in `.tis`/`.tii` and flushes all
    /// four sinks.
    pub fn finish(mut self) -> io::Result<()> {
        let term_count = self.term_count;
        self.tis.seek(SeekFrom::Start(4))?;
        self.tis.write_all(&term_count.to_be_bytes())?;

        let tii_count = (term_count - term_count % INDEX_INTERVAL) / INDEX_INTERVAL + 1;
        self.tii.seek(SeekFrom::Start(4))?;
        self.tii.write_all(&tii_count.to_be_bytes())?;

        self.tis.flush()?;
        self.tii.flush()?;
        self.frq.flush()?;
        self.prx.flush()?;
        Ok(())
    }
}

pub fn write_term_dict(terms: &[TermPostings]) -> DictFiles {
    let mut tis = io::Cursor::new(Vec::new());
    let mut tii = io::Cursor::new(Vec::new());
    let mut frq = Vec::new();
    let mut prx = Vec::new();
    {
        let mut writer = TermDictStreamWriter::new(&mut tis, &mut tii, &mut frq, &mut prx)
            .expect("writing to an in-memory Vec<u8> cannot fail");
        for term in terms {
            writer
                .add_term(term)
                .expect("writing to an in-memory Vec<u8> cannot fail");
        }
        writer
            .finish()
            .expect("writing to an in-memory Vec<u8> cannot fail");
    }

    DictFiles {
        tis: tis.into_inner(),
        tii: tii.into_inner(),
        frq,
        prx,
    }
}

fn write_header(out: &mut Vec<u8>) {
    write_i32_be(out, MARKER);
    write_i64_be(out, 0); // placeholder termCount (back-patched at offset 4)
    write_i32_be(out, INDEX_INTERVAL as i32);
    write_i32_be(out, SKIP_INTERVAL);
    write_i32_be(out, MAX_SKIP_LEVELS);
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
        assert_eq!(
            read_all_positions(&f.frq, &f.prx, title_wf).unwrap(),
            vec![(0, vec![2]), (1, vec![1])]
        );
    }

    #[test]
    fn tis_header_has_marker_and_backpatched_term_count() {
        let docs = vec![WriterDoc {
            fields: vec![WriterField::text("t", "a b c")],
        }];
        let inv = invert(&docs, &WriterOpts::default());
        let f = write_term_dict(&inv.terms);
        assert_eq!(&f.tis[0..4], &[0xFF, 0xFF, 0xFF, 0xFD]); // marker
        let mut pos = 4;
        assert_eq!(read_u64_be(&f.tis, &mut pos).unwrap(), 3); // 3 terms: a,b,c
    }

    #[test]
    fn tii_starts_with_synthetic_entry_and_count_one_for_small_batch() {
        let docs = vec![WriterDoc {
            fields: vec![WriterField::text("t", "a b c")],
        }];
        let inv = invert(&docs, &WriterOpts::default());
        let f = write_term_dict(&inv.terms);
        // header 24 bytes; count == 1 (fewer than indexInterval terms)
        let mut pos = 4;
        assert_eq!(read_u64_be(&f.tii, &mut pos).unwrap(), 1);
        // synthetic entry: VInt(0) VInt(0) Int(0xFFFFFFFF) byte(0x0F) VInt0 VInt0 VInt0 VInt(24)
        let expected = [
            0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x0F, 0x00, 0x00, 0x00, 0x18,
        ];
        assert_eq!(&f.tii[24..24 + expected.len()], &expected);
        assert_eq!(f.tii.len(), 24 + expected.len()); // only header + synthetic
    }

    #[test]
    fn tii_samples_every_index_interval_terms() {
        // 300 unique terms in one field -> count = (300 - 300%128)/128 + 1 = 3
        let value: String = (0..300)
            .map(|i| format!("w{i:04}"))
            .collect::<Vec<_>>()
            .join(" ");
        let docs = vec![WriterDoc {
            fields: vec![WriterField::text("t", &value)],
        }];
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

    /// Builds a hand-crafted, ZSL-canonically-sorted term list covering: multiple fields,
    /// shared prefixes within a field, >128 terms (to exercise `.tii` sampling AND the
    /// term-count back-patch), and multi-doc postings with positions.
    fn multi_field_sample_terms() -> Vec<TermPostings> {
        let mut terms = Vec::new();

        // field 0 ("body"): 150 terms sharing the "shared" prefix, sorted ascending, so
        // consecutive terms within the field share a common prefix. Some have multi-doc
        // postings with positions to exercise freq/prox deltas.
        for i in 0..150usize {
            let text = format!("shared{i:04}");
            let docs = if i % 10 == 0 {
                vec![(i, vec![0u32, 3]), (i + 1000, vec![1u32])]
            } else {
                vec![(i, vec![0u32])]
            };
            terms.push(TermPostings {
                field_num: 0,
                text,
                docs,
            });
        }

        // field 1 ("title"): a handful of terms, some sharing prefixes, to make sure
        // prefix-sharing does NOT leak across fields (field 0's last term is "shared0149").
        for (text, docs) in [
            ("alpha".to_string(), vec![(2usize, vec![5u32])]),
            ("alphabet".to_string(), vec![(3usize, vec![0u32, 1, 2])]),
            ("beta".to_string(), vec![(4usize, vec![7u32])]),
        ] {
            terms.push(TermPostings {
                field_num: 1,
                text,
                docs,
            });
        }

        terms
    }

    #[test]
    fn stream_writer_matches_batch_writer_byte_for_byte() {
        let terms = multi_field_sample_terms();
        assert!(terms.len() > 128, "must exceed indexInterval to test .tii sampling + patch");

        let expected = write_term_dict(&terms);

        let mut tis = std::io::Cursor::new(Vec::new());
        let mut tii = std::io::Cursor::new(Vec::new());
        let mut frq = Vec::new();
        let mut prx = Vec::new();
        {
            let mut writer =
                TermDictStreamWriter::new(&mut tis, &mut tii, &mut frq, &mut prx).unwrap();
            for term in &terms {
                writer.add_term(term).unwrap();
            }
            writer.finish().unwrap();
        }

        assert_eq!(tis.into_inner(), expected.tis, "tis mismatch");
        assert_eq!(tii.into_inner(), expected.tii, "tii mismatch");
        assert_eq!(frq, expected.frq, "frq mismatch");
        assert_eq!(prx, expected.prx, "prx mismatch");
    }
}
