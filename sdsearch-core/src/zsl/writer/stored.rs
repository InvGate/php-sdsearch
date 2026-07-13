//! Stored fields (.fdt/.fdx) writer. Inverse of `zsl::stored`. Mirrors
//! `SegmentWriter::addStoredFields`. `.fdx`: one Long per doc = offset of the block in `.fdt`.
//! `.fdt` per doc: `VInt(count)` and per field `VInt(field_num)`, a flags byte
//! `(tokenized?0x01)|(binary?0x02)` and the value (modified-UTF-8 String; never binary here).
//! The trailing `\n` that compactText adds is preserved verbatim (it is part of the value).

use super::invert::StoredField;
use crate::zsl::bytes::{write_modified_utf8, write_vint};
use std::io::{self, Write};

/// Writes `.fdt`/`.fdx` incrementally, one doc at a time. `.fdx` gets the current `.fdt`
/// length (i64 BE) at the start of each doc; `.fdt` gets `VInt(count)` + per-field
/// `VInt(field_num)` + flags + `String(value)`. No back-patch: same byte layout as
/// `write_stored`, just emitted doc-by-doc instead of built in one pass.
pub struct StoredStreamWriter<Fdt: Write, Fdx: Write> {
    fdt: Fdt,
    fdx: Fdx,
    fdt_len: u64,
}

impl<Fdt: Write, Fdx: Write> StoredStreamWriter<Fdt, Fdx> {
    pub fn new(fdt: Fdt, fdx: Fdx) -> Self {
        Self {
            fdt,
            fdx,
            fdt_len: 0,
        }
    }

    /// Appends one doc's stored fields: an `.fdx` pointer to the current `.fdt` length,
    /// then the doc's `.fdt` block.
    pub fn add_doc(&mut self, fields: &[StoredField]) -> io::Result<()> {
        self.fdx.write_all(&(self.fdt_len as i64).to_be_bytes())?;

        // Build the block in RAM first (same layout as `write_stored`) so we can measure its
        // length without requiring `Fdt: Seek`, then stream it out in one write.
        let mut block = Vec::new();
        write_vint(&mut block, fields.len() as u64);
        for sf in fields {
            write_vint(&mut block, sf.field_num as u64);
            block.push(u8::from(sf.tokenized)); // never binary here
            write_modified_utf8(&mut block, &sf.value);
        }

        self.fdt.write_all(&block)?;
        self.fdt_len += block.len() as u64;
        Ok(())
    }

    /// Flushes both sinks. No back-patch is needed for this format.
    pub fn finish(mut self) -> io::Result<()> {
        self.fdt.flush()?;
        self.fdx.flush()?;
        Ok(())
    }
}

/// Writes (fdt, fdx) for the per-doc stored fields.
pub fn write_stored(docs_stored: &[Vec<StoredField>]) -> (Vec<u8>, Vec<u8>) {
    let mut fdt = Vec::new();
    let mut fdx = Vec::new();
    {
        let mut writer = StoredStreamWriter::new(&mut fdt, &mut fdx);
        for fields in docs_stored {
            writer
                .add_doc(fields)
                .expect("writing to an in-memory Vec<u8> cannot fail");
        }
        writer
            .finish()
            .expect("flushing an in-memory Vec<u8> cannot fail");
    }
    (fdt, fdx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zsl::fields::FieldInfo;
    use crate::zsl::stored::read_stored_fields;

    fn sample_docs() -> Vec<Vec<StoredField>> {
        vec![
            // multiple fields, tokenized flag on and off
            vec![
                StoredField {
                    field_num: 0,
                    value: "New workflow\n".into(),
                    tokenized: true,
                },
                StoredField {
                    field_num: 1,
                    value: "42".into(),
                    tokenized: false,
                },
            ],
            // empty doc (no stored fields at all)
            vec![],
            // unicode value (multi-byte chars + NUL)
            vec![StoredField {
                field_num: 2,
                value: "über\u{1F680}\u{0}end".into(),
                tokenized: true,
            }],
            // single field, not tokenized
            vec![StoredField {
                field_num: 3,
                value: "TICKET-12345".into(),
                tokenized: false,
            }],
        ]
    }

    // --- independent oracle: the ORIGINAL (pre-streaming, commit 583e39b) `write_stored`
    // algorithm, duplicated here under a `reference_*` name so it does NOT call into
    // `StoredStreamWriter`/`write_stored` (the current module's real writer, now a thin wrapper
    // over the streaming writer) — comparing against those would make the byte-identity test
    // compare the streaming writer against itself (tautological). `write_vint`/`write_modified_utf8`
    // are reused from `crate::zsl::bytes` since they are untouched by the streaming refactor;
    // `write_i64_be` (only used here) is imported locally for the same reason.

    fn reference_write_stored(docs_stored: &[Vec<StoredField>]) -> (Vec<u8>, Vec<u8>) {
        use crate::zsl::bytes::write_i64_be;
        let mut fdt = Vec::new();
        let mut fdx = Vec::new();
        for fields in docs_stored {
            write_i64_be(&mut fdx, fdt.len() as i64); // offset of this doc's block in .fdt
            write_vint(&mut fdt, fields.len() as u64);
            for sf in fields {
                write_vint(&mut fdt, sf.field_num as u64);
                fdt.push(u8::from(sf.tokenized)); // never binary here
                write_modified_utf8(&mut fdt, &sf.value);
            }
        }
        (fdt, fdx)
    }

    #[test]
    fn stream_writer_matches_batch_writer_byte_for_byte() {
        let docs = sample_docs();

        // Independent oracle: the pre-streaming reference implementation (duplicated from git
        // commit 583e39b, before `write_stored` became a thin wrapper over
        // `StoredStreamWriter`), NOT `write_stored` itself (which now shares its code with the
        // streaming writer under test and would make this comparison tautological).
        let (expected_fdt, expected_fdx) = reference_write_stored(&docs);

        let mut fdt_buf = Vec::new();
        let mut fdx_buf = Vec::new();
        {
            let mut writer = StoredStreamWriter::new(&mut fdt_buf, &mut fdx_buf);
            for fields in &docs {
                writer.add_doc(fields).unwrap();
            }
            writer.finish().unwrap();
        }

        assert_eq!(fdt_buf, expected_fdt, "fdt mismatch");
        assert_eq!(fdx_buf, expected_fdx, "fdx mismatch");
    }

    #[test]
    fn stored_roundtrips_through_reader_preserving_trailing_newline() {
        let stored = vec![
            vec![
                StoredField {
                    field_num: 0,
                    value: "New workflow\n".into(),
                    tokenized: true,
                },
                StoredField {
                    field_num: 1,
                    value: "42".into(),
                    tokenized: false,
                },
            ],
            vec![StoredField {
                field_num: 0,
                value: "other\n".into(),
                tokenized: true,
            }],
        ];
        let (fdt, fdx) = write_stored(&stored);
        let fields = vec![
            FieldInfo {
                name: "title".into(),
                is_indexed: true,
            },
            FieldInfo {
                name: "id".into(),
                is_indexed: false,
            },
        ];

        let d0 = read_stored_fields(&fdx, &fdt, &fields, 0).unwrap();
        assert_eq!(d0.get("title").unwrap(), "New workflow\n"); // \n preserved
        assert_eq!(d0.get("id").unwrap(), "42");

        let d1 = read_stored_fields(&fdx, &fdt, &fields, 1).unwrap();
        assert_eq!(d1.get("title").unwrap(), "other\n");
        assert!(!d1.contains_key("id"));
    }

    #[test]
    fn stored_raw_roundtrips_field_num_and_tokenized_flag() {
        use crate::zsl::stored::{StoredRaw, read_stored_raw};
        let stored = vec![
            vec![
                StoredField {
                    field_num: 0,
                    value: "New workflow\n".into(),
                    tokenized: true,
                },
                StoredField {
                    field_num: 2,
                    value: "42".into(),
                    tokenized: false,
                },
            ],
            vec![StoredField {
                field_num: 0,
                value: "other\n".into(),
                tokenized: true,
            }],
        ];
        let (fdt, fdx) = write_stored(&stored);
        // doc 0: preserves order, field_num and the tokenized flag
        assert_eq!(
            read_stored_raw(&fdx, &fdt, 0).unwrap(),
            vec![
                StoredRaw {
                    field_num: 0,
                    value: "New workflow\n".into(),
                    tokenized: true,
                    is_binary: false
                },
                StoredRaw {
                    field_num: 2,
                    value: "42".into(),
                    tokenized: false,
                    is_binary: false
                },
            ]
        );
        // the host application does not index binaries: the round-trip must never report is_binary=true
        assert!(
            read_stored_raw(&fdx, &fdt, 0)
                .unwrap()
                .iter()
                .all(|r| !r.is_binary)
        );
        assert_eq!(
            read_stored_raw(&fdx, &fdt, 1).unwrap(),
            vec![StoredRaw {
                field_num: 0,
                value: "other\n".into(),
                tokenized: true,
                is_binary: false
            }]
        );
        // out-of-range doc → empty
        assert!(read_stored_raw(&fdx, &fdt, 9).unwrap().is_empty());
    }
}
