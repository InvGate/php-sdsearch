//! Stored fields (.fdt/.fdx) writer. Inverse of `zsl::stored`. Mirrors
//! `SegmentWriter::addStoredFields`. `.fdx`: one Long per doc = offset of the block in `.fdt`.
//! `.fdt` per doc: `VInt(count)` and per field `VInt(field_num)`, a flags byte
//! `(tokenized?0x01)|(binary?0x02)` and the value (modified-UTF-8 String; never binary here).
//! The trailing `\n` that compactText adds is preserved verbatim (it is part of the value).

use super::invert::StoredField;
use crate::zsl::bytes::{write_i64_be, write_modified_utf8, write_vint};

/// Writes (fdt, fdx) for the per-doc stored fields.
pub fn write_stored(docs_stored: &[Vec<StoredField>]) -> (Vec<u8>, Vec<u8>) {
    let mut fdt = Vec::new();
    let mut fdx = Vec::new();
    for fields in docs_stored {
        write_i64_be(&mut fdx, fdt.len() as i64); // offset of this doc's block in .fdt
        write_vint(&mut fdt, fields.len() as u64);
        for sf in fields {
            write_vint(&mut fdt, sf.field_num as u64);
            fdt.push(if sf.tokenized { 0x01 } else { 0x00 }); // never binary here
            write_modified_utf8(&mut fdt, &sf.value);
        }
    }
    (fdt, fdx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zsl::fields::FieldInfo;
    use crate::zsl::stored::read_stored_fields;

    #[test]
    fn stored_roundtrips_through_reader_preserving_trailing_newline() {
        let stored = vec![
            vec![
                StoredField { field_num: 0, value: "New workflow\n".into(), tokenized: true },
                StoredField { field_num: 1, value: "42".into(), tokenized: false },
            ],
            vec![StoredField { field_num: 0, value: "other\n".into(), tokenized: true }],
        ];
        let (fdt, fdx) = write_stored(&stored);
        let fields = vec![
            FieldInfo { name: "title".into(), is_indexed: true },
            FieldInfo { name: "id".into(), is_indexed: false },
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
        use crate::zsl::stored::{read_stored_raw, StoredRaw};
        let stored = vec![
            vec![
                StoredField { field_num: 0, value: "New workflow\n".into(), tokenized: true },
                StoredField { field_num: 2, value: "42".into(), tokenized: false },
            ],
            vec![StoredField { field_num: 0, value: "other\n".into(), tokenized: true }],
        ];
        let (fdt, fdx) = write_stored(&stored);
        // doc 0: preserves order, field_num and the tokenized flag
        assert_eq!(
            read_stored_raw(&fdx, &fdt, 0).unwrap(),
            vec![
                StoredRaw { field_num: 0, value: "New workflow\n".into(), tokenized: true, is_binary: false },
                StoredRaw { field_num: 2, value: "42".into(), tokenized: false, is_binary: false },
            ]
        );
        // the host application does not index binaries: the round-trip must never report is_binary=true
        assert!(read_stored_raw(&fdx, &fdt, 0).unwrap().iter().all(|r| !r.is_binary));
        assert_eq!(
            read_stored_raw(&fdx, &fdt, 1).unwrap(),
            vec![StoredRaw { field_num: 0, value: "other\n".into(), tokenized: true, is_binary: false }]
        );
        // out-of-range doc → empty
        assert!(read_stored_raw(&fdx, &fdt, 9).unwrap().is_empty());
    }
}
