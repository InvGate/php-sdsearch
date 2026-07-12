//! Field infos (.fnm) writer. Inverse of `zsl::fields`. Mirrors `SegmentWriter::_dumpFNM`:
//! VInt(count) + per field `writeString(name)` + a flags byte `(indexed?0x01)|(termVector?0x02)`.
//! The host application does not use term vectors, so the flag is 0x01 (indexed) or 0x00.

use super::invert::FieldMeta;
use crate::zsl::bytes::{write_modified_utf8, write_vint};

pub fn write_fnm(fields: &[FieldMeta]) -> Vec<u8> {
    let mut out = Vec::new();
    write_vint(&mut out, fields.len() as u64);
    for f in fields {
        write_modified_utf8(&mut out, &f.name);
        out.push(u8::from(f.indexed));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zsl::fields::{FieldInfo, read_field_infos};

    #[test]
    fn fnm_roundtrips_names_and_indexed_flag_through_reader() {
        let fields = vec![
            FieldMeta {
                name: "title".into(),
                indexed: true,
            },
            FieldMeta {
                name: "id_attr".into(),
                indexed: false,
            },
        ];
        let fnm = write_fnm(&fields);
        assert_eq!(
            read_field_infos(&fnm).unwrap(),
            vec![
                FieldInfo {
                    name: "title".into(),
                    is_indexed: true
                },
                FieldInfo {
                    name: "id_attr".into(),
                    is_indexed: false
                },
            ]
        );
    }
}
