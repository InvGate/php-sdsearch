//! Field infos reader (.fnm): field names + flags, indexed by field number.
use crate::zsl::bytes::{checked_capacity, read_byte, read_modified_utf8, read_vint};

#[derive(Debug, PartialEq)]
pub struct FieldInfo {
    pub name: String,
    pub is_indexed: bool,
}

pub fn read_field_infos(fnm: &[u8]) -> std::io::Result<Vec<FieldInfo>> {
    let mut pos = 0usize;
    let count = read_vint(fnm, &mut pos)? as usize;
    let mut out = Vec::with_capacity(checked_capacity(count, fnm.len()));
    for _ in 0..count {
        let name = read_modified_utf8(fnm, &mut pos)?;
        let flags = read_byte(fnm, &mut pos)?;
        out.push(FieldInfo {
            name,
            is_indexed: flags & 0x01 != 0,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_names_and_indexed_flag() {
        // VInt(2); "title" indexed(0x01); "id_attr" not indexed(0x00)
        let mut buf = vec![0x02];
        buf.push(0x05);
        buf.extend_from_slice(b"title");
        buf.push(0x01);
        buf.push(0x07);
        buf.extend_from_slice(b"id_attr");
        buf.push(0x00);
        let fields = read_field_infos(&buf).unwrap();
        assert_eq!(
            fields,
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

    #[test]
    fn read_field_infos_errors_on_truncation() {
        // VInt(2) but no field bytes follow
        assert!(read_field_infos(&[0x02]).is_err());
    }
}
