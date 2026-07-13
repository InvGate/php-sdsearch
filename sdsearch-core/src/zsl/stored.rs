//! Stored fields reader (.fdt indexed by .fdx).
use crate::zsl::bytes::{
    checked_capacity, read_byte, read_modified_utf8, read_u64_be, read_vint, truncated,
};
use crate::zsl::fields::FieldInfo;
use std::collections::HashMap;

/// raw stored field: field number LOCAL to the segment, value, and the `tokenized`
/// (bit 0x01 of `.fdt`) and `is_binary` (bit 0x02 of `.fdt`) flags. Exact inverse of
/// `writer::invert::StoredField` — the merge uses it to copy stored fields preserving order,
/// field_num, and the `tokenized` flag (needed to reproduce the `.fdt` bytes),
/// then remapping field_num to the merged segment. `is_binary` is only a defensive
/// guard in the merge: the host application doesn't index binaries, so it must always be `false`.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredRaw {
    pub field_num: usize,
    pub value: String,
    pub tokenized: bool,
    pub is_binary: bool,
}

/// reads a doc's stored fields in write order (with field_num + flag), without
/// resolving names. Returns empty if the doc is out of range of the `.fdx`.
pub fn read_stored_raw(fdx: &[u8], fdt: &[u8], doc_id: usize) -> std::io::Result<Vec<StoredRaw>> {
    let mut out = Vec::new();
    let idx_pos = doc_id * 8;
    if idx_pos + 8 > fdx.len() {
        return Ok(out); // doc out of range of this .fdx: no stored fields, not an error
    }
    let mut p = idx_pos;
    let fdt_off = read_u64_be(fdx, &mut p)? as usize;
    let mut pos = fdt_off;
    let stored_count = read_vint(fdt, &mut pos)? as usize;
    out.reserve(checked_capacity(
        stored_count,
        fdt.len().saturating_sub(pos),
    ));
    for _ in 0..stored_count {
        let field_num = read_vint(fdt, &mut pos)? as usize;
        let flags = read_byte(fdt, &mut pos)?;
        let tokenized = flags & 0x01 != 0;
        let is_binary = flags & 0x02 != 0;
        let value = if is_binary {
            let len = read_vint(fdt, &mut pos)? as usize;
            let end = pos.checked_add(len).ok_or_else(|| truncated(pos))?;
            let bytes = fdt.get(pos..end).ok_or_else(|| truncated(pos))?;
            pos = end;
            String::from_utf8_lossy(bytes).into_owned()
        } else {
            read_modified_utf8(fdt, &mut pos)?
        };
        out.push(StoredRaw {
            field_num,
            value,
            tokenized,
            is_binary,
        });
    }
    Ok(out)
}

/// reads a doc's stored fields resolving field_num -> name via `fields`.
/// Delegates `.fdt`/`.fdx` parsing to [`read_stored_raw`]; entries whose `field_num`
/// is out of range of `fields` are dropped.
pub fn read_stored_fields(
    fdx: &[u8],
    fdt: &[u8],
    fields: &[FieldInfo],
    doc_id: usize,
) -> std::io::Result<HashMap<String, String>> {
    let mut out = HashMap::new();
    for r in read_stored_raw(fdx, fdt, doc_id)? {
        if let Some(fi) = fields.get(r.field_num) {
            out.insert(fi.name.clone(), r.value);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zsl::cfs::CompoundFile;
    use crate::zsl::fields::read_field_infos;
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
    fn stored_fields_match_zsl_oracle_for_doc0() {
        let cf = cfs();
        let fnm = cf
            .names()
            .into_iter()
            .find(|n| n.ends_with(".fnm"))
            .unwrap();
        let fdx = cf
            .names()
            .into_iter()
            .find(|n| n.ends_with(".fdx"))
            .unwrap();
        let fdt = cf
            .names()
            .into_iter()
            .find(|n| n.ends_with(".fdt"))
            .unwrap();
        let fields = read_field_infos(cf.sub(&fnm).unwrap()).unwrap();
        let stored =
            read_stored_fields(cf.sub(&fdx).unwrap(), cf.sub(&fdt).unwrap(), &fields, 0).unwrap();
        // FULL parity with what ZSL stored for doc 0 (read from the oracle).
        // tokenized Text fields (title, users) carry a trailing '\n' that compactText()
        // adds — it's a faithful part of the bytes, NOT trimmed.
        assert_eq!(stored, oracle_doc0_stored());
    }

    fn oracle_doc0_stored() -> std::collections::HashMap<String, String> {
        #[derive(serde::Deserialize)]
        struct Oracle {
            docs: Vec<OracleDoc>,
        }
        #[derive(serde::Deserialize)]
        struct OracleDoc {
            stored: std::collections::HashMap<String, String>,
        }
        let raw = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/zsl_expected.json"
        ))
        .expect("oracle missing");
        let o: Oracle = serde_json::from_str(&raw).unwrap();
        o.docs.into_iter().next().unwrap().stored
    }

    #[test]
    fn read_stored_raw_errors_on_corrupt_binary_len() {
        // fdx: one doc pointing at fdt offset 0
        let fdx = 0u64.to_be_bytes().to_vec();
        // fdt: storedCount=1, field_num=0, flags=0x02 (binary), len=VInt(200) but no bytes
        let fdt = vec![0x01, 0x00, 0x02, 0xC8, 0x01];
        assert!(read_stored_raw(&fdx, &fdt, 0).is_err());
    }
}
