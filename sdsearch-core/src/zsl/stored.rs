//! Stored fields reader (.fdt indexed by .fdx).
use crate::zsl::bytes::{
    checked_capacity, read_byte, read_modified_utf8, read_u64_be, read_vint, skip_modified_utf8,
    truncated,
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

/// Reads the value of ONE stored field of `doc_id`, identified by its segment-LOCAL
/// `field_num`, without materializing the doc's other fields.
///
/// Same `.fdt`/`.fdx` walk as [`read_stored_raw`], but an entry whose `field_num` does not match
/// is STEPPED OVER instead of decoded: the `String` allocated per skipped field is exactly the
/// cost this avoids. A doc in the host application carries ~11-17 stored fields, so resolving one
/// of them via [`read_stored_fields`] allocates that many strings plus a `HashMap` to throw all
/// but one away — this does one allocation.
///
/// Returns the FIRST entry matching `field_num` in write order (so a multi-valued field resolves
/// to its first value), or `None` when the doc has no such field or is out of range of the `.fdx`.
pub fn read_stored_field(
    fdx: &[u8],
    fdt: &[u8],
    doc_id: usize,
    field_num: usize,
) -> std::io::Result<Option<String>> {
    let idx_pos = doc_id * 8;
    if idx_pos + 8 > fdx.len() {
        return Ok(None); // doc out of range of this .fdx: no stored fields, not an error
    }
    let mut p = idx_pos;
    let fdt_off = read_u64_be(fdx, &mut p)? as usize;
    let mut pos = fdt_off;
    let stored_count = read_vint(fdt, &mut pos)? as usize;
    for _ in 0..stored_count {
        let num = read_vint(fdt, &mut pos)? as usize;
        let flags = read_byte(fdt, &mut pos)?;
        let is_binary = flags & 0x02 != 0;
        let wanted = num == field_num;
        if is_binary {
            // length-prefixed raw bytes: skippable by byte count, unlike modified UTF-8.
            let len = read_vint(fdt, &mut pos)? as usize;
            let end = pos.checked_add(len).ok_or_else(|| truncated(pos))?;
            let bytes = fdt.get(pos..end).ok_or_else(|| truncated(pos))?;
            pos = end;
            if wanted {
                return Ok(Some(String::from_utf8_lossy(bytes).into_owned()));
            }
        } else if wanted {
            return Ok(Some(read_modified_utf8(fdt, &mut pos)?));
        } else {
            skip_modified_utf8(fdt, &mut pos)?;
        }
    }
    Ok(None)
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
    fn read_stored_field_matches_the_full_read_for_every_field() {
        // `read_stored_field` skips entries instead of decoding them, so a mis-sized skip would
        // desynchronize the walk and yield a neighbouring field's bytes. Asserting equality
        // against `read_stored_fields` for EVERY field of EVERY doc pins that down against the
        // path the ZSL oracle already validates.
        let cf = cfs();
        let names = cf.names();
        let find = |ext: &str| names.iter().find(|n| n.ends_with(ext)).unwrap().clone();
        let fields = read_field_infos(cf.sub(&find(".fnm")).unwrap()).unwrap();
        let fdx = cf.sub(&find(".fdx")).unwrap();
        let fdt = cf.sub(&find(".fdt")).unwrap();

        let doc_count = fdx.len() / 8;
        assert!(doc_count > 0, "fixture must have stored docs");
        for doc_id in 0..doc_count {
            let all = read_stored_fields(fdx, fdt, &fields, doc_id).unwrap();
            for (num, fi) in fields.iter().enumerate() {
                let one = read_stored_field(fdx, fdt, doc_id, num).unwrap();
                assert_eq!(
                    one.as_deref(),
                    all.get(&fi.name).map(String::as_str),
                    "doc {doc_id} field {} (num {num})",
                    fi.name
                );
            }
        }
    }

    #[test]
    fn read_stored_field_returns_the_first_of_a_repeated_field() {
        // A doc may carry the same field twice (`Document::add` called twice for one name).
        // The `.fdt` keeps both, in write order, so the reader must resolve to the FIRST and
        // stop — which is what makes field sort place such a doc under a defined value instead
        // of whichever entry happened to be scanned last. Hand-built bytes because neither
        // fixture has a repeated field, and `HashMap`-backed readers cannot express one.
        use crate::zsl::bytes::{write_modified_utf8, write_vint};

        let mut fdt = Vec::new();
        write_vint(&mut fdt, 3); // three stored entries
        for (num, value) in [(7usize, "first"), (2, "other"), (7, "second")] {
            write_vint(&mut fdt, num as u64);
            fdt.push(0x00); // not tokenized, not binary
            write_modified_utf8(&mut fdt, value);
        }
        let fdx = 0u64.to_be_bytes(); // one doc, starting at offset 0 of the .fdt

        assert_eq!(
            read_stored_field(&fdx, &fdt, 0, 7).unwrap().as_deref(),
            Some("first")
        );
        // and the entry AFTER a skipped one still resolves, proving the skip landed correctly
        assert_eq!(
            read_stored_field(&fdx, &fdt, 0, 2).unwrap().as_deref(),
            Some("other")
        );
    }

    #[test]
    fn read_stored_field_is_none_out_of_range() {
        let cf = cfs();
        let names = cf.names();
        let find = |ext: &str| names.iter().find(|n| n.ends_with(ext)).unwrap().clone();
        let fdx = cf.sub(&find(".fdx")).unwrap();
        let fdt = cf.sub(&find(".fdt")).unwrap();
        let fields = read_field_infos(cf.sub(&find(".fnm")).unwrap()).unwrap();

        // a field number past the schema: the doc exists, the field does not
        assert_eq!(
            read_stored_field(fdx, fdt, 0, fields.len() + 99).unwrap(),
            None
        );
        // a doc past the end of the .fdx: not an error, just nothing (same contract as
        // `read_stored_raw`, which the FFI relies on to degrade instead of panicking)
        assert_eq!(read_stored_field(fdx, fdt, fdx.len(), 0).unwrap(), None);
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
