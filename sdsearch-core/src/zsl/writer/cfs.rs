//! Compound file (.cfs) writer. Inverse of `zsl::cfs`. Mirrors `SegmentWriter::_generateCFS`:
//! `VInt(fileCount)` + a directory of `[Long(offset) + String(fullName)]` per sub-file, then
//! the data blocks with the offsets back-patched. The full name is
//! `<segmentName><ext>` (e.g. `_0.fnm`). ZSL's file order is
//! `.fdx .fdt .fnm .nrm .tis .tii .frq .prx` (stored first, then fnm/nrm, then dict).

use crate::zsl::bytes::{write_i64_be, write_modified_utf8, write_vint};

/// Packs the `(ext, data)` sub-files into a `.cfs`. `segment_name` is the prefix
/// (e.g. `"_0"`); the name in the directory is `segment_name + ext`.
pub fn write_cfs(segment_name: &str, files: &[(&str, &[u8])]) -> Vec<u8> {
    let mut out = Vec::new();
    write_vint(&mut out, files.len() as u64);

    // directory: per file a Long placeholder (offset) + the full name.
    let mut ptr_positions = Vec::with_capacity(files.len());
    for (ext, _) in files {
        ptr_positions.push(out.len());
        write_i64_be(&mut out, 0); // placeholder, back-patched below
        write_modified_utf8(&mut out, &format!("{segment_name}{ext}"));
    }

    // data: fix the real offset (back-patch in the buffer) and append the bytes.
    for (i, (_, data)) in files.iter().enumerate() {
        let data_offset = out.len() as u64;
        out[ptr_positions[i]..ptr_positions[i] + 8].copy_from_slice(&data_offset.to_be_bytes());
        out.extend_from_slice(data);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zsl::cfs::CompoundFile;
    use std::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    fn temp_path() -> std::path::PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("sdsearch_cfs_{}_{}.cfs", std::process::id(), n))
    }

    #[test]
    fn cfs_roundtrips_sub_files_through_reader() {
        let fnm = b"field-infos-bytes".to_vec();
        let tis = b"term-dict-bytes-longer".to_vec();
        let frq = vec![0u8, 1, 2, 3, 4];
        let files: Vec<(&str, &[u8])> =
            vec![(".fnm", &fnm), (".tis", &tis), (".frq", &frq)];

        let cfs = write_cfs("_7", &files);

        let path = temp_path();
        std::fs::write(&path, &cfs).unwrap();
        let cf = CompoundFile::open(&path).unwrap();

        let names = cf.names();
        assert!(names.contains(&"_7.fnm".to_string()), "names={names:?}");
        assert!(names.contains(&"_7.tis".to_string()));
        assert!(names.contains(&"_7.frq".to_string()));

        assert_eq!(cf.sub("_7.fnm").unwrap(), &fnm[..]);
        assert_eq!(cf.sub("_7.tis").unwrap(), &tis[..]);
        assert_eq!(cf.sub("_7.frq").unwrap(), &frq[..]);

        std::fs::remove_file(&path).ok();
    }
}
