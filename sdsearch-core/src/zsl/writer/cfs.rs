//! Compound file (.cfs) writer. Inverse of `zsl::cfs`. Mirrors `SegmentWriter::_generateCFS`:
//! `VInt(fileCount)` + a directory of `[Long(offset) + String(fullName)]` per sub-file, then
//! the data blocks with the offsets back-patched. The full name is
//! `<segmentName><ext>` (e.g. `_0.fnm`). ZSL's file order is
//! `.fdx .fdt .fnm .nrm .tis .tii .frq .prx` (stored first, then fnm/nrm, then dict).

use crate::zsl::bytes::{write_i64_be, write_modified_utf8, write_vint};
use std::io::{self, Write};
use std::path::Path;

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

/// Source for one CFS sub-file fed to `write_cfs_streaming`: either an in-memory buffer or an
/// on-disk temp file. `Path` sources are streamed with a fixed-size buffer (`std::io::copy`),
/// never loaded fully into RAM.
pub enum CfsSource<'a> {
    Mem(&'a [u8]),
    Path(&'a Path),
}

impl CfsSource<'_> {
    fn len(&self) -> io::Result<u64> {
        match self {
            CfsSource::Mem(data) => Ok(data.len() as u64),
            CfsSource::Path(path) => Ok(std::fs::metadata(path)?.len()),
        }
    }

    /// Writes this source's bytes to `out`. `expected_len` must be the exact value that `len()`
    /// returned for this same source and that was used to compute its CFS directory offset. If
    /// the number of bytes actually copied differs (e.g. the underlying file was truncated or
    /// replaced between the `len()` call and this write — a TOCTOU race), this returns an
    /// `io::Error` instead of silently emitting a `.cfs` whose directory offsets don't match the
    /// data actually written.
    fn write_to<W: Write>(&self, out: &mut W, expected_len: u64) -> io::Result<()> {
        match self {
            CfsSource::Mem(data) => out.write_all(data),
            CfsSource::Path(path) => {
                let mut f = std::fs::File::open(path)?;
                let copied = io::copy(&mut f, out)?;
                if copied != expected_len {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        format!(
                            "cfs source {}: expected {expected_len} bytes (length used to \
                             compute the directory offset) but copied {copied}; file changed \
                             size between measurement and write",
                            path.display()
                        ),
                    ));
                }
                Ok(())
            }
        }
    }
}

/// Streaming counterpart to `write_cfs`: identical byte layout (`VInt(fileCount)` + directory
/// `[Long(offset) + String(fullName)]` + data blocks). Every source length is known up front
/// (`Mem`: slice len; `Path`: file metadata), so all directory offsets are computed directly and
/// the directory is written to `out` once — no back-patch into `out` is needed (the small
/// in-RAM directory buffer is patched before anything is written). Each data block is then
/// streamed into `out` in turn; `Path` sources are copied via `std::io::copy` and never loaded
/// fully into memory. Each `Path` source's length is measured once and that same value is used
/// both for its directory offset and to verify the number of bytes actually copied, so a source
/// that changes size between the two (e.g. a truncated temp file) is caught as an error rather
/// than silently producing a `.cfs` with offsets that don't match its data.
///
/// On `Err`, `out` may already contain a partial write (directory and/or some data blocks
/// written before the failing source was reached); callers must discard it and must not treat it
/// as a valid, committable `.cfs`.
pub fn write_cfs_streaming<W: Write>(
    out: &mut W,
    segment_name: &str,
    files: &[(&str, CfsSource)],
) -> io::Result<()> {
    // Build the directory in a small in-RAM buffer first: its length depends only on the file
    // count and names (not on the — possibly huge — data), so once built we know every data
    // block's final offset and can patch them here, still in RAM, before writing anything to
    // `out`.
    let mut dir = Vec::new();
    write_vint(&mut dir, files.len() as u64);
    let mut ptr_positions = Vec::with_capacity(files.len());
    for (ext, _) in files {
        ptr_positions.push(dir.len());
        write_i64_be(&mut dir, 0); // placeholder, patched below (in RAM, not yet in `out`)
        write_modified_utf8(&mut dir, &format!("{segment_name}{ext}"));
    }

    let mut offset = dir.len() as u64;
    let mut lengths = Vec::with_capacity(files.len());
    for (i, (_, src)) in files.iter().enumerate() {
        let len = src.len()?;
        dir[ptr_positions[i]..ptr_positions[i] + 8].copy_from_slice(&offset.to_be_bytes());
        offset += len;
        lengths.push(len);
    }

    out.write_all(&dir)?;
    for ((_, src), len) in files.iter().zip(lengths.iter()) {
        src.write_to(out, *len)?;
    }
    Ok(())
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

    fn temp_path_named(tag: &str) -> std::path::PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "sdsearch_cfs_{}_{}_{}.tmp",
            std::process::id(),
            n,
            tag
        ))
    }

    #[test]
    fn write_cfs_streaming_matches_write_cfs_for_mem_sources() {
        let fnm = b"field-infos-bytes".to_vec();
        let tis = b"term-dict-bytes-longer".to_vec();
        let frq = vec![0u8, 1, 2, 3, 4];
        let files: Vec<(&str, &[u8])> = vec![(".fnm", &fnm), (".tis", &tis), (".frq", &frq)];
        let expected = write_cfs("_7", &files);

        let streaming_files: Vec<(&str, CfsSource)> = vec![
            (".fnm", CfsSource::Mem(&fnm)),
            (".tis", CfsSource::Mem(&tis)),
            (".frq", CfsSource::Mem(&frq)),
        ];
        let mut actual = Vec::new();
        write_cfs_streaming(&mut actual, "_7", &streaming_files).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn write_cfs_streaming_matches_write_cfs_with_mixed_path_and_mem_sources() {
        let fnm = b"field-infos-bytes".to_vec();
        let tis = b"term-dict-bytes-longer-for-the-path-source-case".to_vec();
        let frq = vec![0u8, 1, 2, 3, 4];

        // All-Mem oracle: same sub-files, same order, produced by the existing `write_cfs`.
        let files: Vec<(&str, &[u8])> = vec![(".fnm", &fnm), (".tis", &tis), (".frq", &frq)];
        let expected = write_cfs("_9", &files);

        // Write the ".tis" data to a temp file and pass it as a `Path` source; the rest stay
        // `Mem`. Output must still be byte-identical to the all-Mem oracle above.
        let tis_path = temp_path_named("tis");
        std::fs::write(&tis_path, &tis).unwrap();

        let streaming_files: Vec<(&str, CfsSource)> = vec![
            (".fnm", CfsSource::Mem(&fnm)),
            (".tis", CfsSource::Path(&tis_path)),
            (".frq", CfsSource::Mem(&frq)),
        ];
        let mut actual = Vec::new();
        write_cfs_streaming(&mut actual, "_9", &streaming_files).unwrap();

        std::fs::remove_file(&tis_path).ok();

        assert_eq!(actual, expected);
    }

    #[test]
    fn write_to_errors_when_path_source_shrinks_after_len_was_measured() {
        let path = temp_path_named("toctou");
        std::fs::write(&path, b"this file starts out long enough to matter").unwrap();

        let src = CfsSource::Path(&path);
        // Same call `write_cfs_streaming` makes to compute the CFS directory offset.
        let measured_len = src.len().unwrap();

        // Simulate a TOCTOU race: the file is truncated/replaced by something else between the
        // `len()` call used for the directory offset and the actual data write.
        std::fs::write(&path, b"short").unwrap();

        let mut out = Vec::new();
        let err = src.write_to(&mut out, measured_len).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn write_to_succeeds_when_expected_len_matches_actual_file_size() {
        // Guard against the fix spuriously erroring on a legitimate full copy.
        let path = temp_path_named("full_copy");
        let data = b"a normal file that is not touched after measurement".to_vec();
        std::fs::write(&path, &data).unwrap();

        let src = CfsSource::Path(&path);
        let measured_len = src.len().unwrap();
        assert_eq!(measured_len, data.len() as u64);

        let mut out = Vec::new();
        src.write_to(&mut out, measured_len).unwrap();
        assert_eq!(out, data);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn cfs_roundtrips_sub_files_through_reader() {
        let fnm = b"field-infos-bytes".to_vec();
        let tis = b"term-dict-bytes-longer".to_vec();
        let frq = vec![0u8, 1, 2, 3, 4];
        let files: Vec<(&str, &[u8])> = vec![(".fnm", &fnm), (".tis", &tis), (".frq", &frq)];

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
