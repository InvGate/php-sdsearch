//! Durable writes of the index's control files. Atomic rename ALWAYS (eliminates torn writes
//! of segments.gen, covers kill -9); fsync is added only on the optimize() flips. `std`-only
//! (Windows-safe): std::fs::rename replaces atomically on Unix and Windows. A single writer
//! holds the write-lock, so the `.tmp` suffix never collides.

use std::io::Write;
use std::path::Path;

/// Atomic replace via `<path>.tmp` + `rename` (NO fsync). Covers torn writes (`kill -9`) via
/// the atomic rename. Does NOT guarantee durability of the CONTENT across a power loss on file
/// systems that reorder metadata; on ext4 `data=ordered` and NTFS the rename is effectively
/// durable. For the `segments.gen` flip in the incremental commit this is enough: a commit lost
/// to a power cut is re-indexed on the next run (reconciliation by last_update). The `optimize()`
/// flips use `write_durable` (fsync) because recomputing the merge is more expensive.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let file_name = path
        .file_name()
        .ok_or_else(|| std::io::Error::other("path has no file_name"))?;
    let tmp = path.with_file_name(format!("{}.tmp", file_name.to_string_lossy()));
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Writes `bytes` to `<path>.tmp`, does `fsync` (`File::sync_all`) and renames to `path`
/// (atomic replace). fsync + rename guarantees that, after a crash, `path` holds either the old
/// content or the new content COMPLETE and durable — never a partial write. `std`-only
/// (Windows-safe: `File::sync_all` == `FlushFileBuffers`, `rename` == `MoveFileEx`).
/// A single writer holds the write-lock, so the `.tmp` suffix never collides.
pub(crate) fn write_durable(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let file_name = path
        .file_name()
        .ok_or_else(|| std::io::Error::other("path has no file_name"))?;
    let tmp = path.with_file_name(format!("{}.tmp", file_name.to_string_lossy()));
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?; // fsync: the bytes are on disk before the rename
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    #[test]
    fn writes_content_and_leaves_no_tmp() {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("sdsearch_dur_{}_{}", std::process::id(), n));
        std::fs::create_dir_all(&dir).unwrap();
        let target = dir.join("segments.gen");
        write_atomic(&target, b"hello").unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"hello");
        assert!(!dir.join("segments.gen.tmp").exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn overwrites_existing_file() {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("sdsearch_dur_{}_{}", std::process::id(), n));
        std::fs::create_dir_all(&dir).unwrap();
        let target = dir.join("segments.gen");
        std::fs::write(&target, b"old").unwrap();
        write_atomic(&target, b"new-longer").unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"new-longer");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn write_durable_writes_content_and_leaves_no_tmp() {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("sdsearch_durb_{}_{}", std::process::id(), n));
        std::fs::create_dir_all(&dir).unwrap();
        let target = dir.join("segments_5");
        write_durable(&target, b"payload").unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"payload");
        assert!(!dir.join("segments_5.tmp").exists());
        // overwrite
        write_durable(&target, b"payload-2-longer").unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"payload-2-longer");
        std::fs::remove_dir_all(&dir).ok();
    }
}
