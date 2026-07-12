//! ZSL index write-lock: a NON-blocking exclusive lock on `write.lock.file`, released on Drop.
//! Mirrors Zend_Search_Lucene_LockManager::obtainWriteLock (const WRITE_LOCK_FILE,
//! flock(LOCK_EX)). Uses std's cross-platform file locking (Rust 1.89+): `flock(2)` on Unix
//! and `LockFileEx` on Windows — the SAME syscalls as PHP's flock() on each platform, so it
//! excludes another native writer AND ZSL. Taken in `IndexWriter::open`; a crash/panic releases
//! it (no stuck lock left behind).

use std::fs::{File, OpenOptions, TryLockError};
use std::path::Path;

#[derive(Debug)]
pub struct WriteLock {
    file: File, // keeping the File alive == holding the lock (released when dropped)
}

impl WriteLock {
    /// Takes a NON-blocking exclusive lock on `<dir>/write.lock.file`. `WouldBlock` error if
    /// already held (by another native writer or by ZSL).
    pub fn acquire(index_dir: &Path) -> std::io::Result<WriteLock> {
        let path = index_dir.join("write.lock.file");
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)?;
        match file.try_lock() {
            Ok(()) => Ok(WriteLock { file }),
            Err(TryLockError::WouldBlock) => Err(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                "write lock already held",
            )),
            Err(TryLockError::Error(e)) => Err(e),
        }
    }
}

impl Drop for WriteLock {
    fn drop(&mut self) {
        // best-effort: release the lock explicitly (it is released anyway when the File closes).
        let _ = self.file.unlock();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    fn temp_dir() -> std::path::PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("sdsearch_lock_{}_{}", std::process::id(), n));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn second_acquire_fails_while_held_then_succeeds_after_drop() {
        let dir = temp_dir();
        let lock = WriteLock::acquire(&dir).unwrap();
        // a second acquire on the same dir fails with WouldBlock (lock held)
        let err = WriteLock::acquire(&dir).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::WouldBlock);
        drop(lock);
        // released → can be re-acquired
        let _relock = WriteLock::acquire(&dir).unwrap();
        std::fs::remove_dir_all(&dir).ok();
    }
}
