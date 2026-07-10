//! After a crash mid-batch (kill -9 → Drop does not run), orphan .cfs files remain (not
//! listed in segments_N) and a stale write.lock.file (flock already released by the OS). The engine
//! must open cleanly: it ignores the orphan and re-locks the lock file without issue.

use sdsearch_core::index::IndexReader;
use sdsearch_core::zsl::index::ZslIndex;
use sdsearch_core::zsl::writer::{IndexWriter, WriterOpts};
use sdsearch_core::zsl::segments::read_segment_infos;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn temp_kb() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("sdsearch_unclean_{}_{}", std::process::id(), n));
    std::fs::create_dir_all(&dir).unwrap();
    let src = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/zsl_index_kb"));
    for f in std::fs::read_dir(&src).unwrap() {
        let f = f.unwrap().path();
        let name = f.file_name().unwrap().to_string_lossy();
        if name.contains("lock") || name.ends_with(".sti") { continue; }
        std::fs::copy(&f, dir.join(f.file_name().unwrap())).unwrap();
    }
    dir
}

#[test]
fn opens_clean_with_orphan_cfs_and_stale_lock_file() {
    let dir = temp_kb();
    let base = ZslIndex::open(&dir).unwrap().num_docs();

    // simulate a crash: an orphan .cfs NOT listed in segments_N + a stale write.lock.file.
    // the orphan's bytes = a copy of an existing valid .cfs under an unreferenced name.
    let existing_cfs = std::fs::read_dir(&dir).unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.extension().map(|x| x == "cfs").unwrap_or(false))
        .expect("the KB fixture has a .cfs");
    std::fs::copy(&existing_cfs, dir.join("_99.cfs")).unwrap(); // orphan (not in segments_N)
    std::fs::write(dir.join("write.lock.file"), b"").unwrap();   // stale lock file (not held)

    // the reader ignores the orphan and sees exactly the base docs.
    let reader = ZslIndex::open(&dir).unwrap();
    assert_eq!(reader.num_docs(), base, "the orphan _99.cfs must not be counted");
    // segments_N does not list the orphan.
    let infos = read_segment_infos(&dir).unwrap();
    assert!(!infos.iter().any(|s| s.name == "_99"), "_99 should not be in segments_N");

    // the writer opens: the stale lock file is NOT held → acquire OK.
    let w = IndexWriter::open(&dir, WriterOpts::default()).expect("writer must open despite the leftover lock file");
    drop(w);

    std::fs::remove_dir_all(&dir).ok();
}
