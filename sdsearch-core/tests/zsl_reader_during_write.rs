//! A reader (ZslIndex::open) opens and queries fine WHILE an IndexWriter holds the
//! write-lock. Readers do NOT take the write-lock → searches coexist with an active writer.

use sdsearch_core::index::IndexReader;
use sdsearch_core::zsl::index::ZslIndex;
use sdsearch_core::zsl::writer::{IndexWriter, WriterOpts};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn temp_kb() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("sdsearch_rdwr_{}_{}", std::process::id(), n));
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
fn reader_opens_while_writer_holds_lock() {
    let dir = temp_kb();

    // writer opened → lock taken.
    let writer = IndexWriter::open(&dir, WriterOpts::default()).expect("writer open");

    // reader opens fine despite the lock, and sees the base generation's docs.
    let reader = ZslIndex::open(&dir).expect("reader should open with an active writer");
    assert!(reader.num_docs() > 0, "the reader should see the base docs");

    drop(writer);
    std::fs::remove_dir_all(&dir).ok();
}
