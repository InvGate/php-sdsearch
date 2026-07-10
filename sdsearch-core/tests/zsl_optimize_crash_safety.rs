//! Crash-safety of optimize: the merged segment is referenced only AFTER the atomic
//! `segments.gen` flip. Models the two crash scenarios deterministically: `segments.gen` is
//! the atomic pointer, flipped last, so any half-written `segments_{N+1}`/`.cfs` is ignored
//! until the flip, and any `.cfs` not listed in `segments_N` is ignored post-flip.

use sdsearch_core::index::IndexReader;
use sdsearch_core::zsl::index::ZslIndex;
use sdsearch_core::zsl::writer::{IndexWriter, WriterDoc, WriterField, WriterOpts};
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn temp_kb_full() -> std::path::PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("sdsearch_crash_{}_{}", std::process::id(), n));
    std::fs::create_dir_all(&dir).unwrap();
    let src = std::path::PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/zsl_index_kb"
    ));
    for entry in std::fs::read_dir(&src).unwrap() {
        let p = entry.unwrap().path();
        std::fs::copy(&p, dir.join(p.file_name().unwrap())).unwrap();
    }
    dir
}

fn doc(i: usize) -> WriterDoc {
    WriterDoc { fields: vec![WriterField::text("title", &format!("zqxc unique{i}"))] }
}

/// lowercase base36 (== ZSL) — replicated here because `to_base36` is internal to the crate.
fn base36(mut n: u64) -> String {
    if n == 0 {
        return "0".to_string();
    }
    const D: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut b = Vec::new();
    while n > 0 {
        b.push(D[(n % 36) as usize]);
        n /= 36;
    }
    b.reverse();
    String::from_utf8(b).unwrap()
}

#[test]
fn interrupted_before_gen_flip_reader_keeps_old_generation() {
    let dir = temp_kb_full();

    // multi-seg base (gen advances via the commit)
    let opts = WriterOpts { max_buffered_docs: 2, ..WriterOpts::default() };
    let mut w = IndexWriter::open(&dir, opts).unwrap();
    for i in 0..4 {
        w.add_document(doc(i)).unwrap();
    }
    let rep = w.commit().unwrap();
    let cur_gen = rep.generation; // segments.gen points here
    let before = ZslIndex::open(&dir).unwrap().num_docs(); // 20 + 4

    // simulate a crash IN THE MIDDLE of optimize, before the flip: plant a segments_{cur+1} and a
    // half-merged `.cfs`, WITHOUT touching segments.gen (still at cur_gen).
    std::fs::write(
        dir.join(format!("segments_{}", base36(cur_gen + 1))),
        b"half-written-generation",
    )
    .unwrap();
    std::fs::write(dir.join("_z.cfs"), b"half-written-merged-cfs").unwrap();

    // the reader reads segments.gen -> cur_gen -> segments_{cur} (old, intact) and IGNORES the orphan.
    let idx = ZslIndex::open(&dir).unwrap();
    assert_eq!(idx.num_docs(), before);
    assert_eq!(idx.doc_freq("title", "zqxc"), 4);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn orphan_cfs_after_flip_is_ignored_by_reader() {
    let dir = temp_kb_full();

    // actually optimize an index with deletions => 1 valid merged segment.
    let mut w = IndexWriter::open(&dir, WriterOpts::default()).unwrap();
    w.delete_document(0);
    w.commit().unwrap();
    let w2 = IndexWriter::open(&dir, WriterOpts::default()).unwrap();
    w2.optimize().unwrap();
    let live = ZslIndex::open(&dir).unwrap().num_docs(); // 19

    // simulate a crash POST-flip, pre-unlink: an old orphan `.cfs` was left on disk.
    std::fs::write(dir.join("_2.cfs"), b"orphan-old-segment").unwrap();

    // the reader opens only the segments listed in segments_N => ignores `_2.cfs`.
    let idx = ZslIndex::open(&dir).unwrap();
    assert_eq!(idx.num_docs(), live);

    std::fs::remove_dir_all(&dir).ok();
}
