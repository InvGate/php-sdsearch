//! Stateful (streaming) writer for the ZSL index: open → add_document* → commit. Buffers docs
//! and flushes a segment every `max_buffered_docs`, bounding RAM (unlike the batch append,
//! which retains the whole batch). Maps 1:1 to the host application's indexing loop (open the
//! index once → for each doc: add → optimize once). Reuses `write_segment_cfs` as the flush
//! primitive; commits all flushed segments in ONE generation update. Takes the write-lock on
//! open (excludes another native writer and ZSL).

use super::lock::WriteLock;
use super::segments::{self, Generation, NewSegment};
use super::merge;
use super::{write_segment_cfs, WriterDoc, WriterOpts};
use crate::index::IndexReader;
use crate::zsl::deletes::DeletedDocs;
use crate::zsl::index::ZslIndex;
use crate::zsl::segments::read_segment_infos;
use crate::zsl::writer::deletes::write_del_file;
use std::collections::{BTreeSet, HashMap};
use std::io;
use std::path::{Path, PathBuf};

pub struct IndexWriter {
    dir: PathBuf,
    _lock: WriteLock,      // released on Drop
    base: Generation,      // snapshot of generation N at open()
    base_live_docs: usize, // live docs of generation N (Σ maxDoc − deleted)
    /// base generation's segment table: (name, maxDoc, delGen), in segments_N order.
    base_segments: Vec<(String, u32, i64)>,
    /// buffered deletions keyed by base segment name (ids LOCAL to the segment).
    pending_deletes: HashMap<String, BTreeSet<usize>>,
    next_name_counter: u32,   // starts at base.name_counter, ++ per flush
    flushed: Vec<NewSegment>, // segments flushed in this session (not yet committed)
    buffer: Vec<WriterDoc>,   // in-RAM docs not yet flushed
    max_buffered_docs: usize,
    opts: WriterOpts,
}

/// result of a commit: the resulting generation and the new segments it lists.
#[derive(Debug, Clone, PartialEq)]
pub struct CommitReport {
    pub generation: u64,
    pub segments: Vec<String>,
    pub doc_count: usize,
}

impl IndexWriter {
    /// Opens a streaming writer over an existing ZSL index. Takes the write-lock (error if
    /// already held), snapshots the current generation and its live-doc count.
    pub fn open(dir: &Path, opts: WriterOpts) -> io::Result<IndexWriter> {
        let lock = WriteLock::acquire(dir)?;
        let base = segments::read_generation(dir)?;
        // live-doc count of generation N (for document_count). The reader is dropped here.
        let base_live_docs = ZslIndex::open(dir)?.num_docs();
        let base_segments: Vec<(String, u32, i64)> = read_segment_infos(dir)?
            .into_iter()
            .map(|s| (s.name, s.doc_count as u32, s.del_gen))
            .collect();
        let next_name_counter = base.name_counter;
        let max_buffered_docs = opts.max_buffered_docs.max(1);
        Ok(IndexWriter {
            dir: dir.to_path_buf(),
            _lock: lock,
            base,
            base_live_docs,
            base_segments,
            pending_deletes: HashMap::new(),
            next_name_counter,
            flushed: Vec::new(),
            buffer: Vec::new(),
            max_buffered_docs,
            opts,
        })
    }

    /// Buffers a doc; if the buffer reached `max_buffered_docs`, flushes a segment.
    pub fn add_document(&mut self, doc: WriterDoc) -> io::Result<()> {
        self.buffer.push(doc);
        if self.buffer.len() >= self.max_buffered_docs {
            self.flush_segment()?;
        }
        Ok(())
    }

    /// Marks a doc of the base snapshot (generation N) as deleted. `global_doc_id` is global over
    /// the base segments (Σ maxDoc, incl. deleted). Out of range => silent no-op (matching ZSL's
    /// behavior). The deletion is materialized on commit(). Idempotent.
    pub fn delete_document(&mut self, global_doc_id: usize) {
        let mut base = 0usize;
        for (name, max_doc, _del_gen) in &self.base_segments {
            let seg_len = *max_doc as usize;
            if global_doc_id < base + seg_len {
                let local = global_doc_id - base;
                self.pending_deletes
                    .entry(name.clone())
                    .or_default()
                    .insert(local);
                return;
            }
            base += seg_len;
        }
        // outside [0, base_total): silent no-op.
    }

    /// Total docs the index will see after committing: base-live + flushed + buffered,
    /// minus the buffered deletes (idempotent: each base doc counts only once).
    /// NOTE: if a buffered `gid` was already deleted in the base, this count still subtracts it
    /// (over-counting deletes). The host application never re-deletes an already-deleted doc, so this never manifests.
    pub fn document_count(&self) -> usize {
        let flushed: usize = self.flushed.iter().map(|s| s.doc_count as usize).sum();
        let pending: usize = self.pending_deletes.values().map(|s| s.len()).sum();
        (self.base_live_docs + flushed + self.buffer.len()).saturating_sub(pending)
    }

    /// Flushes the buffer to ONE `_<k>.cfs` (invisible until commit). No-op if the buffer is empty.
    fn flush_segment(&mut self) -> io::Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let seg_name = segments::segment_name(self.next_name_counter);
        let doc_count = write_segment_cfs(&self.dir, &seg_name, &self.buffer, &self.opts)?;
        self.flushed.push(NewSegment {
            name: seg_name,
            doc_count: doc_count as u32,
        });
        self.next_name_counter += 1;
        self.buffer.clear();
        Ok(())
    }

    /// Flushes the remaining buffer and commits: writes ONE segments_{N+1} listing the flushed
    /// segments + the bumped delGens, and flips segments.gen. Empty commit = no-op.
    pub fn commit(mut self) -> io::Result<CommitReport> {
        self.commit_inner()
    }

    /// Materializes pending adds/deletes and, if the index has >1 segment or any deletion,
    /// merges everything into ONE compacted segment. Durability: fsync of the merged `.cfs`
    /// and of `segments_{N+1}` BEFORE the atomic flip; orphan cleanup POST-flip.
    /// Consumes the writer (releases the write-lock on return).
    pub fn optimize(mut self) -> io::Result<CommitReport> {
        // 1) materialize what's pending (flush the buffer + deletes). Reuses the commit logic.
        let commit_rep = self.commit_inner()?;

        // 2) is a merge needed? (== Zend_Search_Lucene::optimize: segCount>1 || hasDeletions)
        let infos = read_segment_infos(&self.dir)?;
        let has_deletes = infos.iter().any(|s| s.del_gen != -1);
        if infos.len() <= 1 && !has_deletes {
            // no-op: already 1 segment with no deletions. Report the index's real total,
            // not the session doc_count (which is 0 if nothing was added). The indexing feed uses
            // Writer::document_count(), but the field's contract must still be correct.
            let total: usize = infos.iter().map(|s| s.doc_count).sum();
            return Ok(CommitReport {
                doc_count: total,
                ..commit_rep
            });
        }

        // 3+4) stream the merge straight to a DURABLE {merged_name}.cfs (fsync happens inside).
        // Peak heap is bounded — postings/positions/stored are streamed through temp files — and
        // the bytes are identical to merge_segments. mmaps are dropped on return, so on Windows
        // the old .cfs can be unlinked afterwards. Name = next from name_counter.
        let gen = segments::read_generation(&self.dir)?;
        let merged_name = segments::segment_name(gen.name_counter);
        let refs: Vec<(String, i64)> = infos.iter().map(|s| (s.name.clone(), s.del_gen)).collect();
        let doc_count = merge::merge_segments_streaming(&self.dir, &merged_name, &refs)?;

        // 5) write segments_{N+1} (fsync) + atomic flip of segments.gen.
        let new_gen =
            segments::write_optimized_generation(&self.dir, &gen, &merged_name, doc_count as u32)?;

        // 6) orphan cleanup POST-flip (best-effort): .cfs/.del/.sti of the old segments.
        //    Never before the flip (crash-safety invariant: the merged segment is referenced only
        //    after the atomic segments.gen flip). The merged _<k> is not in `infos`.
        for info in &infos {
            let _ = std::fs::remove_file(self.dir.join(format!("{}.cfs", info.name)));
            let _ = std::fs::remove_file(self.dir.join(format!("{}.sti", info.name)));
            let del_name = match info.del_gen {
                -1 => None,
                0 => Some(format!("{}.del", info.name)),
                g => Some(format!(
                    "{}_{}.del",
                    info.name,
                    crate::zsl::segments::to_base36(g as u64)
                )),
            };
            if let Some(dn) = del_name {
                let _ = std::fs::remove_file(self.dir.join(dn));
            }
        }

        Ok(CommitReport {
            generation: new_gen,
            segments: vec![merged_name],
            doc_count,
        })
    }

    /// Flushes the remaining buffer and commits: writes ONE segments_{N+1} listing all the
    /// flushed segments and flips segments.gen. Empty commit (nothing flushed) = no-op.
    fn commit_inner(&mut self) -> io::Result<CommitReport> {
        self.flush_segment()?; // remaining buffer → segment

        let pending = std::mem::take(&mut self.pending_deletes);
        // take: after this self.flushed is empty, so Drop does NOT delete the already-committed
        // segments (it only cleans orphans from an abort).
        let flushed = std::mem::take(&mut self.flushed);

        if flushed.is_empty() && pending.is_empty() {
            // empty commit: no segment, no deletes, no flip.
            return Ok(CommitReport {
                generation: self.base.generation,
                segments: Vec::new(),
                doc_count: 0,
            });
        }

        // materialize deletes: for each touched base segment, union with its current .del and write
        // <seg>_<delGen+1>.del (dense); collect the delGen overrides for the new generation.
        let mut del_gen_overrides: HashMap<String, i64> = HashMap::new();
        for (seg_name, new_local_dels) in &pending {
            let (max_doc, cur_del_gen) = self
                .base_segments
                .iter()
                .find(|(n, _, _)| n == seg_name)
                .map(|(_, d, g)| (*d as usize, *g))
                .expect("touched base segment must exist in the base table");

            // union with the current .del (if any): re-read the raw bitmap and add its bits.
            let mut merged: BTreeSet<usize> = new_local_dels.clone();
            if cur_del_gen != -1 {
                let del_path = match cur_del_gen {
                    0 => self.dir.join(format!("{seg_name}.del")),
                    g => self.dir.join(format!(
                        "{seg_name}_{}.del",
                        crate::zsl::segments::to_base36(g as u64)
                    )),
                };
                let bytes = std::fs::read(&del_path)?;
                let dd = DeletedDocs::read(&bytes)?;
                for d in 0..max_doc {
                    if dd.is_deleted(d) {
                        merged.insert(d);
                    }
                }
            }

            let next_gen = if cur_del_gen == -1 {
                1
            } else {
                cur_del_gen + 1
            };
            let del_fname = format!(
                "{seg_name}_{}.del",
                crate::zsl::segments::to_base36(next_gen as u64)
            );
            let del_bytes = write_del_file(max_doc, &merged);
            super::durability::write_atomic(&self.dir.join(&del_fname), &del_bytes)?;
            del_gen_overrides.insert(seg_name.clone(), next_gen);
        }

        let doc_count: usize = flushed.iter().map(|s| s.doc_count as usize).sum();
        let segments: Vec<String> = flushed.iter().map(|s| s.name.clone()).collect();
        let generation = segments::write_generation_with_delgens(
            &self.dir,
            &self.base,
            &del_gen_overrides,
            &flushed,
        )?;

        Ok(CommitReport {
            generation,
            segments,
            doc_count,
        })
    }
}

impl Drop for IndexWriter {
    fn drop(&mut self) {
        // abort without commit: the flushed .cfs are left orphaned (not referenced by any
        // generation → harmless, GC-eligible by ZSL). Best-effort: we delete them. After a
        // commit, `flushed` was emptied (mem::take) → this is a no-op and the segments survive.
        for seg in &self.flushed {
            let _ = std::fs::remove_file(self.dir.join(format!("{}.cfs", seg.name)));
        }
        // `_lock` is released in its own Drop.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zsl::writer::WriterField;
    use std::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    /// copies the ENTIRE KB fixture (incl. `_2.cfs`) to a fresh temp dir.
    fn temp_kb_full() -> std::path::PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("sdsearch_iw_{}_{}", std::process::id(), n));
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

    /// doc with a unique term `zqxmark` in `title` (for doc_freq) + a per-index suffix.
    fn doc_mark(i: usize) -> WriterDoc {
        WriterDoc {
            fields: vec![WriterField::text("title", &format!("zqxmark unique{i}"))],
        }
    }

    #[test]
    fn open_takes_lock_second_open_fails_until_drop() {
        let dir = temp_kb_full();
        let w = IndexWriter::open(&dir, WriterOpts::default()).unwrap();
        assert!(IndexWriter::open(&dir, WriterOpts::default()).is_err()); // lock held
        drop(w);
        let _w2 = IndexWriter::open(&dir, WriterOpts::default()).unwrap(); // released
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn document_count_tracks_base_plus_flushed_plus_buffer() {
        let dir = temp_kb_full();
        let opts = WriterOpts {
            max_buffered_docs: 2,
            ..WriterOpts::default()
        };
        let mut w = IndexWriter::open(&dir, opts).unwrap();
        assert_eq!(w.document_count(), 20); // base KB live docs

        w.add_document(doc_mark(0)).unwrap(); // buffer=1
        assert_eq!(w.document_count(), 21);
        w.add_document(doc_mark(1)).unwrap(); // buffer reaches 2 → flush (seg doc_count=2), buffer=0
        assert_eq!(w.document_count(), 22);
        w.add_document(doc_mark(2)).unwrap(); // buffer=1
        assert_eq!(w.document_count(), 23);

        // one segment flushed to disk but generation intact (invisible until commit)
        assert!(dir.join("_3.cfs").exists());
        assert_eq!(ZslIndex::open(&dir).unwrap().num_docs(), 20);

        drop(w);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn commit_makes_all_flushed_segments_visible_to_reader() {
        let dir = temp_kb_full();
        let before = ZslIndex::open(&dir).unwrap().num_docs();
        assert_eq!(before, 20);

        let opts = WriterOpts {
            max_buffered_docs: 2,
            ..WriterOpts::default()
        };
        let mut w = IndexWriter::open(&dir, opts).unwrap();
        for i in 0..5 {
            w.add_document(doc_mark(i)).unwrap(); // cap 2 → flushes after 2 and 4; buffer=1
        }
        let rep = w.commit().unwrap(); // flush the remaining 1 → 3 segments: _3,_4,_5
        assert_eq!(rep.doc_count, 5);
        assert_eq!(
            rep.segments,
            vec!["_3".to_string(), "_4".to_string(), "_5".to_string()]
        );
        assert_eq!(rep.generation, 7);

        // the native reader sees base + 5 docs, the term spread across the 3 segments
        let idx = ZslIndex::open(&dir).unwrap();
        assert_eq!(idx.num_docs(), before + 5);
        assert_eq!(idx.doc_freq("title", "zqxmark"), 5);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn drop_without_commit_untouched_generation_and_cleans_orphans() {
        let dir = temp_kb_full();
        let opts = WriterOpts {
            max_buffered_docs: 1,
            ..WriterOpts::default()
        };
        {
            let mut w = IndexWriter::open(&dir, opts).unwrap();
            w.add_document(doc_mark(0)).unwrap(); // cap 1 → immediate flush: _3.cfs
            assert!(dir.join("_3.cfs").exists());
            // no commit: goes out of scope → Drop
        }
        assert!(!dir.join("_3.cfs").exists()); // orphan cleaned by Drop
        assert!(!dir.join("segments_7").exists()); // generation NOT flipped
        assert_eq!(ZslIndex::open(&dir).unwrap().num_docs(), 20); // intact
                                                                  // lock released → re-open OK
        let _w = IndexWriter::open(&dir, WriterOpts::default()).unwrap();
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn delete_document_hides_a_base_doc_from_reader() {
        let dir = temp_kb_full();
        let before = ZslIndex::open(&dir).unwrap().num_docs();
        assert_eq!(before, 20);

        let mut w = IndexWriter::open(&dir, WriterOpts::default()).unwrap();
        w.delete_document(5); // global doc 5 (segment _2, local id 5)
        let rep = w.commit().unwrap();
        assert_eq!(rep.generation, 7);

        let idx = ZslIndex::open(&dir).unwrap();
        assert_eq!(idx.num_docs(), before - 1); // one fewer live doc

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn delete_out_of_base_range_is_a_silent_noop() {
        let dir = temp_kb_full();
        let mut w = IndexWriter::open(&dir, WriterOpts::default()).unwrap();
        w.delete_document(9999); // outside [0, 20) -> no-op
        let rep = w.commit().unwrap();
        assert_eq!(rep.doc_count, 0);
        assert!(rep.segments.is_empty());
        assert_eq!(rep.generation, 6); // KB generation unchanged (no effective delete)
        assert!(!dir.join("segments_7").exists());
        assert_eq!(ZslIndex::open(&dir).unwrap().num_docs(), 20);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn delete_and_add_in_one_commit() {
        let dir = temp_kb_full();
        let mut w = IndexWriter::open(&dir, WriterOpts::default()).unwrap();
        w.delete_document(0);
        w.delete_document(0); // idempotent
        w.add_document(doc_mark(0)).unwrap();
        let rep = w.commit().unwrap();
        assert_eq!(rep.doc_count, 1); // 1 new doc
        let idx = ZslIndex::open(&dir).unwrap();
        assert_eq!(idx.num_docs(), 20 - 1 + 1); // -1 deleted, +1 added
        assert_eq!(idx.doc_freq("title", "zqxmark"), 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn delete_across_two_base_segments_in_one_commit() {
        let dir = temp_kb_full();

        // step 1: build a multi-segment base: _2 (KB, 20 docs) + _3,_4 (4 new docs).
        let opts = WriterOpts {
            max_buffered_docs: 2,
            ..WriterOpts::default()
        };
        let mut w1 = IndexWriter::open(&dir, opts).unwrap();
        for i in 0..4 {
            w1.add_document(doc_mark(i)).unwrap(); // cap 2 → flush after 2 and after 4: _3, _4
        }
        let rep1 = w1.commit().unwrap();
        assert_eq!(rep1.segments, vec!["_3".to_string(), "_4".to_string()]);

        let before = ZslIndex::open(&dir).unwrap().num_docs();
        assert_eq!(before, 20 + 4); // base KB + the 4 just committed

        // step 2: new writer over the now multi-segment base. Deletes a doc in _2 (gid<20) and
        // a doc in _3 (gid in [20,22)) in the SAME commit.
        let mut w2 = IndexWriter::open(&dir, WriterOpts::default()).unwrap();
        w2.delete_document(3); // segment _2, local id 3
        w2.delete_document(20); // segment _3, local id 0 (first flushed doc)
        let rep2 = w2.commit().unwrap();
        assert_eq!(rep2.doc_count, 0); // no new docs, only deletes
        assert!(rep2.segments.is_empty());

        // the deletion must be reflected: 2 fewer live docs, no more no less.
        let idx = ZslIndex::open(&dir).unwrap();
        assert_eq!(idx.num_docs(), before - 2);

        // exactly the two touched segments (_2, _3) have their delGen bumped to 1; the
        // untouched segment (_4) stays at -1.
        let infos = read_segment_infos(&dir).unwrap();
        let del_gen_of = |name: &str| infos.iter().find(|s| s.name == name).unwrap().del_gen;
        assert_eq!(del_gen_of("_2"), 1);
        assert_eq!(del_gen_of("_3"), 1);
        assert_eq!(del_gen_of("_4"), -1);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn optimize_collapses_multi_segment_with_deletes_to_one() {
        let dir = temp_kb_full();

        // multi-seg base + deletes
        let opts = WriterOpts {
            max_buffered_docs: 2,
            ..WriterOpts::default()
        };
        let mut w = IndexWriter::open(&dir, opts).unwrap();
        for i in 0..4 {
            w.add_document(doc_mark(i)).unwrap();
        }
        w.commit().unwrap();
        let mut w2 = IndexWriter::open(&dir, WriterOpts::default()).unwrap();
        w2.delete_document(0);
        w2.delete_document(20);
        w2.commit().unwrap();

        let live = ZslIndex::open(&dir).unwrap().num_docs(); // 22
        assert_eq!(read_segment_infos(&dir).unwrap().len(), 3); // _2,_3,_4

        // OPTIMIZE
        let w3 = IndexWriter::open(&dir, WriterOpts::default()).unwrap();
        let rep = w3.optimize().unwrap();

        // a single segment, no deletions, live docs preserved
        let infos = read_segment_infos(&dir).unwrap();
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].del_gen, -1);
        assert_eq!(rep.doc_count, live);
        let idx = ZslIndex::open(&dir).unwrap();
        assert_eq!(idx.num_docs(), live);
        assert_eq!(idx.doc_freq("title", "zqxmark"), 3); // 4 - 1 deleted

        // orphan cleanup: the old .cfs were deleted; the merged one exists
        assert!(!dir.join("_2.cfs").exists());
        assert!(!dir.join("_3.cfs").exists());
        assert!(!dir.join("_4.cfs").exists());
        assert!(dir.join(format!("{}.cfs", infos[0].name)).exists());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn optimize_single_segment_no_deletes_is_noop() {
        let dir = temp_kb_full(); // KB: 1 segment, no deletions, gen 6
        let w = IndexWriter::open(&dir, WriterOpts::default()).unwrap();
        let rep = w.optimize().unwrap();
        // nothing to merge: generation intact
        assert_eq!(rep.generation, 6);
        assert_eq!(read_segment_infos(&dir).unwrap().len(), 1);
        assert_eq!(ZslIndex::open(&dir).unwrap().num_docs(), 20);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn optimize_noop_reports_real_doc_count() {
        let dir = temp_kb_full(); // KB: 1 segment, no deletions
        let before = ZslIndex::open(&dir).unwrap().num_docs();
        let w = IndexWriter::open(&dir, WriterOpts::default()).unwrap();
        // KB is 1 segment with no deletions → optimize is a no-op.
        let rep = w.optimize().unwrap();
        assert_eq!(
            rep.doc_count, before,
            "optimize no-op must report the index total"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn optimize_single_segment_with_deletes_removes_del_file() {
        let dir = temp_kb_full();
        let mut w = IndexWriter::open(&dir, WriterOpts::default()).unwrap();
        w.delete_document(5);
        w.commit().unwrap();
        assert!(ZslIndex::open(&dir).unwrap().num_docs() == 19);

        // 1 segment BUT with deletions => optimize must run (collapses the .del)
        let w2 = IndexWriter::open(&dir, WriterOpts::default()).unwrap();
        w2.optimize().unwrap();
        let infos = read_segment_infos(&dir).unwrap();
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].del_gen, -1); // no .del after optimize
        assert_eq!(infos[0].doc_count, 19); // maxDoc == live docs (compacted)
        assert_eq!(ZslIndex::open(&dir).unwrap().num_docs(), 19);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn document_count_subtracts_pending_deletes() {
        let dir = temp_kb_full();
        let mut w = IndexWriter::open(&dir, WriterOpts::default()).unwrap();
        assert_eq!(w.document_count(), 20);
        w.delete_document(0);
        w.delete_document(1);
        w.delete_document(0); // idempotent: does not over-subtract
        assert_eq!(w.document_count(), 18);
        drop(w);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn empty_commit_is_a_noop() {
        let dir = temp_kb_full();
        let w = IndexWriter::open(&dir, WriterOpts::default()).unwrap();
        let rep = w.commit().unwrap();
        assert_eq!(rep.doc_count, 0);
        assert!(rep.segments.is_empty());
        assert_eq!(rep.generation, 6); // KB generation unchanged
        assert!(!dir.join("segments_7").exists());
        assert_eq!(ZslIndex::open(&dir).unwrap().num_docs(), 20);
        std::fs::remove_dir_all(&dir).ok();
    }
}
