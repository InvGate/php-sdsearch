//! Native merge / `optimize()`. Collapses N segments (+ deletes) into the bytes of ONE
//! byte-faithful compacted segment. GOLDEN RULE: postings/positions/norm-bytes/stored are COPIED
//! and doc-ids are RENUMBERED densely; NEVER re-tokenized (avoids divergence from unrecoverable
//! boosts). Does not write to disk or touch the generation (`optimize()` does that).
//!
//! 4 phases (== `Zend_Search_Lucene_Index_SegmentMerger::merge`):
//!   1. fields: first-seen union while iterating segments in `segments_N` order.
//!   2. docs: per segment, per LIVE doc (skips `.del`), renumber to a dense increasing id;
//!      copies stored (remapping field_num) and the norm bytes per indexed field.
//!   3. terms: per (field, text) unions postings from all segments, remapping doc-ids;
//!      sorts by new doc-id; drops terms with no live docs.
//!   4. serializes reusing the `write_segment_cfs` primitives (norms COPIED verbatim via write_norms_raw).

use super::cfs::{write_cfs_streaming, CfsSource};
use super::invert::{FieldMeta, StoredField, TermPostings};
use super::{assemble_cfs, fnm, norms, stored, terms};
use crate::index::IndexReader;
use crate::zsl::segment::ZslSegment;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap};
use std::fs::File;
use std::io;
use std::path::Path;

/// merge result: the bytes of the merged `.cfs` + the number of live docs.
pub struct MergeResult {
    pub cfs_bytes: Vec<u8>,
    pub doc_count: usize,
}

/// Merges `segments` (name, del_gen) — in `segments_N` order — into a `.cfs` named
/// `merged_name`. See the module doc.
///
/// Retained as the differential-test oracle (see `assert_streaming_byte_identical` below) and
/// as a potential future small-index fallback; it is NOT on the production `optimize()` path,
/// which uses [`merge_segments_streaming`].
pub fn merge_segments(
    index_dir: &Path,
    merged_name: &str,
    segments: &[(String, i64)],
) -> io::Result<MergeResult> {
    // open all source segments (mmap). They are dropped when the function returns, so the
    // caller can unlink the old `.cfs` afterwards (Windows: no unlink of a mapped file).
    let segs: Vec<ZslSegment> = segments
        .iter()
        .map(|(name, dg)| ZslSegment::open_named(index_dir, name, *dg))
        .collect::<io::Result<_>>()?;

    // ---- phase 1: field union by first-seen ----
    let mut field_index: HashMap<String, usize> = HashMap::new();
    let mut fields: Vec<FieldMeta> = Vec::new();
    for seg in &segs {
        for fi in seg.field_infos() {
            let idx = *field_index.entry(fi.name.clone()).or_insert_with(|| {
                fields.push(FieldMeta {
                    name: fi.name.clone(),
                    indexed: false,
                });
                fields.len() - 1
            });
            fields[idx].indexed |= fi.is_indexed;
        }
    }

    // ---- phase 2: dense renumbering + copy of stored and norm bytes ----
    // doc_maps[si][local] = Some(new_id) if live, None if deleted.
    let mut doc_maps: Vec<Vec<Option<usize>>> = Vec::with_capacity(segs.len());
    let mut stored: Vec<Vec<StoredField>> = Vec::new();
    // norm_cols[merged_field] = column of bytes (empty if the field is not indexed).
    let mut norm_cols: Vec<Vec<u8>> = vec![Vec::new(); fields.len()];
    let mut next_id = 0usize;
    for seg in &segs {
        let local_fields = seg.field_infos();
        let mut map = vec![None; seg.max_doc()];
        // `local` is a semantic doc-id (indexes is_deleted/stored_raw/norm_bytes and feeds
        // doc_maps), not a mere iteration index: hence the range loop, not an iter.
        #[allow(clippy::needless_range_loop)]
        for local in 0..seg.max_doc() {
            if seg.is_deleted(local) {
                continue;
            }
            map[local] = Some(next_id);

            // stored: copy in order, remapping field_num local -> merged.
            let raw = seg.stored_raw(local)?;
            let mut remapped: Vec<StoredField> = Vec::with_capacity(raw.len());
            for r in raw {
                if r.is_binary {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "merge: binary stored field not supported (local field_num {})",
                            r.field_num
                        ),
                    ));
                }
                let name = &local_fields
                    .get(r.field_num)
                    .ok_or_else(|| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("merge: stored field_num {} out of range", r.field_num),
                        )
                    })?
                    .name;
                remapped.push(StoredField {
                    field_num: field_index[name],
                    value: r.value,
                    tokenized: r.tokenized,
                });
            }
            stored.push(remapped);

            // norms: for each indexed merged field, copy the segment's raw byte (255 if the
            // field does not exist / has no norm in this segment => lengthNorm(0), like ZSL).
            for (mf, field) in fields.iter().enumerate() {
                if !field.indexed {
                    continue;
                }
                let byte = seg
                    .norm_bytes(&field.name)
                    .and_then(|col| col.get(local).copied())
                    .unwrap_or(255);
                norm_cols[mf].push(byte);
            }
            next_id += 1;
        }
        doc_maps.push(map);
    }
    let doc_count = next_id;

    // ---- phase 3: term merge (copies postings/positions, remaps doc-ids) ----
    // invariant: a segment with indexed fields MUST have .prx; otherwise the merge would drop
    // positions silently. Fail loudly (the host application always writes .prx for indexed fields).
    for seg in &segs {
        let has_indexed = seg.field_infos().iter().any(|fi| fi.is_indexed);
        if has_indexed && !seg.has_prx() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "merge: segment with indexed fields but no .prx — positions unavailable",
            ));
        }
    }
    // (merged_field_num, text) -> (new_doc -> positions). BTreeMap => new doc-ids ascending.
    let mut term_map: HashMap<(usize, String), BTreeMap<usize, Vec<u32>>> = HashMap::new();
    for (si, seg) in segs.iter().enumerate() {
        for (field_name, text) in seg.all_terms() {
            let mf = field_index[&field_name];
            // positions_all ALREADY filters deletes and yields (local -> positions) in one pass.
            // Use `text` before moving it into the map key => no clone per (segment, term).
            let all = seg.positions_all(&field_name, &text);
            let entry = term_map.entry((mf, text)).or_default();
            for (local, positions) in all {
                if let Some(new_id) = doc_maps[si][local] {
                    entry.insert(new_id, positions);
                }
            }
        }
    }
    // terms with no live docs are dropped (== ZSL). Build sorted TermPostings.
    let mut merged_terms: Vec<TermPostings> = term_map
        .into_iter()
        .filter(|(_, docs)| !docs.is_empty())
        .map(|((field_num, text), docs)| TermPostings {
            field_num,
            text,
            docs: docs.into_iter().collect(), // BTreeMap => (doc_id, positions) ascending
        })
        .collect();
    // ZSL's order: `fieldName · \0 · text` (byte-wise). `\0` is the minimum byte => comparing the
    // (fieldName, text) tuple is equivalent (== writer::invert).
    merged_terms.sort_by(|a, b| {
        (fields[a.field_num].name.as_str(), a.text.as_str())
            .cmp(&(fields[b.field_num].name.as_str(), b.text.as_str()))
    });

    // ---- phase 4: serialize (norms COPIED, not re-encoded) ----
    let fnm_bytes = fnm::write_fnm(&fields);
    let (fdt, fdx) = stored::write_stored(&stored);
    let nrm = norms::write_norms_raw(&norm_cols);
    let dict = terms::write_term_dict(&merged_terms);
    let cfs_bytes = assemble_cfs(merged_name, &fnm_bytes, &fdt, &fdx, &nrm, &dict);

    Ok(MergeResult {
        cfs_bytes,
        doc_count,
    })
}

/// Streaming, bounded-memory counterpart of [`merge_segments`]. Reproduces its 4 phases EXACTLY
/// (byte-identical `.cfs`) but streams the big blocks (`.fdt`/`.frq`/`.prx`) through temp files
/// under `index_dir` instead of building them fully in RAM, and writes the merged `.cfs` durably
/// (fsync) to `index_dir/{merged_name}.cfs` itself before returning. Returns the live `doc_count`.
///
/// Only the small per-field/index blocks stay in RAM (`.fnm`/`.fdx`/`.nrm`/`.tis`/`.tii` +
/// `doc_maps`/`norm_cols`), so peak heap is independent of total text volume. The term merge is a
/// k-way merge across each segment's [`ZslSegment::term_cursor`] (already in ZSL canonical order),
/// so terms are fed to the streaming dict writer in the SAME order the batch path sorts them into.
///
/// Temp files (`{merged_name}.fdt.tmp` / `.frq.tmp` / `.prx.tmp`) are removed on BOTH the success
/// and error paths.
pub fn merge_segments_streaming(
    index_dir: &Path,
    merged_name: &str,
    segments: &[(String, i64)],
) -> io::Result<usize> {
    let fdt_tmp = index_dir.join(format!("{merged_name}.fdt.tmp"));
    let frq_tmp = index_dir.join(format!("{merged_name}.frq.tmp"));
    let prx_tmp = index_dir.join(format!("{merged_name}.prx.tmp"));

    let result = merge_streaming_inner(
        index_dir,
        merged_name,
        segments,
        &fdt_tmp,
        &frq_tmp,
        &prx_tmp,
    );

    // Clean up temp files on BOTH the success and error paths (best-effort).
    let _ = std::fs::remove_file(&fdt_tmp);
    let _ = std::fs::remove_file(&frq_tmp);
    let _ = std::fs::remove_file(&prx_tmp);

    result
}

fn merge_streaming_inner(
    index_dir: &Path,
    merged_name: &str,
    segments: &[(String, i64)],
    fdt_tmp: &Path,
    frq_tmp: &Path,
    prx_tmp: &Path,
) -> io::Result<usize> {
    let segs: Vec<ZslSegment> = segments
        .iter()
        .map(|(name, dg)| ZslSegment::open_named(index_dir, name, *dg))
        .collect::<io::Result<_>>()?;

    // ---- phase 1: field union by first-seen (identical to merge_segments) ----
    let mut field_index: HashMap<String, usize> = HashMap::new();
    let mut fields: Vec<FieldMeta> = Vec::new();
    for seg in &segs {
        for fi in seg.field_infos() {
            let idx = *field_index.entry(fi.name.clone()).or_insert_with(|| {
                fields.push(FieldMeta {
                    name: fi.name.clone(),
                    indexed: false,
                });
                fields.len() - 1
            });
            fields[idx].indexed |= fi.is_indexed;
        }
    }

    // ---- phase 2: dense renumbering; stream stored to a temp .fdt (fdx kept in RAM), norms in RAM ----
    let mut doc_maps: Vec<Vec<Option<usize>>> = Vec::with_capacity(segs.len());
    let mut norm_cols: Vec<Vec<u8>> = vec![Vec::new(); fields.len()];
    let mut next_id = 0usize;

    let mut fdx_buf: Vec<u8> = Vec::new();
    let mut stored_writer = stored::StoredStreamWriter::new(File::create(fdt_tmp)?, &mut fdx_buf);

    for seg in &segs {
        let local_fields = seg.field_infos();
        let mut map = vec![None; seg.max_doc()];
        // `local` is a semantic doc-id (indexes is_deleted/stored_raw/norm_bytes and feeds
        // doc_maps), not a mere iteration index: hence the range loop, not an iter.
        #[allow(clippy::needless_range_loop)]
        for local in 0..seg.max_doc() {
            if seg.is_deleted(local) {
                continue;
            }
            map[local] = Some(next_id);

            // stored: stream in order, remapping field_num local -> merged.
            let raw = seg.stored_raw(local)?;
            let mut remapped: Vec<StoredField> = Vec::with_capacity(raw.len());
            for r in raw {
                if r.is_binary {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "merge: binary stored field not supported (local field_num {})",
                            r.field_num
                        ),
                    ));
                }
                let name = &local_fields
                    .get(r.field_num)
                    .ok_or_else(|| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("merge: stored field_num {} out of range", r.field_num),
                        )
                    })?
                    .name;
                remapped.push(StoredField {
                    field_num: field_index[name],
                    value: r.value,
                    tokenized: r.tokenized,
                });
            }
            stored_writer.add_doc(&remapped)?;

            // norms: for each indexed merged field, copy the segment's raw byte (255 if absent).
            for (mf, field) in fields.iter().enumerate() {
                if !field.indexed {
                    continue;
                }
                let byte = seg
                    .norm_bytes(&field.name)
                    .and_then(|col| col.get(local).copied())
                    .unwrap_or(255);
                norm_cols[mf].push(byte);
            }
            next_id += 1;
        }
        doc_maps.push(map);
    }
    let doc_count = next_id;
    stored_writer.finish()?; // flush + close the temp .fdt; fdx_buf is now complete

    // ---- phase 3 guard: a segment with indexed fields MUST have .prx (identical to merge_segments) ----
    for seg in &segs {
        let has_indexed = seg.field_infos().iter().any(|fi| fi.is_indexed);
        if has_indexed && !seg.has_prx() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "merge: segment with indexed fields but no .prx — positions unavailable",
            ));
        }
    }

    // ---- phase 3: k-way term merge → streamed to temp .frq/.prx, .tis/.tii kept in RAM ----
    // .tis/.tii are Write+Seek (Cursor over a Vec, so the header term-counts can be back-patched);
    // .frq/.prx are the big blocks, streamed to temp files.
    let mut tis_buf = io::Cursor::new(Vec::new());
    let mut tii_buf = io::Cursor::new(Vec::new());
    let mut dict_writer = terms::TermDictStreamWriter::new(
        &mut tis_buf,
        &mut tii_buf,
        File::create(frq_tmp)?,
        File::create(prx_tmp)?,
    )?;

    // one cursor per segment; each yields (field, term) ascending. A min-heap (BinaryHeap is a
    // max-heap, so keys are wrapped in Reverse) drives the k-way merge. Keys are cloned into the
    // heap (bounded: at most one small (field, term) per segment at a time) because the cursors'
    // borrows end at each `peek`, so we cannot hold `&str` across an `advance`.
    let mut cursors: Vec<_> = segs.iter().map(|s| s.term_cursor()).collect();
    let mut heap: BinaryHeap<Reverse<(String, String, usize)>> = BinaryHeap::new();
    for (si, cur) in cursors.iter().enumerate() {
        if let Some((f, t)) = cur.peek() {
            heap.push(Reverse((f.to_string(), t.to_string(), si)));
        }
    }

    while let Some(Reverse((field, term, first_si))) = heap.pop() {
        // gather EVERY segment currently positioned at this exact (field, term). Advancing a
        // cursor and re-pushing its next (strictly greater) term keeps the heap in sync.
        let mut contributing: Vec<usize> = vec![first_si];
        cursors[first_si].advance();
        if let Some((f, t)) = cursors[first_si].peek() {
            heap.push(Reverse((f.to_string(), t.to_string(), first_si)));
        }
        loop {
            let same = matches!(heap.peek(), Some(Reverse((f, t, _))) if *f == field && *t == term);
            if !same {
                break;
            }
            let Reverse((_, _, si)) = heap.pop().unwrap();
            contributing.push(si);
            cursors[si].advance();
            if let Some((f, t)) = cursors[si].peek() {
                heap.push(Reverse((f.to_string(), t.to_string(), si)));
            }
        }

        // Contributing segments in ASCENDING index (+ locals ascending within each) => new
        // doc-ids come out ascending, matching merge_segments' BTreeMap<new_id, _> ordering
        // (dense renumbering assigns lower ids to earlier segments).
        contributing.sort_unstable();
        let mf = field_index[&field];
        let mut docs: Vec<(usize, Vec<u32>)> = Vec::new();
        for &si in &contributing {
            // positions_all ALREADY filters deletes. It may return an unordered HashMap, so sort
            // locals ascending before remapping so the new ids come out ascending.
            let mut locals: Vec<(usize, Vec<u32>)> =
                segs[si].positions_all(&field, &term).into_iter().collect();
            locals.sort_by_key(|(local, _)| *local);
            for (local, positions) in locals {
                if let Some(new_id) = doc_maps[si][local] {
                    docs.push((new_id, positions));
                }
            }
        }
        // terms with no live docs are dropped (== ZSL / merge_segments).
        if docs.is_empty() {
            continue;
        }
        dict_writer.add_term(&TermPostings {
            field_num: mf,
            text: term,
            docs,
        })?;
    }
    dict_writer.finish()?; // back-patch term counts + close the temp .frq/.prx

    // ---- phase 4: assemble the .cfs (small blocks from RAM, big ones streamed from temp files) ----
    let fnm_bytes = fnm::write_fnm(&fields);
    let nrm = norms::write_norms_raw(&norm_cols);
    let tis_bytes = tis_buf.into_inner();
    let tii_bytes = tii_buf.into_inner();

    // Same file order as `assemble_cfs`: .fdx .fdt .fnm .nrm .tis .tii .frq .prx.
    let files: Vec<(&str, CfsSource)> = vec![
        (".fdx", CfsSource::Mem(&fdx_buf)),
        (".fdt", CfsSource::Path(fdt_tmp)),
        (".fnm", CfsSource::Mem(&fnm_bytes)),
        (".nrm", CfsSource::Mem(&nrm)),
        (".tis", CfsSource::Mem(&tis_bytes)),
        (".tii", CfsSource::Mem(&tii_bytes)),
        (".frq", CfsSource::Path(frq_tmp)),
        (".prx", CfsSource::Path(prx_tmp)),
    ];

    // Write the merged .cfs durably: stream it into the file, then fsync BEFORE returning so the
    // generation flip in optimize() only references a durable .cfs.
    let cfs_path = index_dir.join(format!("{merged_name}.cfs"));
    let mut cfs_file = File::create(&cfs_path)?;
    write_cfs_streaming(&mut cfs_file, merged_name, &files)?;
    cfs_file.sync_all()?;

    Ok(doc_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::IndexReader;
    use crate::zsl::index::ZslIndex;
    use crate::zsl::segment::ZslSegment;
    use crate::zsl::segments::read_segment_infos;
    use crate::zsl::writer::{IndexWriter, WriterDoc, WriterField, WriterOpts};
    use std::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    fn temp_kb_full() -> std::path::PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("sdsearch_merge_{}_{}", std::process::id(), n));
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

    fn doc_mark(i: usize) -> WriterDoc {
        WriterDoc {
            fields: vec![WriterField::text("title", &format!("zqxmark unique{i}"))],
        }
    }

    /// writes the merged bytes as the ONLY segment in a new dir and opens it with the reader.
    fn read_merged(cfs_bytes: &[u8], doc_count: usize) -> ZslSegment {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let out =
            std::env::temp_dir().join(format!("sdsearch_merged_{}_{}", std::process::id(), n));
        std::fs::create_dir_all(&out).unwrap();
        std::fs::write(out.join("_m.cfs"), cfs_bytes).unwrap();
        let seg = ZslSegment::open_named(&out, "_m", -1).unwrap();
        assert_eq!(seg.max_doc(), doc_count); // dense renumbering: maxDoc == live docs
        std::fs::remove_dir_all(&out).ok();
        seg
    }

    #[test]
    fn merges_multi_segment_with_deletes_compacting_and_renumbering() {
        let dir = temp_kb_full();

        // multi-segment base: KB _2 (20 docs) + _3,_4 (4 docs with unique term 'zqxmark')
        let opts = WriterOpts {
            max_buffered_docs: 2,
            ..WriterOpts::default()
        };
        let mut w = IndexWriter::open(&dir, opts).unwrap();
        for i in 0..4 {
            w.add_document(doc_mark(i)).unwrap();
        }
        w.commit().unwrap();

        // delete global doc 0 (in _2) and 20 (first doc of _3) in one commit
        let mut w2 = IndexWriter::open(&dir, WriterOpts::default()).unwrap();
        w2.delete_document(0);
        w2.delete_document(20);
        w2.commit().unwrap();

        let before = ZslIndex::open(&dir).unwrap();
        let live = before.num_docs(); // 20 + 4 - 2 = 22
        assert_eq!(live, 22);
        // NOTE on reader semantics: `doc_freq` is DELETE-AGNOSTIC (comes from `.tis`, like
        // Lucene) → still 4 even though a doc with 'zqxmark' is deleted. `postings_for` DOES
        // filter deletes → 3 live docs with the term (gid 20 deleted). The merge rebuilds
        // `.tis` from live docs only, so the MERGED docFreq must be 3.
        assert_eq!(before.doc_freq("title", "zqxmark"), 4); // .tis, incl. the deleted one
        let mark_live = before.postings_for("title", "zqxmark").len(); // live = 3
        assert_eq!(mark_live, 3);
        drop(before);

        // MERGE
        let infos = read_segment_infos(&dir).unwrap();
        let refs: Vec<(String, i64)> = infos.iter().map(|s| (s.name.clone(), s.del_gen)).collect();
        let result = merge_segments(&dir, "_m", &refs).unwrap();
        assert_eq!(result.doc_count, live);

        // the native reader re-reads the merged segment: live docs, compacted term, no deletions
        let seg = read_merged(&result.cfs_bytes, result.doc_count);
        assert_eq!(seg.num_docs(), live);
        // the merge excludes the deleted doc → merged `.tis` docFreq == live docs (3)
        assert_eq!(seg.doc_freq("title", "zqxmark"), mark_live);
        // densely renumbered postings: all doc-ids < doc_count, ascending
        let post = seg.postings_for("title", "zqxmark");
        assert_eq!(post.len(), mark_live);
        assert!(post.iter().all(|(d, _)| *d < result.doc_count));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    #[should_panic(expected = "binary stored field not supported")]
    fn merge_panics_on_binary_stored_field_detected_via_is_binary_flag() {
        // hand-crafts a minimal segment (ONE doc, ONE NON-indexed stored field) whose `.fdt`
        // has the binary flag (0x02) set, to exercise the `is_binary == true` branch of
        // `read_stored_raw` through the `merge_segments` guard. The normal writer NEVER
        // produces this (`writer::invert::StoredField` has no `is_binary`; the host application
        // does not index binaries) — this test covers flag detection as defense in depth.
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("sdsearch_merge_bin_{}_{}", std::process::id(), n));
        std::fs::create_dir_all(&dir).unwrap();

        let fields = vec![FieldMeta {
            name: "bin_field".to_string(),
            indexed: false,
        }];
        let fnm_bytes = fnm::write_fnm(&fields);

        // .fdt of a doc with one field: VInt(stored_count=1) + VInt(field_num=0) + flags(0x02)
        // + VInt(len) + raw bytes. Same layout `read_stored_raw` expects for a binary.
        let payload = b"binpayload";
        let mut fdt = vec![0x01u8, 0x00u8, 0x02u8];
        crate::zsl::bytes::write_vint(&mut fdt, payload.len() as u64);
        fdt.extend_from_slice(payload);
        // .fdx: a single doc, offset 0 into `.fdt` (u64 big-endian).
        let fdx = vec![0u8; 8];

        let nrm = norms::write_norms_raw(&[Vec::new()]); // non-indexed field: empty column
        let dict = terms::write_term_dict(&[]); // no indexed terms

        let cfs_bytes = assemble_cfs("_x", &fnm_bytes, &fdt, &fdx, &nrm, &dict);
        std::fs::write(dir.join("_x.cfs"), &cfs_bytes).unwrap();

        // the panic happens INSIDE merge_segments (guard in the stored `.map()`); if it ever
        // stops panicking, this `unwrap()` surfaces the failure instead of a silent Result.
        merge_segments(&dir, "_m", &[("_x".to_string(), -1)]).unwrap();
    }

    #[test]
    fn kb_segment_has_prx() {
        let dir = temp_kb_full();
        let infos = read_segment_infos(&dir).unwrap();
        for info in &infos {
            let seg = ZslSegment::open_named(&dir, &info.name, info.del_gen).unwrap();
            assert!(seg.has_prx(), "the KB fixture should have .prx");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    #[should_panic(expected = "indexed fields but no .prx")]
    fn merge_panics_on_indexed_field_without_prx_file() {
        // hand-crafts a minimal segment (ONE indexed field, ONE doc with no stored fields) whose
        // `.cfs` has NO `.prx` at all — unlike `assemble_cfs`, which ALWAYS adds the 8 canonical
        // extensions (`.prx` included, even if empty), which would keep `has_prx()` returning
        // `true` (it looks at the compound file's sub-file LIST, not their content). The directory
        // is built by hand via `write_cfs` with a sub-file list that omits `.prx` deliberately, to
        // exercise the guard branch in `merge_segments` that `kb_segment_has_prx` does not cover
        // (that test only exercises the happy path: KB DOES have `.prx`, the assert never fires).
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("sdsearch_merge_noprx_{}_{}", std::process::id(), n));
        std::fs::create_dir_all(&dir).unwrap();

        let fields = vec![FieldMeta {
            name: "body".to_string(),
            indexed: true,
        }];
        let fnm_bytes = fnm::write_fnm(&fields);

        // a doc with no stored fields: enough for `open_named` to compute max_doc == 1.
        let (fdt, fdx) = stored::write_stored(&[Vec::new()]);

        // empty dict: the guard runs BEFORE iterating terms, so no real posting is needed —
        // only that `.tis`/`.frq` exist (they are mandatory in `open_from`).
        let dict = terms::write_term_dict(&[]);

        // directory built by hand (NOT `assemble_cfs`): only the extensions REQUIRED by
        // `ZslSegment::open_from` (.fdx .fdt .fnm .tis .frq), without `.nrm`/`.tii` (optional) and
        // without `.prx` (deliberately omitted).
        let files: Vec<(&str, &[u8])> = vec![
            (".fdx", &fdx),
            (".fdt", &fdt),
            (".fnm", &fnm_bytes),
            (".tis", &dict.tis),
            (".frq", &dict.frq),
        ];
        let cfs_bytes = crate::zsl::writer::cfs::write_cfs("_x", &files);
        std::fs::write(dir.join("_x.cfs"), &cfs_bytes).unwrap();

        // sanity: the segment opens clean (indexed field, no .prx) before going through merge.
        let seg = ZslSegment::open_named(&dir, "_x", -1).unwrap();
        assert!(seg.field_infos().iter().any(|fi| fi.is_indexed));
        assert!(!seg.has_prx());
        drop(seg);

        merge_segments(&dir, "_m", &[("_x".to_string(), -1)]).unwrap();
    }

    // ---- differential gate: streaming merge must be byte-identical to the batch oracle ----

    /// Runs BOTH the batch oracle (`merge_segments`, bytes captured in RAM) and
    /// `merge_segments_streaming` (writes `_m.cfs`) over the SAME `merged_name` and asserts the
    /// resulting `.cfs` bytes are identical. The name must match: the CFS sub-file directory
    /// embeds `{name}{ext}`, so different names would differ trivially. Also asserts the temp
    /// files were cleaned up.
    fn assert_streaming_byte_identical(dir: &std::path::Path, refs: &[(String, i64)]) {
        let oracle = merge_segments(dir, "_m", refs).unwrap(); // RAM only; does NOT write
        let doc_count = merge_segments_streaming(dir, "_m", refs).unwrap(); // writes _m.cfs (fsync)
        assert_eq!(doc_count, oracle.doc_count, "doc_count mismatch");
        let streamed = std::fs::read(dir.join("_m.cfs")).unwrap();
        assert_eq!(
            streamed, oracle.cfs_bytes,
            "merged .cfs bytes differ (streaming vs batch)"
        );
        // temp files removed on the success path
        assert!(!dir.join("_m.fdt.tmp").exists());
        assert!(!dir.join("_m.frq.tmp").exists());
        assert!(!dir.join("_m.prx.tmp").exists());
    }

    #[test]
    fn streaming_matches_batch_multi_segment_no_deletes() {
        let dir = temp_kb_full();
        let opts = WriterOpts {
            max_buffered_docs: 2,
            ..WriterOpts::default()
        };
        let mut w = IndexWriter::open(&dir, opts).unwrap();
        for i in 0..4 {
            w.add_document(doc_mark(i)).unwrap();
        }
        w.commit().unwrap();

        let infos = read_segment_infos(&dir).unwrap();
        assert!(infos.len() >= 2, "expected multi-segment base");
        assert!(
            infos.iter().all(|s| s.del_gen == -1),
            "scenario (a) must have NO deletes"
        );
        let refs: Vec<(String, i64)> = infos.iter().map(|s| (s.name.clone(), s.del_gen)).collect();
        assert_streaming_byte_identical(&dir, &refs);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn streaming_matches_batch_deletes_across_two_base_segments() {
        let dir = temp_kb_full();
        let opts = WriterOpts {
            max_buffered_docs: 2,
            ..WriterOpts::default()
        };
        let mut w = IndexWriter::open(&dir, opts).unwrap();
        for i in 0..4 {
            w.add_document(doc_mark(i)).unwrap();
        }
        w.commit().unwrap();

        // delete a doc in _2 (gid 0) and the first doc of _3 (gid 20), in one commit.
        let mut w2 = IndexWriter::open(&dir, WriterOpts::default()).unwrap();
        w2.delete_document(0);
        w2.delete_document(20);
        w2.commit().unwrap();

        let infos = read_segment_infos(&dir).unwrap();
        assert!(infos.iter().any(|s| s.del_gen != -1), "scenario (b) needs deletes");
        let refs: Vec<(String, i64)> = infos.iter().map(|s| (s.name.clone(), s.del_gen)).collect();
        assert_streaming_byte_identical(&dir, &refs);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn streaming_matches_batch_single_segment_with_deletes() {
        let dir = temp_kb_full();
        let mut w = IndexWriter::open(&dir, WriterOpts::default()).unwrap();
        w.delete_document(5);
        w.commit().unwrap();

        let infos = read_segment_infos(&dir).unwrap();
        assert_eq!(infos.len(), 1, "scenario (c) is single-segment");
        assert_ne!(infos[0].del_gen, -1, "must have a .del");
        let refs: Vec<(String, i64)> = infos.iter().map(|s| (s.name.clone(), s.del_gen)).collect();
        assert_streaming_byte_identical(&dir, &refs);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn streaming_matches_batch_all_docs_deleted_zero_live_docs() {
        // Empty-merge scenario: delete EVERY doc in the single-segment KB fixture (all 20,
        // global ids 0..20 — the KB fixture is exactly one segment, `_2`, with `max_doc() ==
        // 20`, see `sdsearch-core/src/zsl/segment.rs`), so the merge input has ZERO live docs.
        // This exercises phase 2/3 with empty `stored`/`term_map` (empty `.fdt`/`.fdx`, no
        // terms survive the "drop terms with no live docs" filter) — the smallest possible
        // merge input, not covered by the other differential scenarios (which all retain >=1
        // live doc).
        let dir = temp_kb_full();
        let mut w = IndexWriter::open(&dir, WriterOpts::default()).unwrap();
        for gid in 0..20 {
            w.delete_document(gid);
        }
        w.commit().unwrap();

        let infos = read_segment_infos(&dir).unwrap();
        assert_eq!(infos.len(), 1, "scenario is single-segment, fully deleted");
        assert_eq!(
            ZslIndex::open(&dir).unwrap().num_docs(),
            0,
            "all docs must be deleted"
        );

        let refs: Vec<(String, i64)> = infos.iter().map(|s| (s.name.clone(), s.del_gen)).collect();
        let oracle = merge_segments(&dir, "_m", &refs).unwrap();
        assert_eq!(
            oracle.doc_count, 0,
            "merge of a fully-deleted segment has 0 live docs"
        );

        assert_streaming_byte_identical(&dir, &refs);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn streaming_errors_on_binary_stored_field() {
        // same hand-crafted binary-flagged segment as the batch guard test, but exercised via the
        // streaming path (which returns an Err rather than panicking through an unwrap).
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("sdsearch_smerge_bin_{}_{}", std::process::id(), n));
        std::fs::create_dir_all(&dir).unwrap();

        let fields = vec![FieldMeta {
            name: "bin_field".to_string(),
            indexed: false,
        }];
        let fnm_bytes = fnm::write_fnm(&fields);
        let payload = b"binpayload";
        let mut fdt = vec![0x01u8, 0x00u8, 0x02u8];
        crate::zsl::bytes::write_vint(&mut fdt, payload.len() as u64);
        fdt.extend_from_slice(payload);
        let fdx = vec![0u8; 8];
        let nrm = norms::write_norms_raw(&[Vec::new()]);
        let dict = terms::write_term_dict(&[]);
        let cfs_bytes = assemble_cfs("_x", &fnm_bytes, &fdt, &fdx, &nrm, &dict);
        std::fs::write(dir.join("_x.cfs"), &cfs_bytes).unwrap();

        let err = merge_segments_streaming(&dir, "_m", &[("_x".to_string(), -1)]).unwrap_err();
        assert!(
            err.to_string().contains("binary stored field not supported"),
            "unexpected error: {err}"
        );
        // partial temp files (if any) are cleaned up even on the error path
        assert!(!dir.join("_m.fdt.tmp").exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn streaming_errors_on_indexed_field_without_prx() {
        // same hand-crafted no-.prx segment as the batch guard test, exercised via streaming.
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir()
            .join(format!("sdsearch_smerge_noprx_{}_{}", std::process::id(), n));
        std::fs::create_dir_all(&dir).unwrap();

        let fields = vec![FieldMeta {
            name: "body".to_string(),
            indexed: true,
        }];
        let fnm_bytes = fnm::write_fnm(&fields);
        let (fdt, fdx) = stored::write_stored(&[Vec::new()]);
        let dict = terms::write_term_dict(&[]);
        let files: Vec<(&str, &[u8])> = vec![
            (".fdx", &fdx),
            (".fdt", &fdt),
            (".fnm", &fnm_bytes),
            (".tis", &dict.tis),
            (".frq", &dict.frq),
        ];
        let cfs_bytes = crate::zsl::writer::cfs::write_cfs("_x", &files);
        std::fs::write(dir.join("_x.cfs"), &cfs_bytes).unwrap();

        let err = merge_segments_streaming(&dir, "_m", &[("_x".to_string(), -1)]).unwrap_err();
        assert!(
            err.to_string().contains("indexed fields but no .prx"),
            "unexpected error: {err}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn merge_segments_collapses_multiseg_with_delete() {
        let dir = temp_kb_full();

        // cap=1 → each add flushes its own segment: KB (_2) + 2 new ones (_3, _4).
        let opts = WriterOpts {
            max_buffered_docs: 1,
            ..WriterOpts::default()
        };
        let mut w = IndexWriter::open(&dir, opts).unwrap();
        w.add_document(WriterDoc {
            fields: vec![WriterField::text("title", "zqmerge alpha")],
        })
        .unwrap();
        w.add_document(WriterDoc {
            fields: vec![WriterField::text("title", "zqmerge beta")],
        })
        .unwrap();
        w.commit().unwrap();

        // + a delete on the base (gid 0, lives in _2): the merge must EXCLUDE that doc, not
        // just concatenate segments.
        let mut w2 = IndexWriter::open(&dir, WriterOpts::default()).unwrap();
        w2.delete_document(0);
        w2.commit().unwrap();

        let infos = read_segment_infos(&dir).unwrap();
        assert!(
            infos.len() >= 3,
            "expected multi-segment (KB + 2 flushed), got {}",
            infos.len()
        );
        let live = ZslIndex::open(&dir).unwrap().num_docs();

        // DIRECT merge of the primitive (not via `IndexWriter::optimize()`) over all the
        // current segments.
        let refs: Vec<(String, i64)> = infos.iter().map(|s| (s.name.clone(), s.del_gen)).collect();
        let gen = crate::zsl::writer::segments::read_generation(&dir).unwrap();
        let merged_name = crate::zsl::writer::segments::segment_name(gen.name_counter);
        let merged = merge_segments(&dir, &merged_name, &refs).unwrap();
        assert_eq!(
            merged.doc_count, live,
            "merge doc_count must equal the total live docs"
        );

        // write the merged `.cfs` and flip the generation by hand (same steps 4-5 as
        // `optimize()`, but exercising `merge_segments` directly) to re-read with the full
        // index reader, not the ad-hoc `read_merged` helper of the other tests.
        std::fs::write(dir.join(format!("{merged_name}.cfs")), &merged.cfs_bytes).unwrap();
        crate::zsl::writer::segments::write_optimized_generation(
            &dir,
            &gen,
            &merged_name,
            merged.doc_count as u32,
        )
        .unwrap();

        // the merged `.cfs` is assemblable: the full index opens and is queryable.
        // (byte-for-byte fidelity of the merge content is covered by the differential harness;
        //  this test exercises the direct primitive: doc_count + a witness term.)
        assert_eq!(read_segment_infos(&dir).unwrap().len(), 1);
        let idx = ZslIndex::open(&dir).unwrap();
        assert_eq!(idx.num_docs(), live);
        // witness term: both added docs carry 'zqmerge', none deleted => docFreq 2.
        assert_eq!(idx.doc_freq("title", "zqmerge"), 2);
        assert_eq!(idx.postings_for("title", "zqmerge").len(), 2);

        std::fs::remove_dir_all(&dir).ok();
    }
}
