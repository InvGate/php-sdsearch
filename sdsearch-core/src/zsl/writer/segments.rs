//! Writer for ZSL's generation protocol: reads the current generation and writes the
//! next one (segments_{N+1} + double-written segments.gen). Mirrors `Writer::_updateSegments`
//! and Zend Lucene's segment-naming scheme, for an add-only APPEND of ONE
//! (`write_appended_generation`) or N (`write_generation_with_new_segments`, streaming commit)
//! new segments.
//!
//! The records of the existing segments are copied VERBATIM from the old segments_N (same
//! target format), so only the new records are appended and the header is patched
//! (version+1, nameCounter += N, segCount += N). Each new segment is named `_<base36(k)>`
//! with `k` = old nameCounter, k+1, …; the new file writes `nameCounter+N` so that ZSL,
//! when it resumes, does not reuse those names (safe handoff).

use crate::zsl::bytes::{
    read_modified_utf8, read_u32_be, read_u64_be, write_i32_be, write_i64_be, write_modified_utf8,
    write_u32_be,
};
use crate::zsl::segments::to_base36;
use std::path::Path;

const FORMAT_2_1: u32 = 0xFFFF_FFFD;
const FORMAT_2_3: u32 = 0xFFFF_FFFC;
const GEN_FORMAT: u32 = 0xFFFF_FFFE;
const HEADER_LEN: usize = 20; // format(4) + version(8) + nameCounter(4) + segCount(4)

fn io_err(msg: &str) -> std::io::Error {
    std::io::Error::other(msg.to_string())
}

/// snapshot of the current generation, with the segments_N bytes and the record range.
pub struct Generation {
    pub generation: u64,
    pub name_counter: u32,
    pub format: u32,
    pub version: u64,
    pub seg_count: u32,
    /// raw bytes of the current segments_N.
    pub data: Vec<u8>,
    /// [start, end) of the segment records region within `data`.
    pub records_range: (usize, usize),
    /// per existing segment: (name, absolute offset of the delGen `long` within `data`).
    pub record_delgen_offsets: Vec<(String, usize)>,
}

/// a new segment to list in the generation (name + doc count).
#[derive(Debug, Clone, PartialEq)]
pub struct NewSegment {
    pub name: String,
    pub doc_count: u32,
}

/// segment name for a counter (== Zend Lucene's segment-naming scheme).
pub fn segment_name(counter: u32) -> String {
    format!("_{}", to_base36(counter as u64))
}

/// inverse of `to_base36`: parses a lowercase base-36 string to a number. `None` if any
/// byte isn't a valid base-36 digit or the string is empty — defensive, so a file we don't
/// recognize is never touched by the generation-manifest pruning below.
fn from_base36(s: &str) -> Option<u64> {
    if s.is_empty() {
        return None;
    }
    let mut n: u64 = 0;
    for b in s.bytes() {
        let digit = match b {
            b'0'..=b'9' => b - b'0',
            b'a'..=b'z' => b - b'a' + 10,
            _ => return None,
        };
        n = n.checked_mul(36)?.checked_add(digit as u64)?;
    }
    Some(n)
}

/// Best-effort deletion of stale generation manifests: any `segments_<base36>` whose parsed
/// number is strictly less than `keep_from_gen`. Only ever touches files matching that exact
/// pattern — never `segments.gen`, never `.cfs`/`.del`/`.sti`/`.tmp`/lock files, never the
/// legacy no-suffix `segments` (generation 0) file. Comparison is NUMERIC (parsed), not
/// lexical, so e.g. `segments_10` (36) is correctly treated as newer than `segments_z` (35).
/// Individual `read_dir`/`remove_file` errors are ignored (mirrors the orphan `.cfs` cleanup
/// in `IndexWriter::optimize`): this is housekeeping, never load-bearing for correctness.
///
/// MUST be called only AFTER `segments.gen` has been durably flipped to the new generation.
/// Callers pass `keep_from_gen = new_gen - 1` so both the current generation and the
/// immediately-previous one survive — a grace window for lock-free concurrent readers that
/// may have read the old `segments.gen` value an instant before the flip and are about to
/// open the generation manifest it pointed to.
fn prune_old_generations(index_dir: &Path, keep_from_gen: u64) {
    let entries = match std::fs::read_dir(index_dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let Some(name) = entry.file_name().to_str().map(str::to_string) else {
            continue;
        };
        let Some(suffix) = name.strip_prefix("segments_") else {
            continue;
        };
        let Some(n) = from_base36(suffix) else {
            continue;
        };
        if n < keep_from_gen {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// Name of the deferred-deletion sidecar. Not a `segments_*` manifest, `.cfs`, or `.del`, so it
/// is invisible to `prune_old_generations`, to readers, and to segment enumeration.
const PENDING_DELETIONS_FILE: &str = "pending_deletions";

/// Best-effort, one-generation-DEFERRED reclamation of segment data files (`_<name>.cfs/.sti/
/// .del`) superseded by a merge. `optimize()` cannot delete the old segments immediately: the
/// flip keeps the previous generation's manifest as a grace window for lock-free readers, and
/// that manifest still references the old files — deleting them now would leave such a reader
/// with dangling references (and on Windows the unlink of a file still mapped by a reader fails
/// silently, leaking it). So each flip instead does two things. First, it deletes the files
/// recorded by the PREVIOUS flip — whose referencing manifest this flip's `prune_old_generations`
/// has just removed, so they are now safe — retrying/carrying forward any that still fail (e.g. a
/// Windows reader still holds the mapping). Second, it records `new_superseded` (this flip's
/// just-superseded files, still referenced by the kept previous manifest) for the NEXT flip.
///
/// Because every flip advances the generation by exactly one, a file recorded here is always
/// referenced solely by manifests the next prune removes. The ONLY failure mode of a bug here is
/// a disk leak — it NEVER deletes a file that is still referenced (it only ever lists already-
/// superseded files). Must be called AFTER the durable flip + prune. Best-effort: all errors
/// (missing sidecar, unreadable, unwritable) are ignored.
///
/// `new_superseded` are file NAMES relative to `index_dir` (e.g. `"_2.cfs"`), not paths.
pub(crate) fn process_pending_deletions(index_dir: &Path, new_superseded: &[String]) {
    let sidecar = index_dir.join(PENDING_DELETIONS_FILE);

    // 1) reclaim the previous round's files; carry forward any that could not be deleted yet.
    let mut carry: Vec<String> = Vec::new();
    if let Ok(contents) = std::fs::read_to_string(&sidecar) {
        for name in contents.lines() {
            let name = name.trim();
            if name.is_empty() {
                continue;
            }
            let path = index_dir.join(name);
            match std::fs::remove_file(&path) {
                Ok(()) => {}
                // already gone (deleted earlier / never existed): treat as reclaimed.
                Err(_) if !path.exists() => {}
                // still present but unlink failed (e.g. Windows: mapped by a reader) → retry next flip.
                Err(_) => carry.push(name.to_string()),
            }
        }
    }

    // 2) next round's list = carried-forward failures + this flip's newly-superseded files.
    let mut next = carry;
    for f in new_superseded {
        if !next.contains(f) {
            next.push(f.clone());
        }
    }

    if next.is_empty() {
        let _ = std::fs::remove_file(&sidecar);
    } else {
        let _ = std::fs::write(&sidecar, next.join("\n"));
    }
}

fn read_gen_number(index_dir: &Path) -> std::io::Result<u64> {
    let data = std::fs::read(index_dir.join("segments.gen"))?;
    let mut pos = 0usize;
    if read_u32_be(&data, &mut pos)? != GEN_FORMAT {
        return Err(io_err("wrong segments.gen format"));
    }
    let g1 = read_u64_be(&data, &mut pos)?;
    let g2 = read_u64_be(&data, &mut pos)?;
    if g1 != g2 {
        return Err(io_err("segments.gen mid-write (g1 != g2)"));
    }
    Ok(g1)
}

/// walks `seg_count` records from `pos`; returns (final offset, [(name, delGen offset)]).
fn walk_records(
    data: &[u8],
    mut pos: usize,
    format: u32,
    seg_count: u32,
) -> std::io::Result<(usize, Vec<(String, usize)>)> {
    let is_2_3 = format == FORMAT_2_3;
    let mut offsets = Vec::with_capacity(seg_count as usize);
    for _ in 0..seg_count {
        let name = read_modified_utf8(data, &mut pos)?;
        pos += 4; // docCount
        offsets.push((name, pos)); // the delGen (long) starts here
        pos += 8; // delGen (long)
        if is_2_3 {
            let doc_store_offset = read_u32_be(data, &mut pos)?;
            if doc_store_offset != 0xFFFF_FFFF {
                let _seg = read_modified_utf8(data, &mut pos)?;
                pos += 1; // docStoreIsCompound byte
            }
        }
        pos += 1; // hasSingleNormFile byte
        let num_field = read_u32_be(data, &mut pos)?;
        if num_field != 0xFFFF_FFFF {
            return Err(io_err("separate norm files unsupported (optimize index)"));
        }
        pos += 1; // isCompound byte
    }
    Ok((pos, offsets))
}

/// reads the index's current generation.
pub fn read_generation(index_dir: &Path) -> std::io::Result<Generation> {
    let generation = read_gen_number(index_dir)?;
    let fname = if generation == 0 {
        "segments".to_string()
    } else {
        format!("segments_{}", to_base36(generation))
    };
    let data = std::fs::read(index_dir.join(&fname))?;

    let mut pos = 0usize;
    let format = read_u32_be(&data, &mut pos)?;
    if format != FORMAT_2_1 && format != FORMAT_2_3 {
        return Err(io_err("unsupported segments file format"));
    }
    let version = read_u64_be(&data, &mut pos)?;
    let name_counter = read_u32_be(&data, &mut pos)?;
    let seg_count = read_u32_be(&data, &mut pos)?;
    debug_assert_eq!(pos, HEADER_LEN);

    let (records_end, record_delgen_offsets) = walk_records(&data, pos, format, seg_count)?;

    Ok(Generation {
        generation,
        name_counter,
        format,
        version,
        seg_count,
        data,
        records_range: (HEADER_LEN, records_end),
        record_delgen_offsets,
    })
}

/// Writes the next generation adding ONE new segment (add-only). Wrapper of
/// `write_generation_with_new_segments` for the batch append.
pub fn write_appended_generation(
    index_dir: &Path,
    gen: &Generation,
    new_segment_name: &str,
    new_segment_doc_count: u32,
) -> std::io::Result<u64> {
    write_generation_with_new_segments(
        index_dir,
        gen,
        &[NewSegment {
            name: new_segment_name.to_string(),
            doc_count: new_segment_doc_count,
        }],
    )
}

/// Writes the next generation adding N new segments (add-only), WITHOUT changing the delGen
/// of the existing ones. Wrapper of `write_generation_with_delgens` with empty overrides.
pub fn write_generation_with_new_segments(
    index_dir: &Path,
    gen: &Generation,
    new_segments: &[NewSegment],
) -> std::io::Result<u64> {
    write_generation_with_delgens(
        index_dir,
        gen,
        &std::collections::HashMap::new(),
        new_segments,
        gen.name_counter + new_segments.len() as u32,
    )
}

/// Writes the next generation: copies the existing records verbatim BUT patches in-place the
/// delGen (long, 8 bytes BE) of the segments present in `del_gen_overrides`; adds a record
/// per new segment (delGen=-1). Writes segments_{N+1} first and only then flips
/// segments.gen via atomic rename (a crash leaves the index readable at the old generation).
pub fn write_generation_with_delgens(
    index_dir: &Path,
    gen: &Generation,
    del_gen_overrides: &std::collections::HashMap<String, i64>,
    new_segments: &[NewSegment],
    name_counter: u32,
) -> std::io::Result<u64> {
    let added = new_segments.len() as u32;
    let mut out = Vec::new();
    write_u32_be(&mut out, gen.format);
    write_i64_be(&mut out, (gen.version + 1) as i64);
    write_u32_be(&mut out, name_counter); // high-water mark, not gen.name_counter + added
    write_u32_be(&mut out, gen.seg_count + added);

    // existing records verbatim (copy), then in-place delGen patch for the overrides.
    let records = &gen.data[gen.records_range.0..gen.records_range.1];
    let mut records = records.to_vec();
    for (name, delgen_off) in &gen.record_delgen_offsets {
        if let Some(&g) = del_gen_overrides.get(name) {
            // offset relative to the start of the records region
            let rel = delgen_off - gen.records_range.0;
            records[rel..rel + 8].copy_from_slice(&g.to_be_bytes());
        }
    }
    out.extend_from_slice(&records);

    // one record per new segment (compound, no del, single norm file) — same format as an add-only append
    for seg in new_segments {
        write_modified_utf8(&mut out, &seg.name);
        write_u32_be(&mut out, seg.doc_count);
        write_i32_be(&mut out, -1); // delGen (long -1) = two ints 0xFFFFFFFF
        write_i32_be(&mut out, -1);
        if gen.format == FORMAT_2_3 {
            write_i32_be(&mut out, -1); // docStoreOffset = none
        }
        out.push(0x01); // hasSingleNormFile = true
        write_i32_be(&mut out, -1); // numField = 0xFFFFFFFF (single norm file)
        out.push(0x01); // isCompound = true
    }

    let new_gen = gen.generation + 1;
    let seg_fname = format!("segments_{}", to_base36(new_gen));
    std::fs::write(index_dir.join(&seg_fname), &out)?;

    // only now do we flip segments.gen (double-written), via atomic rename.
    let mut g = Vec::new();
    write_u32_be(&mut g, GEN_FORMAT);
    write_i64_be(&mut g, new_gen as i64);
    write_i64_be(&mut g, new_gen as i64);
    super::durability::write_atomic(&index_dir.join("segments.gen"), &g)?;

    // best-effort prune of stale generation manifests, only now that the flip is durable.
    prune_old_generations(index_dir, new_gen.saturating_sub(1));

    Ok(new_gen)
}

/// Writes the next generation REPLACING all segments with ONE (merge/optimize).
/// Unlike `write_generation_with_new_segments` (add-only, copies old records),
/// `segments_{N+1}` lists ONLY the merged segment (delGen -1), with
/// `name_counter+1` and `seg_count=1`. Durability: `write_durable` (fsync) of
/// `segments_{N+1}` BEFORE the atomic flip of `segments.gen`. Returns the new generation.
pub fn write_optimized_generation(
    index_dir: &Path,
    gen: &Generation,
    merged_name: &str,
    merged_doc_count: u32,
) -> std::io::Result<u64> {
    let mut out = Vec::new();
    write_u32_be(&mut out, gen.format);
    write_i64_be(&mut out, (gen.version + 1) as i64);
    write_u32_be(&mut out, gen.name_counter + 1); // the merged segment consumed a name
    write_u32_be(&mut out, 1); // seg_count = 1

    // a single record: the merged segment (compound, no del, single norm file).
    write_modified_utf8(&mut out, merged_name);
    write_u32_be(&mut out, merged_doc_count);
    write_i32_be(&mut out, -1); // delGen (long -1) = two ints 0xFFFFFFFF
    write_i32_be(&mut out, -1);
    if gen.format == FORMAT_2_3 {
        write_i32_be(&mut out, -1); // docStoreOffset = none
    }
    out.push(0x01); // hasSingleNormFile = true
    write_i32_be(&mut out, -1); // numField = 0xFFFFFFFF (single norm file)
    out.push(0x01); // isCompound = true

    let new_gen = gen.generation + 1;
    let seg_fname = format!("segments_{}", to_base36(new_gen));
    // fsync of segments_{N+1} before the flip.
    super::durability::write_durable(&index_dir.join(&seg_fname), &out)?;

    // flip of segments.gen (double-written) via atomic rename.
    let mut g = Vec::new();
    write_u32_be(&mut g, GEN_FORMAT);
    write_i64_be(&mut g, new_gen as i64);
    write_i64_be(&mut g, new_gen as i64);
    super::durability::write_atomic(&index_dir.join("segments.gen"), &g)?;
    // Make the segments.gen rename durable at the directory level (best-effort; real fsync on
    // Unix, no-op on Windows). write_durable already fsync'd the segments_{N+1} dirent; the
    // segments.gen flip goes through write_atomic (no fsync), so sync the directory here so the
    // optimize() durability promise also covers the generation pointer's rename.
    super::durability::sync_dir(index_dir);

    // best-effort prune of stale generation manifests, only now that the flip is durable.
    prune_old_generations(index_dir, new_gen.saturating_sub(1));

    Ok(new_gen)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zsl::segments::read_segment_infos;
    use std::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    /// copies the KB fixture's generation files to a fresh temp dir.
    fn temp_kb_gen() -> std::path::PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("sdsearch_gen_{}_{}", std::process::id(), n));
        std::fs::create_dir_all(&dir).unwrap();
        let src = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/zsl_index_kb"
        ));
        for f in ["segments.gen", "segments_6"] {
            std::fs::copy(src.join(f), dir.join(f)).unwrap();
        }
        dir
    }

    fn temp_empty_dir(tag: &str) -> std::path::PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("sdsearch_pd_{tag}_{}_{}", std::process::id(), n));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn process_pending_deletions_deletes_listed_and_records_new() {
        let dir = temp_empty_dir("rec");
        // previous round's file, listed in the sidecar, is now safe to delete.
        std::fs::write(dir.join("_old.cfs"), b"x").unwrap();
        std::fs::write(dir.join(PENDING_DELETIONS_FILE), "_old.cfs").unwrap();
        // this round's just-superseded file, still referenced by the kept manifest.
        std::fs::write(dir.join("_new.cfs"), b"y").unwrap();

        process_pending_deletions(&dir, &["_new.cfs".to_string()]);

        // the previously-listed file is reclaimed...
        assert!(!dir.join("_old.cfs").exists());
        // ...the new superseded file is recorded for the NEXT flip, NOT deleted now...
        assert!(dir.join("_new.cfs").exists());
        let sidecar = std::fs::read_to_string(dir.join(PENDING_DELETIONS_FILE)).unwrap();
        assert_eq!(sidecar.trim(), "_new.cfs");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn process_pending_deletions_removes_sidecar_when_nothing_pending() {
        let dir = temp_empty_dir("empty");
        std::fs::write(dir.join("_old.cfs"), b"x").unwrap();
        std::fs::write(dir.join(PENDING_DELETIONS_FILE), "_old.cfs").unwrap();

        process_pending_deletions(&dir, &[]); // no new superseded files

        assert!(!dir.join("_old.cfs").exists());
        // nothing left to defer → the sidecar file itself is removed (no empty-file litter).
        assert!(!dir.join(PENDING_DELETIONS_FILE).exists());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn appends_new_segment_over_real_kb_generation() {
        let dir = temp_kb_gen();

        let gen = read_generation(&dir).unwrap();
        assert_eq!(gen.generation, 6);
        assert_eq!(gen.name_counter, 3);
        assert_eq!(gen.format, FORMAT_2_1);
        assert_eq!(gen.seg_count, 1);
        assert_eq!(gen.records_range, (20, 41));

        let name = segment_name(gen.name_counter);
        assert_eq!(name, "_3");

        let new_gen = write_appended_generation(&dir, &gen, &name, 5).unwrap();
        assert_eq!(new_gen, 7);

        // the trusted reader sees both segments, old intact + new
        let infos = read_segment_infos(&dir).unwrap();
        assert_eq!(infos.len(), 2);
        assert_eq!(infos[0].name, "_2");
        assert_eq!(infos[0].doc_count, 20);
        assert_eq!(infos[0].del_gen, -1);
        assert_eq!(infos[1].name, "_3");
        assert_eq!(infos[1].doc_count, 5);
        assert_eq!(infos[1].del_gen, -1);

        // nameCounter bumped to 4 in the new segments_7
        let sb = std::fs::read(dir.join("segments_7")).unwrap();
        let mut p = 12;
        assert_eq!(read_u32_be(&sb, &mut p).unwrap(), 4);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn appends_two_new_segments_in_one_generation() {
        let dir = temp_kb_gen();
        let gen = read_generation(&dir).unwrap();
        assert_eq!(gen.name_counter, 3);
        assert_eq!(gen.seg_count, 1);

        let new_gen = write_generation_with_new_segments(
            &dir,
            &gen,
            &[
                NewSegment {
                    name: "_3".into(),
                    doc_count: 3,
                },
                NewSegment {
                    name: "_4".into(),
                    doc_count: 4,
                },
            ],
        )
        .unwrap();
        assert_eq!(new_gen, 7);

        // the trusted reader sees the 3 segments: old intact + 2 new
        let infos = read_segment_infos(&dir).unwrap();
        assert_eq!(infos.len(), 3);
        assert_eq!(infos[0].name, "_2");
        assert_eq!(infos[0].doc_count, 20);
        assert_eq!(infos[1].name, "_3");
        assert_eq!(infos[1].doc_count, 3);
        assert_eq!(infos[2].name, "_4");
        assert_eq!(infos[2].doc_count, 4);

        // nameCounter bumped to 5 (3 + 2) in the new segments_7
        let sb = std::fs::read(dir.join("segments_7")).unwrap();
        let mut p = 12; // format(4)+version(8) → nameCounter
        assert_eq!(read_u32_be(&sb, &mut p).unwrap(), 5);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn bumps_delgen_of_existing_segment_in_place() {
        use std::collections::HashMap;
        let dir = temp_kb_gen();
        let gen = read_generation(&dir).unwrap();
        // KB: one segment "_2" with delGen -1
        let mut overrides = HashMap::new();
        overrides.insert("_2".to_string(), 1i64);

        let new_gen =
            write_generation_with_delgens(&dir, &gen, &overrides, &[], gen.name_counter).unwrap();
        assert_eq!(new_gen, 7);

        // the trusted reader sees _2 with delGen bumped to 1, docCount intact
        let infos = read_segment_infos(&dir).unwrap();
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].name, "_2");
        assert_eq!(infos[0].doc_count, 20);
        assert_eq!(infos[0].del_gen, 1);

        // segments.gen flipped (no residual .tmp)
        assert!(!dir.join("segments.gen.tmp").exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn optimized_generation_lists_only_merged_segment() {
        let dir = temp_kb_gen(); // KB: segments_6 (gen 6), one segment _2 (20 docs)
        let gen = read_generation(&dir).unwrap();
        assert_eq!(gen.name_counter, 3);
        assert_eq!(gen.seg_count, 1);

        // simulate a merged `.cfs` present (content irrelevant to read_segment_infos)
        std::fs::write(dir.join("_3.cfs"), b"merged-cfs-placeholder").unwrap();

        let new_gen = write_optimized_generation(&dir, &gen, "_3", 62).unwrap();
        assert_eq!(new_gen, 7);

        // the new generation lists ONLY the merged segment, delGen -1, docCount 62
        let infos = read_segment_infos(&dir).unwrap();
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].name, "_3");
        assert_eq!(infos[0].doc_count, 62);
        assert_eq!(infos[0].del_gen, -1);

        // nameCounter bumped to 4 (3 + 1)
        let sb = std::fs::read(dir.join("segments_7")).unwrap();
        let mut p = 12; // format(4)+version(8) -> nameCounter
        assert_eq!(read_u32_be(&sb, &mut p).unwrap(), 4);
        assert!(!dir.join("segments.gen.tmp").exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn write_optimized_generation_handles_format_2_3() {
        let dir = temp_kb_gen();
        let mut gen = read_generation(&dir).unwrap();
        gen.format = FORMAT_2_3; // force the docStoreOffset branch
        let new_gen = write_optimized_generation(&dir, &gen, "_9", 42).unwrap();
        assert_eq!(new_gen, gen.generation + 1);
        // re-read: the new generation exists and lists 1 segment.
        let reread = read_generation(&dir).unwrap();
        assert_eq!(reread.generation, new_gen);
        assert_eq!(reread.format, FORMAT_2_3);
        assert_eq!(reread.seg_count, 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn write_generation_with_delgens_prunes_old_manifests_keeps_current_and_previous() {
        use std::collections::HashMap;
        let dir = temp_kb_gen(); // KB: segments_6 (gen 6) + segments.gen
        let mut gen = read_generation(&dir).unwrap();
        for _ in 0..5 {
            let new_gen =
                write_generation_with_delgens(&dir, &gen, &HashMap::new(), &[], gen.name_counter)
                    .unwrap();
            gen = read_generation(&dir).unwrap();
            assert_eq!(gen.generation, new_gen);
        }
        // 5 flips from gen 6 -> gen 11.
        assert_eq!(gen.generation, 11);

        // strictly-older-than-previous manifests are gone (base36: 6,7,8,9 == same digits).
        assert!(!dir.join("segments_6").exists());
        assert!(!dir.join("segments_7").exists());
        assert!(!dir.join("segments_8").exists());
        assert!(!dir.join("segments_9").exists());
        // current (11 == "b") and immediately-previous (10 == "a") survive.
        assert!(dir.join(format!("segments_{}", to_base36(10))).exists());
        assert!(dir.join(format!("segments_{}", to_base36(11))).exists());
        // never touch segments.gen.
        assert!(dir.join("segments.gen").exists());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn prune_uses_numeric_not_lexical_comparison_across_base36_rollover() {
        use std::collections::HashMap;
        let dir = temp_kb_gen(); // KB: gen 6
        let mut gen = read_generation(&dir).unwrap();
        // advance past the base36 rollover ("z" = 35 -> "10" = 36) with margin.
        while gen.generation < 40 {
            write_generation_with_delgens(&dir, &gen, &HashMap::new(), &[], gen.name_counter)
                .unwrap();
            gen = read_generation(&dir).unwrap();
        }
        assert_eq!(gen.generation, 40);

        // a lexical-comparison bug would keep "segments_z" forever (the string "z" sorts
        // after "10"/"13"/etc.); numeric comparison must have pruned it (35 < 40-1).
        assert!(!dir.join("segments_z").exists());
        assert!(!dir.join("segments_10").exists()); // 36, also strictly older than 39

        // current (40) and immediately-previous (39) survive.
        let prev_name = format!("segments_{}", to_base36(39));
        let cur_name = format!("segments_{}", to_base36(40));
        assert!(dir.join(&prev_name).exists());
        assert!(dir.join(&cur_name).exists());

        std::fs::remove_dir_all(&dir).ok();
    }
}
