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

fn read_gen_number(index_dir: &Path) -> std::io::Result<u64> {
    let data = std::fs::read(index_dir.join("segments.gen"))?;
    let mut pos = 0usize;
    if read_u32_be(&data, &mut pos) != GEN_FORMAT {
        return Err(io_err("wrong segments.gen format"));
    }
    let g1 = read_u64_be(&data, &mut pos);
    let g2 = read_u64_be(&data, &mut pos);
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
        let name = read_modified_utf8(data, &mut pos);
        pos += 4; // docCount
        offsets.push((name, pos)); // the delGen (long) starts here
        pos += 8; // delGen (long)
        if is_2_3 {
            let doc_store_offset = read_u32_be(data, &mut pos);
            if doc_store_offset != 0xFFFF_FFFF {
                let _seg = read_modified_utf8(data, &mut pos);
                pos += 1; // docStoreIsCompound byte
            }
        }
        pos += 1; // hasSingleNormFile byte
        let num_field = read_u32_be(data, &mut pos);
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
    let format = read_u32_be(&data, &mut pos);
    if format != FORMAT_2_1 && format != FORMAT_2_3 {
        return Err(io_err("unsupported segments file format"));
    }
    let version = read_u64_be(&data, &mut pos);
    let name_counter = read_u32_be(&data, &mut pos);
    let seg_count = read_u32_be(&data, &mut pos);
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
        &[NewSegment { name: new_segment_name.to_string(), doc_count: new_segment_doc_count }],
    )
}

/// Writes the next generation adding N new segments (add-only), WITHOUT changing the delGen
/// of the existing ones. Wrapper of `write_generation_with_delgens` with empty overrides.
pub fn write_generation_with_new_segments(
    index_dir: &Path,
    gen: &Generation,
    new_segments: &[NewSegment],
) -> std::io::Result<u64> {
    write_generation_with_delgens(index_dir, gen, &std::collections::HashMap::new(), new_segments)
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
) -> std::io::Result<u64> {
    let added = new_segments.len() as u32;
    let mut out = Vec::new();
    write_u32_be(&mut out, gen.format);
    write_i64_be(&mut out, (gen.version + 1) as i64);
    write_u32_be(&mut out, gen.name_counter + added);
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
        assert_eq!(read_u32_be(&sb, &mut p), 4);

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
                NewSegment { name: "_3".into(), doc_count: 3 },
                NewSegment { name: "_4".into(), doc_count: 4 },
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
        assert_eq!(read_u32_be(&sb, &mut p), 5);

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

        let new_gen = write_generation_with_delgens(&dir, &gen, &overrides, &[]).unwrap();
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
        assert_eq!(read_u32_be(&sb, &mut p), 4);
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
}
