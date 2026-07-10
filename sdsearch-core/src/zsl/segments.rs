//! Parser for ZSL's generation files (segments.gen + segments_N).

use crate::zsl::bytes::{
    checked_capacity, read_byte, read_modified_utf8, read_u32_be, read_u64_be,
};
use std::path::Path;

/// minimal per-segment info taken from the generation file.
pub struct SegmentInfo {
    pub name: String,
    /// docCount = the segment's maxDoc (includes deletes); doc-id base across segments.
    pub doc_count: usize,
    /// .del generation; -1 = no deletes, 0 = "<name>.del", >0 = "<name>_<base36>.del".
    pub del_gen: i64,
}

/// base_convert(n, 10, 36) in lowercase, same as ZSL.
pub(crate) fn to_base36(n: u64) -> String {
    if n == 0 {
        return "0".to_string();
    }
    const DIGITS: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut n = n;
    let mut buf = Vec::new();
    while n > 0 {
        buf.push(DIGITS[(n % 36) as usize]);
        n /= 36;
    }
    buf.reverse();
    String::from_utf8(buf).unwrap()
}

fn io_err(msg: &str) -> std::io::Error {
    std::io::Error::other(msg.to_string())
}

/// current generation from segments.gen (format 0xFFFFFFFE, gen1 == gen2).
fn read_generation(index_dir: &Path) -> std::io::Result<u64> {
    let data = std::fs::read(index_dir.join("segments.gen"))?;
    let mut pos = 0usize;
    let format = read_u32_be(&data, &mut pos)?;
    if format != 0xFFFF_FFFE {
        return Err(io_err("wrong segments.gen format"));
    }
    let gen1 = read_u64_be(&data, &mut pos)?;
    let gen2 = read_u64_be(&data, &mut pos)?;
    if gen1 != gen2 {
        return Err(io_err("segments.gen mid-write (gen1 != gen2)"));
    }
    Ok(gen1)
}

/// reads segments_N and returns the segments in order.
pub fn read_segment_infos(index_dir: &Path) -> std::io::Result<Vec<SegmentInfo>> {
    let generation = read_generation(index_dir)?;
    let fname = if generation == 0 {
        "segments".to_string()
    } else {
        format!("segments_{}", to_base36(generation))
    };
    let data = std::fs::read(index_dir.join(&fname))?;
    let mut pos = 0usize;

    let format = read_u32_be(&data, &mut pos)?;
    // 0xFFFFFFFC = FORMAT_2_3, 0xFFFFFFFD = FORMAT_2_1
    let is_2_3 = match format {
        0xFFFF_FFFC => true,
        0xFFFF_FFFD => false,
        _ => return Err(io_err("unsupported segments file format")),
    };
    let _version = read_u64_be(&data, &mut pos)?; // version (long)
    let _name_counter = read_u32_be(&data, &mut pos)?; // segment name counter
    let seg_count = read_u32_be(&data, &mut pos)? as usize;

    let mut out = Vec::with_capacity(checked_capacity(seg_count, data.len()));
    for _ in 0..seg_count {
        let name = read_modified_utf8(&data, &mut pos)?;
        let doc_count = read_u32_be(&data, &mut pos)? as usize;
        let del_gen = read_u64_be(&data, &mut pos)? as i64;

        if is_2_3 {
            let doc_store_offset = read_u32_be(&data, &mut pos)?;
            if doc_store_offset != 0xFFFF_FFFF {
                let _doc_store_segment = read_modified_utf8(&data, &mut pos)?;
                let _doc_store_is_compound = read_byte(&data, &mut pos)?;
            }
        }

        let _has_single_norm_file = read_byte(&data, &mut pos)?;
        let num_field = read_u32_be(&data, &mut pos)?;
        if num_field != 0xFFFF_FFFF {
            // separate norm files => unoptimized index; ZSL doesn't support it either.
            return Err(io_err("separate norm files unsupported (optimize index)"));
        }
        let _is_compound_byte = read_byte(&data, &mut pos)?;

        out.push(SegmentInfo {
            name,
            doc_count,
            del_gen,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{read_segment_infos, to_base36};
    use std::path::PathBuf;

    fn kb_dir() -> PathBuf {
        PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/zsl_index_kb"
        ))
    }

    #[test]
    fn base36_matches_zsl() {
        // ZSL uses base_convert(n, 10, 36) in lowercase
        assert_eq!(to_base36(0), "0");
        assert_eq!(to_base36(10), "a");
        assert_eq!(to_base36(46), "1a");
    }

    #[test]
    fn reads_single_segment_kb_index() {
        let infos = read_segment_infos(&kb_dir()).unwrap();
        assert_eq!(infos.len(), 1);
        // the KB fixture's .cfs is "_2.cfs" => segName "_2"
        assert_eq!(infos[0].name, "_2");
        assert_eq!(infos[0].doc_count, 20);
    }
}
