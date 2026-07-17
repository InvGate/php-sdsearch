//! Norms (.nrm) writer. Inverse of `zsl::norms`. `.nrm` = "NRM" + byte 0xFF + one
//! column per indexed field (in field-number order), one byte per doc.
//! The byte = `encodeNorm(lengthNorm(numTerms) · docBoost)` (ZSL's SmallFloat);
//! a field absent/empty in a doc => `lengthNorm(0)` => byte 255.
//!
//! NOTE: `zsl::norms::decode_norm` now applies the correct SmallFloat exponent bias and is
//! the exact byte-for-byte inverse of the `NORM_TABLE` below (SmallFloat itself is still an
//! 8-bit lossy quantization of the length — that imprecision is inherent to the format, not
//! a bug in the decode). Here we replicate ZSL's REAL SmallFloat:
//! `byte b>0 -> f32::from_bits((b<<21) + 0x30000000)`, and `_floatToByte` (binary search +
//! round to nearest) identical to `Zend_Search_Lucene_Search_Similarity`.

/// SmallFloat table: `NORM_TABLE[b]` = the float value that byte `b` decodes to.
static NORM_TABLE: std::sync::LazyLock<[f32; 256]> = std::sync::LazyLock::new(|| {
    let mut t = [0f32; 256];
    for (b, slot) in t.iter_mut().enumerate().skip(1) {
        *slot = f32::from_bits(((b as u32) << 21).wrapping_add(0x3000_0000));
    }
    t
});

/// DefaultSimilarity's `lengthNorm`: `1/sqrt(numTerms)`. With `num_terms=0` it gives `inf`
/// (absent field), which `encode_norm` maps to 255.
pub fn length_norm(num_terms: u32) -> f32 {
    1.0 / (num_terms as f32).sqrt()
}

/// `Similarity::encodeNorm` / `_floatToByte`: binary search in the table + round
/// to the nearest value. Byte-exact replica of ZSL's algorithm.
pub fn encode_norm(f: f32) -> u8 {
    if f <= 0.0 {
        return 0;
    }
    let table = &*NORM_TABLE;
    let mut lo: i32 = 0;
    let mut hi: i32 = 255;
    while hi >= lo {
        let mid = ((hi + lo) >> 1) as usize;
        let delta = f - table[mid];
        if delta < 0.0 {
            hi = mid as i32 - 1;
        } else if delta > 0.0 {
            lo = mid as i32 + 1;
        } else {
            return mid as u8;
        }
    }
    // round to the nearest between table[hi] and table[hi+1]
    if hi != 255 && f - table[hi as usize] > table[hi as usize + 1] - f {
        (hi + 1) as u8
    } else {
        hi as u8
    }
}

/// Writes the complete `.nrm` for the batch. `norm_lengths[field]` is the field's column
/// (empty if not indexed); `Some(n)` = numTerms, `None` = absent/empty.
pub fn write_norms(norm_lengths: &[Vec<Option<u32>>], doc_boost: f32) -> Vec<u8> {
    let mut out = b"NRM".to_vec();
    out.push(0xFF);
    for col in norm_lengths {
        // empty column = non-indexed field (contributes no bytes to .nrm).
        for &len in col {
            let byte = match len {
                Some(n) => encode_norm(length_norm(n) * doc_boost),
                None => encode_norm(length_norm(0)), // absent: lengthNorm(0) without boost -> 255
            };
            out.push(byte);
        }
    }
    out
}

/// Writes `.nrm` by COPYING already-encoded byte columns (the merge does not re-encode). One
/// column per field in field-number order; empty column = non-indexed field (contributes no
/// bytes). Inverse of `zsl::norms::read_norms`. On merge, norms are copied verbatim.
pub fn write_norms_raw(cols: &[Vec<u8>]) -> Vec<u8> {
    let mut out = b"NRM".to_vec();
    out.push(0xFF);
    for col in cols {
        out.extend_from_slice(col);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zsl::norms::read_norms;

    #[test]
    fn encode_norm_matches_known_zsl_table_values() {
        assert_eq!(encode_norm(1.0), 124); // table[124] == 1.0
        assert_eq!(encode_norm(0.5), 120); // table[120] == 0.5
        assert_eq!(encode_norm(0.0), 0); // <=0 -> 0
    }

    #[test]
    fn length_norm_is_inverse_sqrt() {
        assert_eq!(length_norm(1), 1.0);
        assert_eq!(length_norm(4), 0.5);
    }

    #[test]
    fn absent_field_encodes_to_255() {
        assert_eq!(encode_norm(length_norm(0)), 255);
    }

    #[test]
    fn write_norms_lays_columns_in_field_order_skipping_unindexed() {
        // field0 indexed [1,4 tokens], field1 NOT indexed (empty col), field2 indexed [1, absent]
        let nrm = write_norms(&[vec![Some(1), Some(4)], vec![], vec![Some(1), None]], 1.0);
        assert_eq!(&nrm[0..4], b"NRM\xFF");
        // read_norms receives ONLY the indexed field names, in order
        let cols = read_norms(&nrm, &["f0".into(), "f2".into()], 2);
        assert_eq!(cols["f0"], vec![124, 120]); // 1.0->124, 0.5->120
        assert_eq!(cols["f2"], vec![124, 255]); // 1.0->124, absent->255
    }

    #[test]
    fn write_norms_raw_roundtrips_and_matches_encoded_bytes() {
        // field0 indexed (2 docs), field1 NOT indexed (empty col), field2 indexed (2 docs)
        let lengths = vec![vec![Some(1u32), Some(4)], vec![], vec![Some(1), None]];
        let encoded = write_norms(&lengths, 1.0);
        // raw byte columns, the same the encode would produce:
        let cols = vec![vec![124u8, 120], vec![], vec![124, 255]];
        let raw = write_norms_raw(&cols);
        // 1) byte-identical to write_norms (the merge must not diverge from the encode path)
        assert_eq!(raw, encoded);
        // 2) round-trip via the reader: columns per indexed field, in order
        let cols_back = read_norms(&raw, &["f0".into(), "f2".into()], 2);
        assert_eq!(cols_back["f0"], vec![124, 120]);
        assert_eq!(cols_back["f2"], vec![124, 255]);
    }
}
