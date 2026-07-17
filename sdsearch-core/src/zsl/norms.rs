//! Norms reader (.nrm): one norm byte per doc per indexed field.
use std::collections::HashMap;

pub fn read_norms(
    nrm: &[u8],
    indexed_fields: &[String],
    num_docs: usize,
) -> HashMap<String, Vec<u8>> {
    let mut out = HashMap::new();
    // header: 'NRM' + 1 format byte
    let mut pos = 4usize;
    for field in indexed_fields {
        if pos + num_docs > nrm.len() {
            break;
        }
        out.insert(field.clone(), nrm[pos..pos + num_docs].to_vec());
        pos += num_docs;
    }
    out
}

/// Decodes Lucene's norm byte-float (SmallFloat / Similarity::decodeNorm).
/// `byte b>0 -> f32::from_bits((b << 21) + 0x30000000)`. The `0x30000000` term is the
/// exponent bias (48 << 24); omitting it saturated `approx_field_len` to u32::MAX.
/// Byte-exact inverse of `zsl/writer/norms.rs` NORM_TABLE.
pub fn decode_norm(b: u8) -> f32 {
    if b == 0 {
        return 0.0;
    }
    let bits = (u32::from(b) << 21).wrapping_add(0x3000_0000);
    f32::from_bits(bits)
}

pub fn approx_field_len(norm_byte: u8) -> u32 {
    let n = decode_norm(norm_byte);
    if n <= 0.0 {
        return 1;
    }
    (1.0 / (n * n)).round().max(1.0) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_norm_bytes_per_field_in_field_order() {
        // header 'NRM' + 0xFF, then 2 docs for "title", 2 docs for "body"
        let mut buf = b"NRM".to_vec();
        buf.push(0xFF);
        buf.extend_from_slice(&[0x10, 0x11]); // title
        buf.extend_from_slice(&[0x20, 0x21]); // body
        let norms = read_norms(&buf, &["title".into(), "body".into()], 2);
        assert_eq!(norms.get("title").unwrap(), &vec![0x10, 0x11]);
        assert_eq!(norms.get("body").unwrap(), &vec![0x20, 0x21]);
    }

    #[test]
    fn approx_field_len_is_positive() {
        // a valid norm byte decodes to a length >= 1
        assert!(approx_field_len(0x7C) >= 1);
    }

    #[test]
    fn decode_norm_has_the_smallfloat_bias() {
        // In ZSL's SmallFloat table, byte 124 == 1.0 and byte 120 == 0.5
        // (see zsl/writer/norms.rs NORM_TABLE / encode_norm tests).
        assert!(
            (decode_norm(124) - 1.0).abs() < 1e-6,
            "got {}",
            decode_norm(124)
        );
        assert!(
            (decode_norm(120) - 0.5).abs() < 1e-6,
            "got {}",
            decode_norm(120)
        );
    }

    #[test]
    fn approx_field_len_recovers_small_token_counts() {
        // decode(124)=1.0 => 1/1^2 = 1 token; decode(120)=0.5 => 1/0.5^2 = 4 tokens.
        assert_eq!(approx_field_len(124), 1);
        assert_eq!(approx_field_len(120), 4);
    }
}
