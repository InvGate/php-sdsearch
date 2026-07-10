//! varint serialization helpers (unsigned LEB128) for the postings.

/// writes `v` as a varint at the end of `buf`
pub fn write_vint(buf: &mut Vec<u8>, mut v: u64) {
    loop {
        let mut byte = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            byte |= 0x80;
        }
        buf.push(byte);
        if v == 0 {
            break;
        }
    }
}

/// reads a varint from `data` starting at `pos`, advancing `pos`
pub fn read_vint(data: &[u8], pos: &mut usize) -> u64 {
    let mut result: u64 = 0;
    let mut shift = 0;
    loop {
        let byte = data[*pos];
        *pos += 1;
        result |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_boundary_values() {
        for v in [0u64, 1, 127, 128, 300, 16_383, 16_384, 1_000_000, u32::MAX as u64] {
            let mut buf = Vec::new();
            write_vint(&mut buf, v);
            let mut pos = 0;
            assert_eq!(read_vint(&buf, &mut pos), v, "value {v}");
            assert_eq!(pos, buf.len(), "consumed all bytes for {v}");
        }
    }

    #[test]
    fn sequential_reads_advance_position() {
        let mut buf = Vec::new();
        write_vint(&mut buf, 5);
        write_vint(&mut buf, 200);
        write_vint(&mut buf, 9);
        let mut pos = 0;
        assert_eq!(read_vint(&buf, &mut pos), 5);
        assert_eq!(read_vint(&buf, &mut pos), 200);
        assert_eq!(read_vint(&buf, &mut pos), 9);
        assert_eq!(pos, buf.len());
    }
}
