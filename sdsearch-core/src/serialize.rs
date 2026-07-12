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

/// reads a varint from `data` starting at `pos`, advancing `pos`.
///
/// # Errors
/// Returns `Err` if the buffer ends mid-varint (truncated) or the encoding is
/// overlong (more than 64 bits of payload).
pub fn read_vint(data: &[u8], pos: &mut usize) -> std::io::Result<u64> {
    let mut result: u64 = 0;
    let mut shift = 0u32;
    loop {
        let byte = *data.get(*pos).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("truncated varint at offset {}", *pos),
            )
        })?;
        *pos += 1;
        if shift >= 64 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "overlong varint (more than 64 bits)",
            ));
        }
        result |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_boundary_values() {
        for v in [
            0u64,
            1,
            127,
            128,
            300,
            16_383,
            16_384,
            1_000_000,
            u64::from(u32::MAX),
        ] {
            let mut buf = Vec::new();
            write_vint(&mut buf, v);
            let mut pos = 0;
            assert_eq!(read_vint(&buf, &mut pos).unwrap(), v, "value {v}");
            assert_eq!(pos, buf.len(), "consumed all bytes for {v}");
        }
    }

    #[test]
    fn read_vint_errors_on_truncation() {
        // 0x80 sets the continuation bit but there is no next byte
        let mut pos = 0;
        assert!(read_vint(&[0x80], &mut pos).is_err());
        // empty buffer
        let mut pos = 0;
        assert!(read_vint(&[], &mut pos).is_err());
    }

    #[test]
    fn read_vint_errors_on_overlong_encoding() {
        // 11 continuation bytes then a terminator: would overflow the shift
        let mut data = vec![0x80u8; 11];
        data.push(0x01);
        let mut pos = 0;
        assert!(read_vint(&data, &mut pos).is_err());
    }

    #[test]
    fn sequential_reads_advance_position() {
        let mut buf = Vec::new();
        write_vint(&mut buf, 5);
        write_vint(&mut buf, 200);
        write_vint(&mut buf, 9);
        let mut pos = 0;
        assert_eq!(read_vint(&buf, &mut pos).unwrap(), 5);
        assert_eq!(read_vint(&buf, &mut pos).unwrap(), 200);
        assert_eq!(read_vint(&buf, &mut pos).unwrap(), 9);
        assert_eq!(pos, buf.len());
    }
}
