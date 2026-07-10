//! ZSL format byte primitives: big-endian integers and modified-UTF-8 strings.
pub use crate::serialize::{read_vint, write_vint};

pub fn write_u32_be(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_be_bytes());
}

pub fn write_i32_be(buf: &mut Vec<u8>, v: i32) {
    write_u32_be(buf, v as u32);
}

pub fn write_i64_be(buf: &mut Vec<u8>, v: i64) {
    buf.extend_from_slice(&v.to_be_bytes());
}

/// Writes VInt(charCount) then the chars in modified-UTF-8 (inverse of `read_modified_utf8`).
/// charCount = number of code points; NUL is encoded as C0 80, the rest is standard UTF-8.
/// (BMP only, same as ZSL `writeString` and the reader.)
pub fn write_modified_utf8(buf: &mut Vec<u8>, s: &str) {
    write_vint(buf, s.chars().count() as u64);
    let mut ch = [0u8; 4];
    for c in s.chars() {
        if c == '\u{0}' {
            buf.extend_from_slice(&[0xC0, 0x80]);
        } else {
            buf.extend_from_slice(c.encode_utf8(&mut ch).as_bytes());
        }
    }
}

use std::io::{self, ErrorKind};

/// Error for a read that runs off the end of the buffer.
#[inline]
pub fn truncated(pos: usize) -> io::Error {
    io::Error::new(
        ErrorKind::UnexpectedEof,
        format!("truncated index data at offset {pos}"),
    )
}

/// Clamp a file-derived element count to what could physically fit in the
/// remaining bytes, so a corrupt count cannot request a huge allocation
/// (which would trigger an uncatchable allocator abort). Each element needs at
/// least one byte, so `count` can never legitimately exceed `remaining`.
#[inline]
pub fn checked_capacity(count: usize, remaining: usize) -> usize {
    count.min(remaining)
}

#[inline]
pub fn read_u32_be(data: &[u8], pos: &mut usize) -> io::Result<u32> {
    let end = pos.checked_add(4).ok_or_else(|| truncated(*pos))?;
    let b = data.get(*pos..end).ok_or_else(|| truncated(*pos))?;
    *pos = end;
    Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
}

#[inline]
pub fn read_i32_be(data: &[u8], pos: &mut usize) -> io::Result<i32> {
    Ok(read_u32_be(data, pos)? as i32)
}

#[inline]
pub fn read_u64_be(data: &[u8], pos: &mut usize) -> io::Result<u64> {
    let end = pos.checked_add(8).ok_or_else(|| truncated(*pos))?;
    let b = data.get(*pos..end).ok_or_else(|| truncated(*pos))?;
    *pos = end;
    let mut a = [0u8; 8];
    a.copy_from_slice(b);
    Ok(u64::from_be_bytes(a))
}

/// Reads a single byte, advancing `pos`.
#[inline]
pub fn read_byte(data: &[u8], pos: &mut usize) -> io::Result<u8> {
    let b = *data.get(*pos).ok_or_else(|| truncated(*pos))?;
    *pos += 1;
    Ok(b)
}

/// Reads VInt(charCount) then that many chars in modified-UTF-8.
/// modified-UTF-8 = UTF-8 except NUL is encoded as C0 80.
pub fn read_modified_utf8(data: &[u8], pos: &mut usize) -> io::Result<String> {
    let char_count = read_vint(data, pos)? as usize;
    let mut s = String::with_capacity(checked_capacity(
        char_count,
        data.len().saturating_sub(*pos),
    ));
    for _ in 0..char_count {
        let b0 = read_byte(data, pos)?;
        if b0 & 0x80 == 0 {
            s.push(b0 as char);
        } else if b0 & 0xE0 == 0xC0 {
            let b1 = read_byte(data, pos)?;
            let cp = (((b0 & 0x1F) as u32) << 6) | ((b1 & 0x3F) as u32);
            s.push(char::from_u32(cp).unwrap_or('\u{FFFD}')); // C0 80 -> 0
        } else if b0 & 0xF0 == 0xE0 {
            // 3-byte (BMP).
            let b1 = read_byte(data, pos)?;
            let b2 = read_byte(data, pos)?;
            let cp =
                (((b0 & 0x0F) as u32) << 12) | (((b1 & 0x3F) as u32) << 6) | ((b2 & 0x3F) as u32);
            s.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
        } else {
            // 4-byte (supplementary plane). ZSL's PHP writer stores standard UTF-8 (not Java's
            // modified-UTF-8 with surrogates), so a code point >= U+10000 is stored as ONE
            // 4-byte sequence and the prefix counts it as ONE code point. The native writer
            // (`write_modified_utf8` via `char::encode_utf8`) does the same → round-trip.
            let b1 = read_byte(data, pos)?;
            let b2 = read_byte(data, pos)?;
            let b3 = read_byte(data, pos)?;
            let cp = (((b0 & 0x07) as u32) << 18)
                | (((b1 & 0x3F) as u32) << 12)
                | (((b2 & 0x3F) as u32) << 6)
                | ((b3 & 0x3F) as u32);
            s.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
        }
    }
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::{
        checked_capacity, read_byte, read_i32_be, read_modified_utf8, read_u32_be, read_u64_be,
        write_i32_be, write_i64_be, write_modified_utf8, write_u32_be,
    };

    #[test]
    fn writes_big_endian_integers() {
        let mut buf = Vec::new();
        write_u32_be(&mut buf, 256);
        assert_eq!(buf, [0x00, 0x00, 0x01, 0x00]);

        let mut buf = Vec::new();
        write_i32_be(&mut buf, -3);
        assert_eq!(buf, [0xFF, 0xFF, 0xFF, 0xFD]);

        let mut buf = Vec::new();
        write_i64_be(&mut buf, 5);
        assert_eq!(buf, [0, 0, 0, 0, 0, 0, 0, 5]);
    }

    #[test]
    fn writes_modified_utf8_with_char_count_prefix() {
        let mut buf = Vec::new();
        write_modified_utf8(&mut buf, "hi");
        assert_eq!(buf, [0x02, b'h', b'i']);
    }

    #[test]
    fn encodes_nul_as_c0_80() {
        // one NUL char => VInt(1) + C0 80
        let mut buf = Vec::new();
        write_modified_utf8(&mut buf, "\u{0}");
        assert_eq!(buf, [0x01, 0xC0, 0x80]);
    }

    #[test]
    fn modified_utf8_roundtrips_through_reader() {
        // multibyte (2-byte ü, 3-byte €, 4-byte emoji/CJK-ext) + NUL: charCount counts code
        // points, UTF-8 bytes with NUL->C0 80. ZSL's PHP writer stores standard 4-byte (not
        // Java surrogates) → the reader must decode them as 1 code point (real documents with
        // emoji drifted without the 4-byte branch).
        for s in [
            "",
            "hi",
            "über",
            "a€b",
            "na\u{0}me",
            "TICKET-12345",
            "user@example.com",
            "a\u{1F600}b",
            "\u{20000}",
            "mix \u{1F4A9} end",
            "über\u{1F680}",
        ] {
            let mut buf = Vec::new();
            write_modified_utf8(&mut buf, s);
            // the charCount prefix must be the number of code points
            let mut pos = 0;
            assert_eq!(
                read_modified_utf8(&buf, &mut pos).unwrap(),
                s,
                "roundtrip {s:?}"
            );
            assert_eq!(pos, buf.len(), "consumed all bytes of {s:?}");
        }
    }

    #[test]
    fn reads_big_endian_integers() {
        let mut pos = 0;
        assert_eq!(
            read_u32_be(&[0x00, 0x00, 0x01, 0x00], &mut pos).unwrap(),
            256
        );
        assert_eq!(pos, 4);
        let mut pos = 0;
        assert_eq!(
            read_i32_be(&[0xFF, 0xFF, 0xFF, 0xFD], &mut pos).unwrap(),
            -3
        );
        let mut pos = 0;
        assert_eq!(read_u64_be(&[0, 0, 0, 0, 0, 0, 0, 5], &mut pos).unwrap(), 5);
    }

    #[test]
    fn reads_modified_utf8_string_with_char_count_prefix() {
        // VInt(2) + "hi"
        let data = [0x02, b'h', b'i'];
        let mut pos = 0;
        assert_eq!(read_modified_utf8(&data, &mut pos).unwrap(), "hi");
        assert_eq!(pos, 3);
    }

    #[test]
    fn decodes_c0_80_as_nul() {
        // VInt(1) + C0 80  => one NUL char
        let data = [0x01, 0xC0, 0x80];
        let mut pos = 0;
        assert_eq!(read_modified_utf8(&data, &mut pos).unwrap(), "\u{0}");
        assert_eq!(pos, 3);
    }

    #[test]
    fn read_integers_error_on_short_buffer() {
        let mut pos = 0;
        assert!(read_u32_be(&[0, 1, 2], &mut pos).is_err());
        let mut pos = 0;
        assert!(read_u64_be(&[0; 7], &mut pos).is_err());
        let mut pos = 0;
        assert!(read_byte(&[], &mut pos).is_err());
    }

    #[test]
    fn read_modified_utf8_errors_when_body_truncated() {
        // VInt(3) says 3 code points, but only 1 byte follows
        let mut pos = 0;
        assert!(read_modified_utf8(&[0x03, b'a'], &mut pos).is_err());
    }

    #[test]
    fn checked_capacity_clamps_to_remaining() {
        assert_eq!(checked_capacity(1_000_000, 8), 8);
        assert_eq!(checked_capacity(3, 100), 3);
    }
}
