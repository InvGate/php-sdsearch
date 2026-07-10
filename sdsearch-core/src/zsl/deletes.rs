//! Deletes reader (.del): BitVector; set bit = deleted doc.
//!
//! Only the DENSE (pre-2.1) BitVector layout is supported: `i32 docCount`, `i32 bitCount`,
//! followed by the raw bitmap. ZSL also writes a SPARSE/DGaps layout (used for
//! incremental deletes, `_<seg>_<delGen>.del`), where the first `i32` is the marker
//! `0xFFFFFFFF` followed by a `(VInt dgap, byte)` stream. That layout, and picking the
//! most recent `delGen`, are not implemented yet.
use crate::zsl::bytes::read_i32_be;

/// Marker indicating the `.del` uses the sparse/DGaps layout instead of the dense one.
const SPARSE_MARKER: i32 = -1;

pub struct DeletedDocs {
    bits: Vec<u8>,
}

impl DeletedDocs {
    /// Reads a `.del` with the dense BitVector layout.
    ///
    /// # Errors
    /// Returns `Err` if it detects the sparse/DGaps marker (`0xFFFFFFFF`) in the
    /// first 4 bytes (that layout is not supported yet — see the module doc) or if
    /// the header is truncated.
    pub fn read(del: &[u8]) -> std::io::Result<DeletedDocs> {
        let mut pos = 0usize;
        let doc_count = read_i32_be(del, &mut pos)?;
        if doc_count == SPARSE_MARKER {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "sparse/DGaps .del layout not supported; only dense BitVector .del is handled",
            ));
        }
        let _bit_count = read_i32_be(del, &mut pos)?;
        Ok(DeletedDocs { bits: del.get(pos..).unwrap_or(&[]).to_vec() })
    }

    pub fn none() -> DeletedDocs {
        DeletedDocs { bits: Vec::new() }
    }

    pub fn is_deleted(&self, doc_id: usize) -> bool {
        let byte = doc_id / 8;
        match self.bits.get(byte) {
            Some(b) => b & (1 << (doc_id % 8)) != 0,
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_bitvector_and_reports_deleted() {
        // docCount=10, bitCount=1, one byte with bit 2 set (doc 2 deleted)
        let mut buf = Vec::new();
        buf.extend_from_slice(&10i32.to_be_bytes());
        buf.extend_from_slice(&1i32.to_be_bytes());
        buf.push(0b0000_0100);
        let d = DeletedDocs::read(&buf).unwrap();
        assert!(d.is_deleted(2));
        assert!(!d.is_deleted(0));
        assert!(!d.is_deleted(9));
    }

    #[test]
    fn none_reports_nothing_deleted() {
        assert!(!DeletedDocs::none().is_deleted(0));
    }

    #[test]
    fn errors_on_sparse_dgaps_marker() {
        // 0xFFFFFFFF (-1 as i32) => sparse/DGaps marker, not supported yet
        let mut buf = Vec::new();
        buf.extend_from_slice(&(-1i32).to_be_bytes());
        buf.extend_from_slice(&[0x00, 0x01]); // stub of (VInt dgap, byte); read must Err before touching it
        match DeletedDocs::read(&buf) {
            Err(e) => assert_eq!(e.kind(), std::io::ErrorKind::InvalidData),
            Ok(_) => panic!("expected Err on sparse .del marker"),
        }
    }
}
