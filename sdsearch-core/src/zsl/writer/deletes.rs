//! Deletions (.del) writer: inverse of `zsl::deletes::DeletedDocs::read`.
//! Writes ONLY ZSL's DENSE layout (the only one `writeChanges` produces, SegmentInfo.php:1616):
//! `i32 docCount`, `i32 bitCount`, then `floor(docCount/8)+1` bitmap bytes (bit i = doc i deleted).

use crate::zsl::bytes::write_i32_be;
use std::collections::BTreeSet;

/// Bytes of a dense `.del` for `doc_count` docs, with `deleted` (local ids) marked.
pub fn write_del_file(doc_count: usize, deleted: &BTreeSet<usize>) -> Vec<u8> {
    // ZSL uses floor(docCount/8)+1, NOT ceil(docCount/8) (writeChanges, SegmentInfo.php:1620).
    let byte_count = doc_count / 8 + 1;
    let mut bitmap = vec![0u8; byte_count];
    for &id in deleted {
        if id < doc_count {
            bitmap[id / 8] |= 1 << (id % 8);
        }
    }
    let mut out = Vec::with_capacity(8 + byte_count);
    write_i32_be(&mut out, doc_count as i32);
    write_i32_be(&mut out, deleted.iter().filter(|&&id| id < doc_count).count() as i32);
    out.extend_from_slice(&bitmap);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zsl::deletes::DeletedDocs;

    #[test]
    fn round_trips_through_reader() {
        let mut del = BTreeSet::new();
        del.insert(2);
        del.insert(60);
        let bytes = write_del_file(63, &del);
        let read = DeletedDocs::read(&bytes).unwrap();
        assert!(read.is_deleted(2));
        assert!(read.is_deleted(60));
        assert!(!read.is_deleted(0));
        assert!(!read.is_deleted(62));
    }

    #[test]
    fn byte_exact_dense_layout() {
        let mut del = BTreeSet::new();
        del.insert(2); // byte 0, bit 2 => 0b0000_0100
        let bytes = write_del_file(10, &del);
        // docCount=10 -> byteCount = 10/8+1 = 2; bitCount=1
        assert_eq!(&bytes[0..4], &10i32.to_be_bytes());
        assert_eq!(&bytes[4..8], &1i32.to_be_bytes());
        assert_eq!(&bytes[8..], &[0b0000_0100u8, 0x00]);
        assert_eq!(bytes.len(), 8 + 2);
    }
}
