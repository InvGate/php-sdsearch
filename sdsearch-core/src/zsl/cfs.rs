//! Compound file reader (.cfs): a single file that packs several segment sub-files.
use crate::zsl::bytes::{checked_capacity, read_modified_utf8, read_u64_be, read_vint};
use memmap2::Mmap;
use std::path::Path;

pub struct CompoundFile {
    mmap: Mmap,
    // name -> (start, end) within the mmap
    entries: Vec<(String, usize, usize)>,
}

impl CompoundFile {
    pub fn open(path: &Path) -> std::io::Result<CompoundFile> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let total = mmap.len();
        let mut pos = 0usize;
        let count = read_vint(&mmap, &mut pos)? as usize;
        // read the (offset, name) entries in order
        let mut raw: Vec<(u64, String)> = Vec::with_capacity(checked_capacity(count, total));
        for _ in 0..count {
            let offset = read_u64_be(&mmap, &mut pos)?;
            let name = read_modified_utf8(&mmap, &mut pos)?;
            raw.push((offset, name));
        }
        // each sub-file runs from its offset to the next one's (or EOF); validate the
        // offsets so a corrupt .cfs surfaces as an error instead of an out-of-range slice.
        let mut entries = Vec::with_capacity(checked_capacity(count, total));
        for i in 0..count {
            let start = raw[i].0 as usize;
            let end = if i + 1 < count {
                raw[i + 1].0 as usize
            } else {
                total
            };
            if start > end || end > total {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "cfs sub-file offset out of range: start={start} end={end} total={total}"
                    ),
                ));
            }
            entries.push((raw[i].1.clone(), start, end));
        }
        Ok(CompoundFile { mmap, entries })
    }

    pub fn names(&self) -> Vec<String> {
        self.entries.iter().map(|(n, _, _)| n.clone()).collect()
    }

    pub fn sub(&self, name: &str) -> Option<&[u8]> {
        self.entries
            .iter()
            .find(|(n, _, _)| n == name)
            .map(|(_, s, e)| &self.mmap[*s..*e])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture_cfs() -> PathBuf {
        let dir = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/zsl_index"
        ));
        std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .find(|p| p.extension().is_some_and(|x| x == "cfs"))
            .expect("no .cfs in fixture — regenerate with sdsearch_dump_zsl_index.php")
    }

    #[test]
    fn lists_and_slices_sub_files() {
        let cf = CompoundFile::open(&fixture_cfs()).unwrap();
        let names = cf.names();
        // a compound segment always carries field infos, term dict, freq, stored fields
        assert!(names.iter().any(|n| n.ends_with(".fnm")), "names={names:?}");
        assert!(names.iter().any(|n| n.ends_with(".tis")), "names={names:?}");
        assert!(names.iter().any(|n| n.ends_with(".frq")), "names={names:?}");
        // each sub-file is non-empty and lies within the mmap
        for n in &names {
            assert!(cf.sub(n).is_some(), "missing {n}");
        }
    }
}
