//! Verifies that Term/MultiTerm give identical results in memory and on disk.

use sdsearch_core::doc::{Document, FieldKind};
use sdsearch_core::index::MemoryIndex;
use sdsearch_core::search::{fuzzy_query, multi_term_query, phrase_query, term_query, wildcard_query};
use sdsearch_core::segment::Segment;

fn corpus() -> MemoryIndex {
    let mut idx = MemoryIndex::new();
    for text in [
        "the quick brown fox",
        "quick quick fox runs",
        "lazy dog sleeps",
        "the fox and the dog",
    ] {
        let mut d = Document::new();
        d.add("body", text, FieldKind::Text);
        idx.add_document(d);
    }
    idx
}

fn ids(hits: &[sdsearch_core::search::Hit]) -> Vec<usize> {
    hits.iter().map(|h| h.id).collect()
}

#[test]
fn term_and_multiterm_match_between_memory_and_disk() {
    let mem = corpus();
    let dir = std::env::temp_dir().join(format!("sdsearch_rt_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    mem.write_to(&dir).unwrap();
    let seg = Segment::open(&dir).unwrap();

    // Term
    let a = term_query(&mem, "body", "fox", 0.0, 10);
    let b = term_query(&seg, "body", "fox", 0.0, 10);
    assert_eq!(ids(&a), ids(&b));
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x.score - y.score).abs() < 1e-6, "score mismatch {} vs {}", x.score, y.score);
        assert_eq!(x.fields, y.fields);
    }

    // MultiTerm
    let a = multi_term_query(&mem, "body", &["quick", "dog"], 0.0, 10);
    let b = multi_term_query(&seg, "body", &["quick", "dog"], 0.0, 10);
    assert_eq!(ids(&a), ids(&b));

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn wildcard_and_fuzzy_match_between_memory_and_disk() {
    let mem = corpus();
    let dir = std::env::temp_dir().join(format!("sdsearch_wf_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    mem.write_to(&dir).unwrap();
    let seg = Segment::open(&dir).unwrap();

    // wildcard: prefix "qui" present in the corpus (quick)
    let a = wildcard_query(&mem, "body", "qui*", 2, 0.0, 10);
    let b = wildcard_query(&seg, "body", "qui*", 2, 0.0, 10);
    assert_eq!(ids(&a), ids(&b));
    assert!(!a.is_empty(), "qui* should match quick");

    // fuzzy: typo of "quick" (quikk: 1 edit on the tail "ck"->"kk", similarity 0.8 > 0.6)
    let a = fuzzy_query(&mem, "body", "quikk", 0.6, 3, 0.0, 10);
    let b = fuzzy_query(&seg, "body", "quikk", 0.6, 3, 0.0, 10);
    assert_eq!(ids(&a), ids(&b));
    assert!(!a.is_empty(), "quikk should match quick");

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn phrase_matches_between_memory_and_disk() {
    let mem = corpus();
    let dir = std::env::temp_dir().join(format!("sdsearch_ph_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    mem.write_to(&dir).unwrap();
    let seg = Segment::open(&dir).unwrap();

    // corpus() includes "the quick brown fox" and "the fox and the dog"
    let a = phrase_query(&mem, "body", &["quick", "brown"], 0.0, 10);
    let b = phrase_query(&seg, "body", &["quick", "brown"], 0.0, 10);
    assert_eq!(ids(&a), ids(&b));
    assert!(!a.is_empty(), "quick brown should match");

    // phrase in the wrong order: empty in both backends
    let a = phrase_query(&mem, "body", &["brown", "quick"], 0.0, 10);
    let b = phrase_query(&seg, "body", &["brown", "quick"], 0.0, 10);
    assert_eq!(ids(&a), ids(&b));
    assert!(a.is_empty());

    std::fs::remove_dir_all(&dir).unwrap();
}
