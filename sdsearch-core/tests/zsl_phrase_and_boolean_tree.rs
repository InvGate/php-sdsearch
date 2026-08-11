//! Frases y árboles booleanos sobre un índice ZSL real en disco: se escribe con
//! `zsl::writer::IndexWriter` y se lee con `zsl::index::ZslIndex` (backed by `ZslSegment`), para
//! ejercitar el decode real de posiciones desde `.frq`/`.prx` (`ZslSegment::positions_for`),
//! que es justo donde una frase puede divergir de `MemoryIndex` (cuyas posiciones viven en un
//! `Vec<u32>` en memoria, sin pasar por ningún formato binario).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

use sdsearch_core::index::IndexReader;
use sdsearch_core::query::{BoolNode, QueryParams, build_query, search};
use sdsearch_core::score::Similarity;
use sdsearch_core::zsl::index::ZslIndex;
use sdsearch_core::zsl::writer::{IndexWriter, WriterDoc, WriterField, WriterOpts};

static COUNTER: AtomicU32 = AtomicU32::new(0);

/// Copies the committed KB fixture (20 docs of IT-support vocabulary: "vpn", "laptop",
/// "printer", "password", ... — see `tests/fixtures/zsl_expected_kb.json`) to a fresh temp dir.
/// `IndexWriter::open` requires an existing base index (it reads the generation), so we start
/// from a real committed index, same pattern as `zsl_restrict_filters.rs:18-31` and
/// `zsl_merge_memory_bound.rs:79-107`.
fn copy_kb_base(name: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("sdsearch_tree_{}_{n}_{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let src = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/zsl_index_kb"
    ));
    for entry in std::fs::read_dir(&src).unwrap() {
        let p = entry.unwrap().path();
        std::fs::copy(&p, dir.join(p.file_name().unwrap())).unwrap();
    }
    dir
}

/// Appends our 4-doc corpus on top of the KB fixture via the streaming `IndexWriter`, commits,
/// and reopens as a `ZslIndex`. None of our vocabulary ("fox"/"quick"/"brown"/"lazy"/"dog"/
/// "impresion"/"impresión") appears in the KB fixture, so our docs are the only ones any of
/// this file's queries can match — but their global doc ids continue AFTER the fixture's docs
/// (ids are never renumbered from 0), hence `base_offset`.
fn on_disk(name: &str) -> (PathBuf, usize, ZslIndex) {
    let dir = copy_kb_base(name);
    let base_offset = ZslIndex::open(&dir).unwrap().total_docs();

    let mut w = IndexWriter::open(&dir, WriterOpts::default()).unwrap();
    for (title, body) in [
        ("the quick brown fox", "impresion rota en el piso 3"),
        ("quick quick fox runs", "todo bien"),
        ("lazy dog sleeps", "la impresión rota no anda"),
        ("the fox and the dog", "sin novedad"),
    ] {
        w.add_document(WriterDoc {
            fields: vec![
                WriterField::text("title", title),
                WriterField::text("body", body),
            ],
        })
        .unwrap();
    }
    w.commit().unwrap();

    let idx = ZslIndex::open(&dir).unwrap();
    (dir, base_offset, idx)
}

fn params(tree: BoolNode, accent_insensitive: bool) -> QueryParams {
    QueryParams {
        text: String::new(),
        where_groups: vec![],
        in_groups: vec![],
        fuzzy_similarity: 0.5,
        fuzzy_prefix_len: 3,
        wildcard_min_prefix: 2,
        accent_insensitive,
        synonyms: false,
        field_weights: HashMap::new(),
        similarity: Similarity::Bm25,
        range_filters: vec![],
        match_all: vec![],
        sort: None,
        boolean_tree: Some(tree),
    }
}

fn ids(idx: &ZslIndex, tree: BoolNode, accent: bool) -> Vec<usize> {
    let q = build_query(&params(tree, accent)).unwrap();
    let mut v: Vec<usize> = search(idx, &q, 0.0, 10).iter().map(|h| h.id).collect();
    v.sort_unstable();
    v
}

fn term(p: &str) -> BoolNode {
    BoolNode::Term(p.into())
}

/// shifts our corpus-local ids (0..3) by the KB fixture's doc count, since our 4 docs are
/// appended after the fixture's rather than renumbered from 0.
fn at(base_offset: usize, local_ids: &[usize]) -> Vec<usize> {
    local_ids.iter().map(|i| i + base_offset).collect()
}

#[test]
fn phrase_respects_order_and_adjacency_on_disk() {
    let (dir, base_offset, idx) = on_disk("phrase");

    assert_eq!(
        ids(&idx, term("brown fox"), false),
        at(base_offset, &[0]),
        "la frase existe"
    );
    assert!(
        ids(&idx, term("fox brown"), false).is_empty(),
        "el orden importa"
    );
    // en el doc 0 "brown" está en el medio (no matchea ahí), pero el doc 1
    // ("quick quick fox runs") tiene un segundo "quick" adyacente a "fox".
    assert_eq!(
        ids(&idx, term("quick fox"), false),
        at(base_offset, &[1]),
        "la adyacencia importa: no matchea vía doc 0, pero sí vía el 2do 'quick' del doc 1"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn and_or_not_over_phrase_leaves_on_disk() {
    let (dir, base_offset, idx) = on_disk("boolean");

    // OR: doc 0 y 1 tienen "quick", doc 2 tiene "lazy dog"
    assert_eq!(
        ids(
            &idx,
            BoolNode::Or(vec![term("quick"), term("lazy dog")]),
            false
        ),
        at(base_offset, &[0, 1, 2])
    );
    // AND: solo el doc 0 tiene "quick" Y "brown fox"
    assert_eq!(
        ids(
            &idx,
            BoolNode::And(vec![term("quick"), term("brown fox")]),
            false
        ),
        at(base_offset, &[0])
    );
    // AND NOT: "fox" está en 0, 1 y 3; sacando "quick" queda el 3
    assert_eq!(
        ids(
            &idx,
            BoolNode::And(vec![term("fox"), BoolNode::Not(Box::new(term("quick")))]),
            false
        ),
        at(base_offset, &[3])
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn accent_insensitive_phrase_on_disk() {
    let (dir, base_offset, idx) = on_disk("accent");

    // sin acentos: cada uno encuentra solo su forma literal
    assert_eq!(
        ids(&idx, term("impresion rota"), false),
        at(base_offset, &[0])
    );
    assert_eq!(
        ids(&idx, term("impresión rota"), false),
        at(base_offset, &[2])
    );
    // con accent_insensitive: cualquiera de las dos formas encuentra las dos
    assert_eq!(
        ids(&idx, term("impresion rota"), true),
        at(base_offset, &[0, 2])
    );
    assert_eq!(
        ids(&idx, term("impresión rota"), true),
        at(base_offset, &[0, 2])
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_phrase_never_spans_two_fields_on_disk() {
    // el doc 0 termina el title con "fox" y arranca el body con "impresion": no es una frase.
    let (dir, _base_offset, idx) = on_disk("fields");
    assert!(ids(&idx, term("fox impresion"), false).is_empty());
    std::fs::remove_dir_all(&dir).ok();
}
