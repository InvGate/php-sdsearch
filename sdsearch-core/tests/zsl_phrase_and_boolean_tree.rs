//! Frases y árboles booleanos sobre un segmento ZSL REAL en disco. `MemoryIndex` no ejercita
//! el decode de posiciones del `.prx`, que es justo donde una frase puede divergir.

use std::collections::HashMap;

use sdsearch_core::doc::{Document, FieldKind};
use sdsearch_core::index::MemoryIndex;
use sdsearch_core::query::{BoolNode, QueryParams, build_query, search};
use sdsearch_core::score::Similarity;
use sdsearch_core::segment::Segment;

fn corpus() -> MemoryIndex {
    let mut idx = MemoryIndex::new();
    for (title, body) in [
        ("the quick brown fox", "impresion rota en el piso 3"),
        ("quick quick fox runs", "todo bien"),
        ("lazy dog sleeps", "la impresión rota no anda"),
        ("the fox and the dog", "sin novedad"),
    ] {
        let mut d = Document::new();
        d.add("title", title, FieldKind::Text);
        d.add("body", body, FieldKind::Text);
        idx.add_document(d);
    }
    idx
}

fn on_disk(name: &str) -> (std::path::PathBuf, Segment) {
    let dir = std::env::temp_dir().join(format!("sdsearch_tree_{}_{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    corpus().write_to(&dir).unwrap();
    let seg = Segment::open(&dir).unwrap();
    (dir, seg)
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

fn ids(seg: &Segment, tree: BoolNode, accent: bool) -> Vec<usize> {
    let q = build_query(&params(tree, accent)).unwrap();
    let mut v: Vec<usize> = search(seg, &q, 0.0, 10).iter().map(|h| h.id).collect();
    v.sort_unstable();
    v
}

fn term(p: &str) -> BoolNode {
    BoolNode::Term(p.into())
}

#[test]
fn phrase_respects_order_and_adjacency_on_disk() {
    let (dir, seg) = on_disk("phrase");

    assert_eq!(
        ids(&seg, term("brown fox"), false),
        vec![0],
        "la frase existe"
    );
    assert!(
        ids(&seg, term("fox brown"), false).is_empty(),
        "el orden importa"
    );
    // en el doc 0 "brown" está en el medio (no matchea ahí), pero el doc 1
    // ("quick quick fox runs") tiene un segundo "quick" adyacente a "fox".
    assert_eq!(
        ids(&seg, term("quick fox"), false),
        vec![1],
        "la adyacencia importa: no matchea vía doc 0, pero sí vía el 2do 'quick' del doc 1"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn and_or_not_over_phrase_leaves_on_disk() {
    let (dir, seg) = on_disk("boolean");

    // OR: doc 0 y 1 tienen "quick", doc 2 tiene "lazy dog"
    assert_eq!(
        ids(
            &seg,
            BoolNode::Or(vec![term("quick"), term("lazy dog")]),
            false
        ),
        vec![0, 1, 2]
    );
    // AND: solo el doc 0 tiene "quick" Y "brown fox"
    assert_eq!(
        ids(
            &seg,
            BoolNode::And(vec![term("quick"), term("brown fox")]),
            false
        ),
        vec![0]
    );
    // AND NOT: "fox" está en 0, 1 y 3; sacando "quick" queda el 3
    assert_eq!(
        ids(
            &seg,
            BoolNode::And(vec![term("fox"), BoolNode::Not(Box::new(term("quick")))]),
            false
        ),
        vec![3]
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn accent_insensitive_phrase_on_disk() {
    let (dir, seg) = on_disk("accent");

    // sin acentos: cada uno encuentra solo su forma literal
    assert_eq!(ids(&seg, term("impresion rota"), false), vec![0]);
    assert_eq!(ids(&seg, term("impresión rota"), false), vec![2]);
    // con accent_insensitive: cualquiera de las dos formas encuentra las dos
    assert_eq!(ids(&seg, term("impresion rota"), true), vec![0, 2]);
    assert_eq!(ids(&seg, term("impresión rota"), true), vec![0, 2]);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_phrase_never_spans_two_fields_on_disk() {
    // el doc 0 termina el title con "fox" y arranca el body con "impresion": no es una frase.
    let (dir, seg) = on_disk("fields");
    assert!(ids(&seg, term("fox impresion"), false).is_empty());
    std::fs::remove_dir_all(&dir).ok();
}
