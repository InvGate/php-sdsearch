//! Tokenization parity against real Zend_Search_Lucene output.
//! The fixture is the reference (Zend Lucene) tokenizer's output for these inputs.

use std::fs;

#[derive(serde::Deserialize)]
struct Case {
    input: String,
    tokens: Vec<String>,
}

#[test]
fn matches_zend_lucene_tokenization() {
    let raw = fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/analyzer_parity.json"
    ))
    .expect("fixture missing — generate it with sdsearch_dump_tokens.php");
    let cases: Vec<Case> = serde_json::from_str(&raw).unwrap();

    for c in &cases {
        assert_eq!(
            sdsearch_core::analysis::analyze(&c.input),
            c.tokens,
            "tokenization mismatch for input {:?}",
            c.input
        );
    }
}
