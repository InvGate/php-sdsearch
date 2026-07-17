//! Generates the bundled synonym dictionary blob at `src/synonyms.bin`.
//! Run: `cargo run -p sdsearch-core --example gen_synonyms`
//! Task 3 ships a small curated ES↔EN seed; Task 7 replaces the pair list with
//! the full OMW-derived data (same output path, no code change to the crate).

use sdsearch_core::synonyms;

fn main() {
    // (key, expansion terms). Values are canonical (accented) forms.
    let pairs: &[(&str, &[&str])] = &[
        ("laptop", &["notebook", "portátil"]),
        ("notebook", &["laptop", "portátil"]),
        ("portátil", &["laptop", "notebook"]),
        ("impresora", &["printer"]),
        ("printer", &["impresora"]),
        ("computadora", &["computer", "ordenador"]),
        ("computer", &["computadora", "ordenador"]),
        ("contraseña", &["password", "clave"]),
        ("password", &["contraseña", "clave"]),
        ("archivo", &["file", "fichero"]),
        ("file", &["archivo", "fichero"]),
        ("correo", &["email", "mail"]),
        ("email", &["correo", "mail"]),
    ];
    let dict = synonyms::from_pairs(pairs);
    let bytes = dict.encode();
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/synonyms.bin");
    std::fs::write(path, &bytes).expect("write synonyms.bin");
    println!("wrote {} bytes to {path}", bytes.len());
}
