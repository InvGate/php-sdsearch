//! Generates the bundled synonym dictionary blob at `src/synonyms.bin` from a TSV
//! produced offline from OMW (WordNet EN + MCR ES), linked via the Collaborative
//! Interlingual Index. Regenerate the TSV with the Python extractor documented in
//! docs/API.md (uses the `wn` package; omw-en:1.4 + omw-es:1.4).
//! Run: `cargo run -p sdsearch-core --example gen_synonyms -- <path/to/omw_pairs.tsv>`

use std::io::BufRead;

use sdsearch_core::synonyms;

fn main() {
    let tsv = std::env::args()
        .nth(1)
        .expect("usage: gen_synonyms <pairs.tsv>");
    let file = std::fs::File::open(&tsv).expect("open tsv");
    let reader = std::io::BufReader::new(file);

    let mut owned: Vec<(String, Vec<String>)> = Vec::new();
    for line in reader.lines() {
        let line = line.expect("read line");
        let mut it = line.split('\t');
        let key = match it.next() {
            Some(k) if !k.is_empty() => k.to_string(),
            _ => continue,
        };
        let vals: Vec<String> = it.map(str::to_string).collect();
        if vals.is_empty() {
            continue;
        }
        owned.push((key, vals));
    }

    // from_pairs takes &[(&str, &[&str])]; build borrowed views over `owned`.
    let refs: Vec<(&str, Vec<&str>)> = owned
        .iter()
        .map(|(k, vs)| (k.as_str(), vs.iter().map(String::as_str).collect()))
        .collect();
    let pairs: Vec<(&str, &[&str])> = refs.iter().map(|(k, vs)| (*k, vs.as_slice())).collect();

    let dict = synonyms::from_pairs(&pairs);
    let bytes = dict.encode();
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/synonyms.bin");
    std::fs::write(path, &bytes).expect("write synonyms.bin");
    println!(
        "wrote {} bytes to {path} ({} keys)",
        bytes.len(),
        owned.len()
    );
}
