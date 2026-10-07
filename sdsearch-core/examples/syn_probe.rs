//! throwaway: what does the bundled dictionary actually return for these tokens?
fn main() {
    let d = sdsearch_core::synonyms::global();
    for tok in ["impresora", "printer", "laptop", "notebook", "portatil"] {
        println!("{tok:12} -> {:?}", d.expand(tok));
    }
}
