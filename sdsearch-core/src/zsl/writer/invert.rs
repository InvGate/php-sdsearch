//! Builds the in-RAM inverted index for a batch of docs (conceptual inverse of the postings
//! consumption in `search.rs`). Produces terms ordered by the SAME key as ZSL
//! (`fieldName · \0 · text`, `ksort SORT_STRING`), 1-based positions for text and 0 for keyword,
//! and the per-(indexed field, doc) lengths that feed `.nrm`.

use super::{WriterDoc, WriterOpts};
use crate::analysis::analyze;

#[derive(Debug, Clone, PartialEq)]
pub struct FieldMeta {
    pub name: String,
    pub indexed: bool,
}

/// A doc's stored field: field number, value and the `tokenized` bit (for the `.fdt` flag).
#[derive(Debug, Clone, PartialEq)]
pub struct StoredField {
    pub field_num: usize,
    pub value: String,
    pub tokenized: bool,
}

/// A term's postings: ascending doc-ids, each with its ascending positions.
#[derive(Debug, Clone, PartialEq)]
pub struct TermPostings {
    pub field_num: usize,
    pub text: String,
    pub docs: Vec<(usize, Vec<u32>)>,
}

impl TermPostings {
    pub fn doc_freq(&self) -> u32 {
        self.docs.len() as u32
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Inverted {
    /// fields in field-number order (first-seen), mirroring ZSL's `addField`.
    pub fields: Vec<FieldMeta>,
    /// terms ordered by `fieldName · \0 · text` (byte-wise == `ksort SORT_STRING`).
    pub terms: Vec<TermPostings>,
    /// per field-number: a column of per-doc lengths (`Some(numTerms)` if the field
    /// is present and non-empty in that doc; `None` = absent/empty → norm sentinel).
    /// NON-indexed fields get an empty column.
    pub norm_lengths: Vec<Vec<Option<u32>>>,
    /// per doc: stored fields in order of appearance.
    pub stored: Vec<Vec<StoredField>>,
    pub doc_count: usize,
}

pub fn invert(docs: &[WriterDoc], _opts: &WriterOpts) -> Inverted {
    use std::collections::{BTreeMap, HashMap};

    let doc_count = docs.len();
    let mut field_index: HashMap<String, usize> = HashMap::new();
    let mut fields: Vec<FieldMeta> = Vec::new();
    // (field_num, text) -> (doc -> positions). BTreeMap keeps doc-ids ascending.
    let mut term_map: HashMap<(usize, String), BTreeMap<usize, Vec<u32>>> = HashMap::new();
    let mut field_doc_numterms: HashMap<(usize, usize), u32> = HashMap::new();
    let mut stored: Vec<Vec<StoredField>> = vec![Vec::new(); doc_count];

    for (doc_id, doc) in docs.iter().enumerate() {
        for field in &doc.fields {
            // field number = first-seen order (== ZSL's addField).
            let field_num = *field_index.entry(field.name.clone()).or_insert_with(|| {
                fields.push(FieldMeta { name: field.name.clone(), indexed: false });
                fields.len() - 1
            });
            // NOTE (minor divergence vs ZSL, byte-diagnostic-only): ZSL clones a Text/Keyword
            // field to isIndexed=false if it produced NO tokens in ANY doc; here we mark it
            // indexed by its kind. This is self-consistent (our reader and ZSL's merge tolerate
            // it) and the host application always has content, so it does not affect the
            // interchange bar. Tune only if the byte-diff on the "field empty across the whole
            // batch" edge case matters.
            fields[field_num].indexed |= field.kind.is_indexed();

            if field.stored {
                stored[doc_id].push(StoredField {
                    field_num,
                    value: field.value.clone(),
                    tokenized: field.kind.is_tokenized(),
                });
            }

            if !field.kind.is_indexed() {
                continue;
            }

            if field.kind.is_tokenized() {
                let tokens = analyze(&field.value);
                if tokens.is_empty() {
                    // empty field / no tokens: treated as not indexed in THIS doc.
                    continue;
                }
                let mut position = 0u32;
                for tok in &tokens {
                    position += 1; // 1-based: first token at position 1
                    term_map
                        .entry((field_num, tok.clone()))
                        .or_default()
                        .entry(doc_id)
                        .or_default()
                        .push(position);
                }
                field_doc_numterms.insert((field_num, doc_id), tokens.len() as u32);
            } else {
                // keyword: NOT tokenized; term = whole value (case preserved), position 0.
                if field.value.is_empty() {
                    continue;
                }
                term_map
                    .entry((field_num, field.value.clone()))
                    .or_default()
                    .entry(doc_id)
                    .or_default()
                    .push(0);
                field_doc_numterms.insert((field_num, doc_id), 1);
            }
        }
    }

    let mut terms: Vec<TermPostings> = term_map
        .into_iter()
        .map(|((field_num, text), docmap)| TermPostings {
            field_num,
            text,
            docs: docmap.into_iter().collect(),
        })
        .collect();
    // ZSL's order: `fieldName · \0 · text` byte-wise. Since `\0` is the minimum byte,
    // comparing the (fieldName, text) tuple is equivalent and avoids building the key.
    terms.sort_by(|a, b| {
        (fields[a.field_num].name.as_str(), a.text.as_str())
            .cmp(&(fields[b.field_num].name.as_str(), b.text.as_str()))
    });

    let norm_lengths: Vec<Vec<Option<u32>>> = fields
        .iter()
        .enumerate()
        .map(|(fnum, meta)| {
            if meta.indexed {
                (0..doc_count).map(|d| field_doc_numterms.get(&(fnum, d)).copied()).collect()
            } else {
                Vec::new()
            }
        })
        .collect();

    Inverted { fields, terms, norm_lengths, stored, doc_count }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zsl::writer::WriterField;

    fn find<'a>(inv: &'a Inverted, field_num: usize, text: &str) -> &'a TermPostings {
        inv.terms
            .iter()
            .find(|t| t.field_num == field_num && t.text == text)
            .unwrap_or_else(|| panic!("term ({field_num},{text}) faltante"))
    }

    #[test]
    fn assigns_field_numbers_in_first_seen_order() {
        let docs = vec![WriterDoc {
            fields: vec![
                WriterField::text("title", "New workflow"),
                WriterField::text("body", "hello"),
            ],
        }];
        let inv = invert(&docs, &WriterOpts::default());
        assert_eq!(inv.fields[0].name, "title");
        assert_eq!(inv.fields[1].name, "body");
        assert_eq!(inv.doc_count, 1);
    }

    #[test]
    fn tokenized_positions_are_1_based_and_terms_sorted_by_field_then_text() {
        let docs = vec![
            WriterDoc {
                fields: vec![
                    WriterField::text("title", "New workflow"),
                    WriterField::text("body", "new new"),
                ],
            },
            WriterDoc {
                fields: vec![
                    WriterField::text("title", "workflow done"),
                    WriterField::text("body", ""), // empty -> not indexed
                ],
            },
        ];
        let inv = invert(&docs, &WriterOpts::default());

        // field numbers: title=0, body=1
        let title = 0;
        let body = 1;

        // term order by (fieldName·\0·text): body\0new < title\0done < title\0new < title\0workflow
        let order: Vec<(usize, &str)> = inv.terms.iter().map(|t| (t.field_num, t.text.as_str())).collect();
        assert_eq!(
            order,
            vec![(body, "new"), (title, "done"), (title, "new"), (title, "workflow")]
        );

        // 1-based positions within the field
        assert_eq!(find(&inv, body, "new").docs, vec![(0, vec![1, 2])]); // "new new" -> pos 1,2
        assert_eq!(find(&inv, title, "new").docs, vec![(0, vec![1])]); // "New workflow" -> new@1
        assert_eq!(
            find(&inv, title, "workflow").docs,
            vec![(0, vec![2]), (1, vec![1])] // doc0 workflow@2, doc1 workflow@1
        );
        assert_eq!(find(&inv, title, "done").docs, vec![(1, vec![2])]); // "workflow done" -> done@2

        // docFreq
        assert_eq!(find(&inv, title, "workflow").doc_freq(), 2);
        assert_eq!(find(&inv, body, "new").doc_freq(), 1);
    }

    #[test]
    fn keyword_is_single_term_at_position_zero() {
        let docs = vec![WriterDoc {
            fields: vec![WriterField::keyword("id", "SD-42")],
        }];
        let inv = invert(&docs, &WriterOpts::default());
        // keyword NOT tokenized: term = whole value (case preserved), position 0
        let t = find(&inv, 0, "SD-42");
        assert_eq!(t.docs, vec![(0, vec![0])]);
    }

    #[test]
    fn norm_lengths_track_numterms_with_none_for_empty_fields() {
        let docs = vec![
            WriterDoc {
                fields: vec![
                    WriterField::text("title", "New workflow"), // 2 tokens
                    WriterField::text("body", "new new"),       // 2 tokens
                ],
            },
            WriterDoc {
                fields: vec![
                    WriterField::text("title", "workflow done"), // 2 tokens
                    WriterField::text("body", ""),               // empty -> None
                ],
            },
        ];
        let inv = invert(&docs, &WriterOpts::default());
        assert_eq!(inv.norm_lengths[0], vec![Some(2), Some(2)]); // title
        assert_eq!(inv.norm_lengths[1], vec![Some(2), None]); // body doc1 empty
    }

    #[test]
    fn unindexed_field_is_stored_only_no_term_no_norm() {
        let docs = vec![WriterDoc {
            fields: vec![
                WriterField::text("title", "hi"),
                WriterField::unindexed("id_attr", "999"),
            ],
        }];
        let inv = invert(&docs, &WriterOpts::default());
        // id_attr contributes no terms
        assert!(inv.terms.iter().all(|t| t.field_num == 0));
        assert!(!inv.fields[1].indexed);
        // empty norm column for the non-indexed field
        assert!(inv.norm_lengths[1].is_empty());
        // both fields stored (by default), in order of appearance; title tokenized, id_attr not
        assert_eq!(
            inv.stored[0],
            vec![
                StoredField { field_num: 0, value: "hi".into(), tokenized: true },
                StoredField { field_num: 1, value: "999".into(), tokenized: false },
            ]
        );
    }
}
