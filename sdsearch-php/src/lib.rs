#![cfg_attr(windows, feature(abi_vectorcall))]
//! binding: exposes the native ZSL engine to PHP. JSON marshalling only; the query logic
//! lives in sdsearch-core. The boundary is panic-safe (never aborts the worker).

use ext_php_rs::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

use sdsearch_core::query::{InGroup, Occur, Query, QueryParams, WhereGroup, build_query, search};
use sdsearch_core::zsl::index::ZslIndex;
use sdsearch_core::zsl::runner::search_index;
use sdsearch_core::zsl::writer::{IndexWriter, WriterDoc, WriterField, WriterOpts};

// ---- JSON contract with PHP ----

#[derive(Deserialize)]
struct WhereDto {
    field: String,
    values: Vec<String>,
    occur: String,
}
#[derive(Deserialize)]
struct InDto {
    field: String,
    values: Vec<String>,
}
#[derive(Deserialize)]
struct ParamsDto {
    #[serde(default)]
    text: String,
    #[serde(default)]
    r#where: Vec<WhereDto>,
    #[serde(default)]
    r#in: Vec<InDto>,
    #[serde(default)]
    min_score: f32,
    #[serde(default)]
    limit: u64,
}
#[derive(Serialize)]
struct HitDto {
    id: u64,
    score: f32,
    fields: HashMap<String, String>,
}

fn occur_from(s: &str) -> Occur {
    match s {
        "must" => Occur::Must,
        "mustnot" => Occur::MustNot,
        _ => Occur::Should,
    }
}

/// fallible core: parses params, runs search_index, serializes hits. Errors are returned
/// as String (the boundary converts them into PhpException).
fn run(index_dir: &str, params_json: &str) -> Result<String, String> {
    let dto: ParamsDto =
        serde_json::from_str(params_json).map_err(|e| format!("sdsearch: bad params json: {e}"))?;
    let params = QueryParams {
        text: dto.text,
        where_groups: dto
            .r#where
            .into_iter()
            .map(|w| WhereGroup {
                field: w.field,
                values: w.values,
                occur: occur_from(&w.occur),
            })
            .collect(),
        in_groups: dto
            .r#in
            .into_iter()
            .map(|i| InGroup {
                field: i.field,
                values: i.values,
            })
            .collect(),
        fuzzy_similarity: 0.5,
        fuzzy_prefix_len: 3,
        wildcard_min_prefix: 0,
    };
    let hits = search_index(
        Path::new(index_dir),
        &params,
        dto.min_score,
        dto.limit as usize,
    )
    .map_err(|e| format!("sdsearch: {e}"))?;
    let out: Vec<HitDto> = hits
        .into_iter()
        .map(|h| HitDto {
            id: h.id as u64,
            score: h.score,
            fields: h.fields,
        })
        .collect();
    serde_json::to_string(&out).map_err(|e| format!("sdsearch: serialize hits: {e}"))
}

/// engine version; smoke test that the extension loads.
#[php_function]
pub fn sdsearch_version() -> String {
    sdsearch_core::version().to_string()
}

#[php_class]
#[php(name = "SdSearch\\Engine")]
pub struct Engine;

#[php_impl]
impl Engine {
    pub fn __construct() -> Self {
        Engine
    }

    /// searches a ZSL index (JSON contract). Panic-safe boundary: a Rust panic or Err =>
    /// a catchable PhpException, NEVER an unwind across extern "C".
    pub fn search(&self, index_dir: String, params_json: String) -> PhpResult<String> {
        let result = catch_unwind(AssertUnwindSafe(|| run(&index_dir, &params_json)));
        match result {
            Ok(Ok(json)) => Ok(json),
            Ok(Err(msg)) => Err(PhpException::default(msg)),
            Err(_) => Err(PhpException::default(
                "sdsearch: panic during search".to_string(),
            )),
        }
    }
}

// ---- write FFI: add_document JSON contract ----

#[derive(Deserialize)]
struct WriterFieldDto {
    name: String,
    value: String,
    kind: String,
}
#[derive(Deserialize)]
struct WriterDocDto {
    #[serde(default)]
    fields: Vec<WriterFieldDto>,
}

fn field_from(dto: WriterFieldDto) -> Result<WriterField, String> {
    match dto.kind.as_str() {
        "text" => Ok(WriterField::text(&dto.name, &dto.value)),
        "keyword" => Ok(WriterField::keyword(&dto.name, &dto.value)),
        "unindexed" => Ok(WriterField::unindexed(&dto.name, &dto.value)),
        other => Err(format!("sdsearch: unknown kind: {other}")),
    }
}

fn doc_from_json(doc_json: &str) -> Result<WriterDoc, String> {
    let dto: WriterDocDto =
        serde_json::from_str(doc_json).map_err(|e| format!("sdsearch: bad doc json: {e}"))?;
    let mut fields = Vec::with_capacity(dto.fields.len());
    for f in dto.fields {
        fields.push(field_from(f)?);
    }
    Ok(WriterDoc { fields })
}

/// opens the streaming writer (fallible core of the `open` boundary).
fn open_writer(index_dir: &str) -> Result<IndexWriter, String> {
    IndexWriter::open(Path::new(index_dir), WriterOpts::default()).map_err(|e| {
        if e.kind() == std::io::ErrorKind::WouldBlock {
            "sdsearch: index locked by another writer".to_string()
        } else {
            format!("sdsearch: open: {e}")
        }
    })
}

/// `SdSearch\Writer`: FFI bridge from the native `IndexWriter` to the write path from PHP.
/// JSON marshalling + delegation to `sdsearch-core` only; the boundary is panic-safe
/// (never aborts the worker), like `Engine`.
#[php_class]
#[php(name = "SdSearch\\Writer")]
pub struct Writer {
    inner: Option<IndexWriter>,
    /// cached ZSL reader over the base snapshot, opened ONCE in `open()` and reused by
    /// every `find_doc_id` (avoids re-opening the index —mmap + dictionary— per document, the
    /// dominant cost of the update loop). Valid for the whole batch because the writer holds the
    /// write-lock and does not commit mid-way => the on-disk generation does not change; the
    /// segments the writer buffers stay invisible until commit, so the reader sees the base.
    /// Cleared on `commit`/`optimize` because there the writer is consumed and the base changes.
    reader: Option<ZslIndex>,
}

/// fallible core of `find_doc_id`: runs the read-path build_query+search with an InGroup over
/// `<id_field>_key` against the cached reader (writer's base). Without re-opening the index.
fn resolve_doc_id(index: &ZslIndex, id_field: &str, value: &str) -> Result<i64, String> {
    let params = QueryParams {
        text: String::new(),
        where_groups: Vec::new(),
        in_groups: vec![InGroup {
            field: id_field.to_string(),
            values: vec![value.to_string()],
        }],
        fuzzy_similarity: 0.5,
        fuzzy_prefix_len: 3,
        wildcard_min_prefix: 0,
    };
    let query = build_query(&params).map_err(|e| format!("sdsearch: build_query: {e}"))?;
    let hits = search(index, &query, 0.0, 1);
    Ok(hits.first().map_or(-1, |h| h.id as i64))
}

/// core of `find_doc_ids`: verbatim term query over `<field>:value` (the `field` is LITERAL,
/// already with its `_key` suffix as indexed — unlike `resolve_doc_id`, which adds the suffix) =>
/// ALL live internal doc-ids that match, against the cached reader. For multi-doc dedup /
/// remove (e.g. multi-language categories: one `id_key` with N docs, or `language_key:xx`
/// with many). The ids are in the same space as `delete_document`.
fn resolve_doc_ids(index: &ZslIndex, field: &str, value: &str) -> Vec<i64> {
    let query = Query::Term {
        field: Some(field.to_string()),
        text: value.to_string(),
    };
    let hits = search(index, &query, 0.0, usize::MAX);
    hits.iter().map(|h| h.id as i64).collect()
}

#[php_impl]
#[php(change_method_case = "none")]
impl Writer {
    pub fn __construct() -> Self {
        Writer {
            inner: None,
            reader: None,
        }
    }

    /// opens the streaming writer over an existing ZSL index (takes the write-lock) + a cached
    /// ZSL reader over the same base (for `find_doc_id` without re-opening per document).
    pub fn open(&mut self, index_dir: String) -> PhpResult<()> {
        let result = catch_unwind(AssertUnwindSafe(|| {
            let iw = open_writer(&index_dir)?;
            let reader = ZslIndex::open(Path::new(&index_dir))
                .map_err(|e| format!("sdsearch: open reader: {e}"))?;
            Ok::<_, String>((iw, reader))
        }));
        match result {
            Ok(Ok((iw, reader))) => {
                self.inner = Some(iw);
                self.reader = Some(reader);
                Ok(())
            }
            Ok(Err(msg)) => Err(PhpException::default(msg)),
            Err(_) => Err(PhpException::default("sdsearch: panic in open".to_string())),
        }
    }

    /// Tries to open the writer (takes the write-lock). Returns `true` if it took the lock and
    /// opened, `false` if the lock was already held (WouldBlock) — WITHOUT leaving a writer open.
    /// Throws for any other error. Lets the indexing feed distinguish "index busy" without
    /// string-matching the `open()` message. Panic-safe.
    pub fn try_open(&mut self, index_dir: String) -> PhpResult<bool> {
        let result = catch_unwind(AssertUnwindSafe(|| {
            match IndexWriter::open(Path::new(&index_dir), WriterOpts::default()) {
                Ok(iw) => {
                    let reader = ZslIndex::open(Path::new(&index_dir))
                        .map_err(|e| format!("sdsearch: open reader: {e}"))?;
                    Ok::<_, String>(Some((iw, reader)))
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => Ok(None),
                Err(e) => Err(format!("sdsearch: try_open: {e}")),
            }
        }));
        match result {
            Ok(Ok(Some((iw, reader)))) => {
                self.inner = Some(iw);
                self.reader = Some(reader);
                Ok(true)
            }
            Ok(Ok(None)) => Ok(false),
            Ok(Err(msg)) => Err(PhpException::default(msg)),
            Err(_) => Err(PhpException::default(
                "sdsearch: panic in try_open".to_string(),
            )),
        }
    }

    /// resolves `<id_field>_key:value` against the writer's base snapshot (same lock, without
    /// committing) => GLOBAL internal doc-id, same space as `delete_document`. `-1` if there
    /// is no match (matches the host's DB-id lookup returning false/null). Requires an open writer.
    pub fn find_doc_id(&self, id_field: String, value: String) -> PhpResult<i64> {
        let index = self
            .reader
            .as_ref()
            .ok_or_else(|| PhpException::default("sdsearch: writer not open".to_string()))?;
        let result = catch_unwind(AssertUnwindSafe(|| {
            resolve_doc_id(index, &id_field, &value)
        }));
        match result {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(msg)) => Err(PhpException::default(msg)),
            Err(_) => Err(PhpException::default(
                "sdsearch: panic in find_doc_id".to_string(),
            )),
        }
    }

    /// resolves `<field>:value` (LITERAL field, already with `_key` suffix as indexed) => ALL
    /// live internal doc-ids that match, against the cached reader (without re-opening the index).
    /// For multi-doc remove / dedup: multi-language categories (one `id_key` => N docs), or
    /// `language_key:<lang>` (many). Each id is in the same space as `delete_document`. Returns
    /// [] if no match. Requires an open writer.
    pub fn find_doc_ids(&self, field: String, value: String) -> PhpResult<Vec<i64>> {
        let index = self
            .reader
            .as_ref()
            .ok_or_else(|| PhpException::default("sdsearch: writer not open".to_string()))?;
        let result = catch_unwind(AssertUnwindSafe(|| resolve_doc_ids(index, &field, &value)));
        match result {
            Ok(v) => Ok(v),
            Err(_) => Err(PhpException::default(
                "sdsearch: panic in find_doc_ids".to_string(),
            )),
        }
    }

    /// deletes a doc by its GLOBAL internal id (same space as `find_doc_id`). Negative or
    /// out of range = silent no-op (matches the core's behavior). Requires an open writer.
    pub fn delete_document(&mut self, doc_id: i64) -> PhpResult<()> {
        let iw = self
            .inner
            .as_mut()
            .ok_or_else(|| PhpException::default("sdsearch: writer not open".to_string()))?;
        if doc_id < 0 {
            return Ok(());
        }
        let did = doc_id as usize;
        let result = catch_unwind(AssertUnwindSafe(|| {
            iw.delete_document(did);
        }));
        match result {
            Ok(()) => Ok(()),
            Err(_) => Err(PhpException::default(
                "sdsearch: panic in delete_document".to_string(),
            )),
        }
    }

    /// buffers a doc; JSON contract: `{"fields":[{"name","value","kind":"text"|"keyword"|"unindexed"}]}`.
    pub fn add_document(&mut self, doc_json: String) -> PhpResult<()> {
        let iw = self
            .inner
            .as_mut()
            .ok_or_else(|| PhpException::default("sdsearch: writer not open".to_string()))?;
        let result = catch_unwind(AssertUnwindSafe(|| {
            let doc = doc_from_json(&doc_json)?;
            iw.add_document(doc)
                .map_err(|e| format!("sdsearch: add: {e}"))
        }));
        match result {
            Ok(Ok(())) => Ok(()),
            Ok(Err(msg)) => Err(PhpException::default(msg)),
            Err(_) => Err(PhpException::default(
                "sdsearch: panic in add_document".to_string(),
            )),
        }
    }

    /// commits the buffer + pending deletes; consumes the writer (closes the index).
    pub fn commit(&mut self) -> PhpResult<i64> {
        let iw = self
            .inner
            .take()
            .ok_or_else(|| PhpException::default("sdsearch: writer not open".to_string()))?;
        self.reader = None;
        let result = catch_unwind(AssertUnwindSafe(|| {
            iw.commit().map_err(|e| format!("sdsearch: commit: {e}"))
        }));
        match result {
            Ok(Ok(report)) => Ok(report.doc_count as i64),
            Ok(Err(msg)) => Err(PhpException::default(msg)),
            Err(_) => Err(PhpException::default(
                "sdsearch: panic in commit".to_string(),
            )),
        }
    }

    /// commits and merges everything into a single compacted segment; consumes the writer.
    pub fn optimize(&mut self) -> PhpResult<i64> {
        let iw = self
            .inner
            .take()
            .ok_or_else(|| PhpException::default("sdsearch: writer not open".to_string()))?;
        self.reader = None;
        let result = catch_unwind(AssertUnwindSafe(|| {
            iw.optimize()
                .map_err(|e| format!("sdsearch: optimize: {e}"))
        }));
        match result {
            Ok(Ok(report)) => Ok(report.doc_count as i64),
            Ok(Err(msg)) => Err(PhpException::default(msg)),
            Err(_) => Err(PhpException::default(
                "sdsearch: panic in optimize".to_string(),
            )),
        }
    }

    /// total docs the index will see after commit (live base + flushed/buffered − deletes).
    pub fn document_count(&self) -> PhpResult<i64> {
        let iw = self
            .inner
            .as_ref()
            .ok_or_else(|| PhpException::default("sdsearch: writer not open".to_string()))?;
        let result = catch_unwind(AssertUnwindSafe(|| iw.document_count()));
        match result {
            Ok(count) => Ok(count as i64),
            Err(_) => Err(PhpException::default(
                "sdsearch: panic in document_count".to_string(),
            )),
        }
    }
}

#[php_module]
pub fn module(module: ModuleBuilder) -> ModuleBuilder {
    // the module/extension name is forced to "sdsearch"; registers the classes.
    module
        .name("sdsearch")
        .class::<Engine>()
        .class::<Writer>()
        .function(wrap_function!(sdsearch_version))
}
