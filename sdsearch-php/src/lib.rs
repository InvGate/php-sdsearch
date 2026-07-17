#![cfg_attr(windows, feature(abi_vectorcall))]
//! binding: exposes the native ZSL engine to PHP. JSON marshalling only; the query logic
//! lives in sdsearch-core. The boundary is panic-safe (never aborts the worker).

use ext_php_rs::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::time::Duration;

use sdsearch_core::mlt::{MinShouldMatch, MltParams, RangeFilter};
use sdsearch_core::prf::PrfParams;
use sdsearch_core::query::{InGroup, Occur, Query, QueryParams, WhereGroup, build_query, search};
use sdsearch_core::score::Similarity;
use sdsearch_core::search::Hit;
use sdsearch_core::zsl::index::ZslIndex;
use sdsearch_core::zsl::runner::{more_like_this_index, search_index, search_prf_index};
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
    /// optional: accent-insensitive text matching (Spanish). Omitted = false.
    #[serde(default)]
    accent_insensitive: bool,
    /// optional: per-field score multipliers (field -> weight). Omitted = {} (equal).
    #[serde(default)]
    field_weights: HashMap<String, f32>,
    /// optional scoring algorithm: "bm25" (default) or "tfidf". Omitted = "bm25".
    #[serde(default)]
    similarity: Option<String>,
}
#[derive(Serialize)]
struct HitDto {
    id: u64,
    score: f32,
    fields: HashMap<String, String>,
}

fn default_prf_top_k() -> u64 {
    5
}
fn default_prf_num_terms() -> u64 {
    10
}
fn default_prf_feedback_weight() -> f32 {
    0.3
}
fn default_prf_min_term_freq() -> u32 {
    1
}
fn default_prf_min_doc_freq() -> u64 {
    1
}

#[derive(Deserialize)]
struct PrfDto {
    #[serde(default = "default_prf_top_k")]
    top_k: u64,
    #[serde(default = "default_prf_num_terms")]
    num_terms: u64,
    #[serde(default = "default_prf_feedback_weight")]
    feedback_weight: f32,
    #[serde(default)]
    fields: Vec<String>,
    #[serde(default = "default_prf_min_term_freq")]
    min_term_freq: u32,
    #[serde(default = "default_prf_min_doc_freq")]
    min_doc_freq: u64,
    #[serde(default)]
    max_doc_freq: Option<u64>,
    #[serde(default)]
    posting_budget: Option<u64>,
}

impl Default for PrfDto {
    fn default() -> Self {
        Self {
            top_k: default_prf_top_k(),
            num_terms: default_prf_num_terms(),
            feedback_weight: default_prf_feedback_weight(),
            fields: Vec::new(),
            min_term_freq: default_prf_min_term_freq(),
            min_doc_freq: default_prf_min_doc_freq(),
            max_doc_freq: None,
            posting_budget: None,
        }
    }
}

#[derive(Deserialize)]
struct SemanticParamsDto {
    #[serde(flatten)]
    base: ParamsDto,
    #[serde(default)]
    prf: PrfDto,
}

fn default_min_term_freq() -> u32 {
    2
}
fn default_max_query_terms() -> u64 {
    25
}
fn default_min_doc_freq() -> u64 {
    5
}

#[derive(Deserialize)]
struct MltTermFilterDto {
    field: String,
    value: String,
}

#[derive(Deserialize)]
struct MltRangeFilterDto {
    field: String,
    #[serde(default)]
    from: Option<f64>,
    #[serde(default)]
    to: Option<f64>,
}

/// minimum-should-match from JSON: a number (`2`) or a string (`"2"` / `"30%"`).
#[derive(Deserialize)]
#[serde(untagged)]
enum MinShouldMatchDto {
    Count(u32),
    Spec(String),
}

/// Parse the msm DTO into the core type. A trailing `%` is a percentage; otherwise an integer
/// count. Unparseable input is treated as "off" (None) rather than failing the whole query.
fn parse_min_should_match(dto: Option<MinShouldMatchDto>) -> Option<MinShouldMatch> {
    match dto {
        None => None,
        Some(MinShouldMatchDto::Count(n)) => Some(MinShouldMatch::Count(n)),
        Some(MinShouldMatchDto::Spec(s)) => {
            let s = s.trim();
            match s.strip_suffix('%') {
                Some(pct) => pct.trim().parse::<u8>().ok().map(MinShouldMatch::Percent),
                None => s.parse::<u32>().ok().map(MinShouldMatch::Count),
            }
        }
    }
}

#[derive(Deserialize)]
struct MltParamsDto {
    id_field: String,
    id_value: String,
    #[serde(default)]
    fields: Vec<String>,
    #[serde(default)]
    source_fields: Vec<String>,
    #[serde(default)]
    term_filters: Vec<MltTermFilterDto>,
    #[serde(default)]
    range_filters: Vec<MltRangeFilterDto>,
    #[serde(default)]
    min_should_match: Option<MinShouldMatchDto>,
    #[serde(default = "default_min_term_freq")]
    min_term_freq: u32,
    #[serde(default = "default_max_query_terms")]
    max_query_terms: u64,
    #[serde(default = "default_min_doc_freq")]
    min_doc_freq: u64,
    // absent -> None -> engine infers a safety default from the index size;
    // 0 -> explicitly unbounded/off; n -> explicit cap.
    #[serde(default)]
    max_doc_freq: Option<u64>,
    #[serde(default)]
    posting_budget: Option<u64>,
    #[serde(default)]
    timeout_ms: u64,
    #[serde(default)]
    field_weights: HashMap<String, f32>,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    min_score: f32,
}

fn occur_from(s: &str) -> Occur {
    match s {
        "must" => Occur::Must,
        "mustnot" => Occur::MustNot,
        _ => Occur::Should,
    }
}

/// Maps the shared search DTO into core `QueryParams` (used by both `run` and `run_semantic`).
/// Takes `dto` by value and moves its fields — callers that still need `dto.min_score` /
/// `dto.limit` afterward must read those (both `Copy`) before calling this.
fn query_params_from(dto: ParamsDto) -> Result<QueryParams, String> {
    let similarity = match dto.similarity.as_deref() {
        None | Some("bm25") => Similarity::Bm25,
        Some("tfidf") => Similarity::TfIdf,
        Some(other) => {
            return Err(format!(
                "sdsearch: unknown similarity {other:?} (expected \"bm25\" or \"tfidf\")"
            ));
        }
    };
    Ok(QueryParams {
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
        accent_insensitive: dto.accent_insensitive,
        field_weights: dto.field_weights,
        similarity,
    })
}

/// Maps a `Vec<Hit>` into the JSON hit-array contract shared by `run` and `run_semantic`.
/// `run_mlt` has a different (projecting) variant and is not covered by this helper.
fn hits_to_json(hits: Vec<Hit>) -> Result<String, String> {
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

/// fallible core: parses params, runs search_index, serializes hits. Errors are returned
/// as String (the boundary converts them into PhpException).
fn run(index_dir: &str, params_json: &str) -> Result<String, String> {
    let dto: ParamsDto =
        serde_json::from_str(params_json).map_err(|e| format!("sdsearch: bad params json: {e}"))?;
    let min_score = dto.min_score;
    let limit = dto.limit;
    let params = query_params_from(dto)?;
    let hits = search_index(Path::new(index_dir), &params, min_score, limit as usize)
        .map_err(|e| format!("sdsearch: {e}"))?;
    hits_to_json(hits)
}

/// fallible core of `Engine::semantic_query`: parses the search DTO + optional `prf`
/// object, runs the two-pass PRF search, serializes hits. Errors as String.
fn run_semantic(index_dir: &str, params_json: &str) -> Result<String, String> {
    let dto: SemanticParamsDto =
        serde_json::from_str(params_json).map_err(|e| format!("sdsearch: bad params json: {e}"))?;
    let min_score = dto.base.min_score;
    let limit = dto.base.limit;
    let params = query_params_from(dto.base)?;
    let prf = PrfParams {
        top_k: dto.prf.top_k as usize,
        num_terms: dto.prf.num_terms as usize,
        feedback_weight: dto.prf.feedback_weight,
        fields: dto.prf.fields,
        min_term_freq: dto.prf.min_term_freq,
        min_doc_freq: dto.prf.min_doc_freq as usize,
        max_doc_freq: dto.prf.max_doc_freq.map(|v| v as usize),
        posting_budget: dto.prf.posting_budget.map(|v| v as usize),
    };
    let hits = search_prf_index(
        Path::new(index_dir),
        &params,
        &prf,
        min_score,
        limit as usize,
    )
    .map_err(|e| format!("sdsearch: {e}"))?;
    hits_to_json(hits)
}

/// fallible core of `Engine::more_like_this`: parses MLT params, runs the query, projects
/// `source_fields` (if any), serializes hits. Errors returned as String (boundary -> PhpException).
fn run_mlt(index_dir: &str, params_json: &str) -> Result<String, String> {
    let dto: MltParamsDto = serde_json::from_str(params_json)
        .map_err(|e| format!("sdsearch: bad mlt params json: {e}"))?;
    let params = MltParams {
        fields: dto.fields,
        min_term_freq: dto.min_term_freq,
        max_query_terms: dto.max_query_terms as usize,
        min_doc_freq: dto.min_doc_freq as usize,
        max_doc_freq: dto.max_doc_freq.map(|v| v as usize),
        posting_budget: dto.posting_budget.map(|v| v as usize),
        timeout: if dto.timeout_ms > 0 {
            Some(Duration::from_millis(dto.timeout_ms))
        } else {
            None
        },
        term_filters: dto
            .term_filters
            .into_iter()
            .map(|f| (f.field, f.value))
            .collect(),
        range_filters: dto
            .range_filters
            .into_iter()
            .map(|f| RangeFilter {
                field: f.field,
                from: f.from,
                to: f.to,
            })
            .collect(),
        min_should_match: parse_min_should_match(dto.min_should_match),
        field_weights: dto.field_weights,
        size: dto.size as usize,
        min_score: dto.min_score,
    };
    let hits = more_like_this_index(Path::new(index_dir), &dto.id_field, &dto.id_value, &params)
        .map_err(|e| format!("sdsearch: {e}"))?;
    let project = !dto.source_fields.is_empty();
    let allow: HashSet<String> = dto.source_fields.into_iter().collect();
    let out: Vec<HitDto> = hits
        .into_iter()
        .map(|h| HitDto {
            id: h.id as u64,
            score: h.score,
            fields: if project {
                h.fields
                    .into_iter()
                    .filter(|(k, _)| allow.contains(k))
                    .collect()
            } else {
                h.fields
            },
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
#[php(change_method_case = "none")]
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

    /// More Like This: given a reference doc (by id-field value), returns similar docs as a
    /// JSON hit array. Panic-safe boundary, like `search`.
    pub fn more_like_this(&self, index_dir: String, params_json: String) -> PhpResult<String> {
        let result = catch_unwind(AssertUnwindSafe(|| run_mlt(&index_dir, &params_json)));
        match result {
            Ok(Ok(json)) => Ok(json),
            Ok(Err(msg)) => Err(PhpException::default(msg)),
            Err(_) => Err(PhpException::default(
                "sdsearch: panic during more_like_this".to_string(),
            )),
        }
    }

    /// Semantic search via two-pass pseudo-relevance feedback. Same params object as
    /// `search`, plus an optional `"prf"` object. Panic-safe boundary, like `search`.
    pub fn semantic_query(&self, index_dir: String, params_json: String) -> PhpResult<String> {
        let result = catch_unwind(AssertUnwindSafe(|| run_semantic(&index_dir, &params_json)));
        match result {
            Ok(Ok(json)) => Ok(json),
            Ok(Err(msg)) => Err(PhpException::default(msg)),
            Err(_) => Err(PhpException::default(
                "sdsearch: panic during semantic_query".to_string(),
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
        accent_insensitive: false,
        field_weights: HashMap::new(),
        similarity: Similarity::Bm25,
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
