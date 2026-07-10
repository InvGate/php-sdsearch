# ZSL on-disk index format

This is a byte-level reference for the [Zend Search Lucene](https://framework.zend.com/manual/1.12/en/zend.search.lucene.html)
(ZSL) index format as read and written by `sdsearch`. It complements
[`ARCHITECTURE.md`](../ARCHITECTURE.md), which describes the module layout; here we
describe the bytes themselves.

The layouts below are derived directly from the parsers in `sdsearch-core/src/zsl/`
(readers) and `sdsearch-core/src/zsl/writer/` (serializers). The goal of `sdsearch` is
byte-for-byte compatibility with a real ZSL install in both directions, so this document
describes the *actual* wire format, not an idealized one.

> Scope note: `sdsearch` supports the subset of the format an optimized ZSL index uses in
> practice. Two variants are intentionally **not** supported: the sparse/DGaps `.del`
> layout, and per-field "separate norm files" (an unoptimized index). Both surface as an
> error at open time rather than silent misreads. They are called out in the relevant
> sections below.

---

## 1. Primitive types

Every file is built from five primitives. All fixed-width integers are **big-endian**.

| Type | Size | Encoding |
|---|---|---|
| `Byte` | 1 | raw byte |
| `Int32` | 4 | big-endian, two's-complement when signed |
| `Int64` | 8 | big-endian |
| `VInt` | 1–10 | unsigned LEB128: 7 payload bits per byte, low group first, high bit (`0x80`) = "more bytes follow" |
| `String` | var | `VInt` character count, then the characters in *modified UTF-8* (see §4) |

`VInt` example: `300` = `0b1_0010_1100` → low 7 bits `0101100` with continuation, then
`0000010` → bytes `AC 02`.

Source: `sdsearch-core/src/serialize.rs` (`read_vint`/`write_vint`) and
`sdsearch-core/src/zsl/bytes.rs` (integers + strings).

---

## 2. Directory anatomy

An index is a directory. A generation pointer names the current segment list; the segment
list names the live segments; each segment is a single compound file (`.cfs`) that packs
all of that segment's sub-files, except the deletions bitmap (`.del`), which lives beside
the `.cfs` so it can be rewritten without touching the segment.

```
<index dir>/
├── segments.gen            generation pointer  → which segments_N is current
├── segments_<base36(N)>    the live segment list at generation N
├── _<a>.cfs                segment "_<a>": compound file (see §3)
├── _<a>.del  (or _<a>_<g>.del)   deletions for "_<a>" (outside the .cfs)
├── _<b>.cfs                segment "_<b>"
└── ...

_<a>.cfs  ─packs→   _<a>.fnm     field infos            (§5)
                    _<a>.fdt     stored field data      (§6)
                    _<a>.fdx     stored field index     (§6)
                    _<a>.tis     term dictionary        (§7)
                    _<a>.tii     term index (sparse)    (§7)
                    _<a>.frq     postings: doc + freq   (§8)
                    _<a>.prx     postings: positions    (§8)
                    _<a>.nrm     length norms           (§9)
```

**Multi-segment model.** Incremental writes append new segments; `optimize()`/merge
collapses them into one. A logical index is the concatenation of its segments in list
order: `zsl/index.rs` gives each segment a global document-ID base equal to the running
sum of prior segments' `docCount` (deletes included — the same numbering Lucene's
multi-segment readers use) and routes each lookup to the owning segment.

**Visibility / durability.** A freshly written segment file exists on disk but is
**invisible** until a new `segments_N` lists it and `segments.gen` points at that
generation. The generation flip is the atomic commit point: if the process dies mid-batch,
the old generation is still intact and the orphan segment files are simply never
referenced.

---

## 3. `.cfs` — compound file

A directory of named sub-files followed by their concatenated bodies.

```
┌────────┬─────────────────────────────┐
│ VInt   │ entryCount                  │
├────────┼──────────────┬──────────────┤
│ Int64  │ offset       │ ┐            │
│ String │ name         │ ├─ × entryCount (the directory)
├────────┴──────────────┴──────────────┤
│ <sub-file bodies, concatenated>       │   sub-file i = bytes [offset_i, offset_{i+1})
└───────────────────────────────────────┘       last sub-file runs to EOF
```

`offset` is an absolute byte offset into the `.cfs` where that sub-file's body begins.
The reader mmaps the whole file and hands out `[offset_i, offset_{i+1})` slices (the last
runs to EOF), validating that offsets are non-decreasing and in range.

Source: `zsl/cfs.rs`, `zsl/writer/cfs.rs`.

---

## 4. Modified UTF-8 strings

A `String` is `VInt(charCount)` followed by the characters. Two things differ from plain
byte-length-prefixed UTF-8 and silently corrupt cross-engine reads if missed:

- **The prefix counts *characters* (code points), not bytes.** A 3-code-point string of
  multibyte characters has prefix `3` but more than 3 bytes of body.
- **NUL (`U+0000`) is encoded as two bytes `C0 80`**, never as `00` (Java-style modified
  UTF-8). Every other code point uses standard UTF-8, **including 4-byte sequences** for
  the supplementary planes (emoji, CJK extensions). ZSL's PHP writer stores a supplementary
  code point as one standard 4-byte sequence counted as one character — not as a UTF-16
  surrogate pair — and the reader decodes it the same way, so round-trips are exact.

Source: `zsl/bytes.rs` (`read_modified_utf8`/`write_modified_utf8`). The 4-byte branch
matters for real data: documents with emoji drifted before it was handled.

---

## 5. `.fnm` — field infos

Field number → name + flags. A field's number is its ordinal in this list (0-based); every
other file refers to fields by that number.

```
┌────────┬──────────────┐
│ VInt   │ fieldCount   │
├────────┼──────────────┤
│ String │ name         │ ┐
│ Byte   │ flags        │ ├─ × fieldCount
└────────┴──────────────┘ ┘
flags: bit0 = indexed, bit1 = tokenized, bit2 = stores norms, ...
```

`sdsearch`'s reader keeps `name` and the `indexed` bit (bit 0); the other bits exist in the
byte and are preserved by the writer.

Source: `zsl/fields.rs`.

---

## 6. `.fdt` / `.fdx` — stored fields

Stored fields are the verbatim values returned with a hit (as opposed to the inverted index
used to match). `.fdx` is a flat offset table; `.fdt` holds the data.

**`.fdx` (index):** one `Int64` per document — the byte offset of that document's record in
`.fdt`. The document count of the segment is `len(.fdx) / 8`.

```
.fdx: [ Int64 offset_doc0 ][ Int64 offset_doc1 ]...   (docCount entries)
```

**`.fdt` (data):** at a document's offset,

```
┌────────┬──────────────┐
│ VInt   │ storedCount  │
├────────┼──────────────┤
│ VInt   │ fieldNum     │ ┐   fieldNum is LOCAL to the segment
│ Byte   │ flags        │ │   flags: bit0 = tokenized, bit1 = binary
│ value  │ …            │ ├─ × storedCount
└────────┴──────────────┘ ┘
  value = String (modified UTF-8)        when bit1 (binary) = 0
        = VInt byteLen + byteLen bytes   when bit1 (binary) = 1
```

The host application stores only text, so `binary` is expected to be 0; the reader handles
the binary branch defensively. Tokenized `Text` fields carry a trailing `\n` that ZSL's
`compactText()` adds — it is a faithful part of the stored bytes and is **not** trimmed.

Source: `zsl/stored.rs`.

---

## 7. `.tis` / `.tii` — term dictionary

**`.tis` (full dictionary)** — all terms, grouped by field and sorted by text within a
field, with each term's `docFreq` and pointers into `.frq`/`.prx`.

```
┌────────┬──────────────────┐
│ Int32  │ marker 0xFFFFFFFD │  FORMAT_2_1
│ Int64  │ termCount         │
│ Int32  │ indexInterval     │
│ Int32  │ skipInterval      │
│ Int32  │ maxSkipLevels     │
├────────┼──────────────────┤
│ VInt   │ sharedPrefixChars │ ┐  shared with the PREVIOUS term's text (see below)
│ String │ suffix            │ │
│ VInt   │ fieldNum          │ ├─ × termCount
│ VInt   │ docFreq           │ │
│ VInt   │ freqDelta         │ │  accumulates → absolute .frq pointer
│ VInt   │ proxDelta         │ │  accumulates → absolute .prx pointer
└────────┴──────────────────┘ ┘
```

Prefix sharing: a term's text is `previousText[0..sharedPrefixChars] + suffix`. The writer
emits `sharedPrefixChars = 0` whenever the field number changes, so applying the shared
prefix against the previous text (regardless of field) reproduces the dictionary exactly.
The `freqDelta`/`proxDelta` are cumulative across the **whole file**, not reset per field.
`skipOffset` is omitted because skips are disabled (`docFreq < skipInterval`).

**`.tii` (sparse index)** — the same entry shape plus an `IndexDelta` (`VInt`) pointing into
`.tis`, holding every `indexInterval`-th term so a reader can seek without scanning the
whole `.tis`. `sdsearch`'s reader **does not read `.tii`**: it loads the entire `.tis` into
a compact in-memory buffer and binary-searches it. The writer still emits `.tii` (with a
header and a synthetic initial entry) because a stock ZSL install requires it.

Source: `zsl/terms.rs` (reader), `zsl/writer/terms.rs` (both files).

---

## 8. `.frq` / `.prx` — postings

Per-term document and position lists, reached via the term's `freqPointer`/`proxPointer`
from `.tis`.

**`.frq` (docs + freqs):** `docFreq` entries, doc IDs delta-encoded and ascending. The
term-frequency-of-1 case is folded into the doc delta's low bit:

```
per doc (× docFreq):
  VInt v
     docDelta = v >> 1              docId = previousDocId + docDelta
     if (v & 1) == 1:  freq = 1     (implicit, no extra byte)
     else:             VInt freq
```

**`.prx` (positions):** for each doc, `freq` position deltas (ascending, cumulative):

```
per doc (× docFreq):     ← walk .frq in parallel to know `freq` per doc
  per occurrence (× freq):
    VInt posDelta        position = previousPosition + posDelta
```

Source: `zsl/postings.rs`.

---

## 9. `.nrm` — length norms

One norm byte per document per **indexed** field, used to weight shorter fields higher at
score time.

```
┌──────────────┬───────────────────────────────┐
│ 3 bytes      │ 'N' 'R' 'M'                   │  header
│ Byte         │ format                        │
├──────────────┼───────────────────────────────┤
│ Byte × docCount │ norms for indexed field #1  │  fields in field-number order
│ Byte × docCount │ norms for indexed field #2  │
│ ...             │ ...                         │
└──────────────┴───────────────────────────────┘
```

Each norm byte is Lucene's `SmallFloat` (a.k.a. `Similarity::decodeNorm`) byte-float:

```
if byte == 0:            norm = 0.0
mantissa = byte & 0x07
exponent = (byte >> 3) & 0x1F
norm     = f32::from_bits( (exponent << 24) | (mantissa << 21) )
approx field length ≈ 1 / norm²      (inverse of encodeNorm(1/√len))
```

Source: `zsl/norms.rs`.

---

## 10. `.del` — deletions

A bitset of deleted document IDs for one segment; a set bit means the local document is
deleted. It lives **outside** the `.cfs` so deletions can be rewritten without touching the
segment. The delete generation in `segments_N` selects the filename: `-1` = none, `0` =
`<seg>.del`, `>0` = `<seg>_<base36(gen)>.del`.

Only the **dense** BitVector layout (pre-2.1) is supported:

```
┌────────┬──────────────┐
│ Int32  │ docCount     │   (if this reads 0xFFFFFFFF it's the SPARSE layout → error)
│ Int32  │ bitCount     │
├────────┼──────────────┤
│ Byte × │ bitmap       │   doc d deleted  ⟺  bit (d % 8) of byte (d / 8) is set
└────────┴──────────────┘
```

The sparse/DGaps layout (first `Int32` == `0xFFFFFFFF`, then a `(VInt dgap, Byte)` stream)
is rejected with an error rather than misread.

Source: `zsl/deletes.rs`.

---

## 11. `segments.gen` and `segments_N`

**`segments.gen`** — the generation pointer, written twice for torn-write detection.

```
┌────────┬──────────────┐
│ Int32  │ format 0xFFFFFFFE │
│ Int64  │ gen1         │
│ Int64  │ gen2         │   must equal gen1 (else the file was caught mid-write)
└────────┴──────────────┘
```

The current segment list is `segments_<base36(gen)>`, or `segments` when `gen == 0`.
`base36` is lowercase (`base_convert(n, 10, 36)` in ZSL): `10 → "a"`, `46 → "1a"`.

**`segments_N`** — the live segment list.

```
┌────────┬───────────────────────────────┐
│ Int32  │ format  0xFFFFFFFC (2.3) | 0xFFFFFFFD (2.1) │
│ Int64  │ version                       │
│ Int32  │ nameCounter                   │
│ Int32  │ segCount                      │
├────────┼───────────────────────────────┤   per segment (× segCount):
│ String │ name                          │
│ Int32  │ docCount   (maxDoc, incl. deletes) │
│ Int64  │ delGen     (-1 none / 0 / >0) │
│ ─ if format 2.3: ─────────────────────│
│ Int32  │ docStoreOffset                │   if != 0xFFFFFFFF:
│ String │   docStoreSegment  (optional) │      String + Byte follow
│ Byte   │   docStoreIsCompound (optional)│
│ ───────────────────────────────────────│
│ Byte   │ hasSingleNormFile             │
│ Int32  │ numField   (must be 0xFFFFFFFF)│   anything else = separate norm files → error
│ Byte   │ isCompound                    │
└────────┴───────────────────────────────┘
```

`docCount` is the per-segment maxDoc (deletes included) and is what the global document-ID
base is built from. `numField` other than `0xFFFFFFFF` means "separate norm files" (an
unoptimized index), which neither ZSL nor `sdsearch` supports — optimize the index first.

Source: `zsl/segments.rs`.

---

## 12. Worked hex examples

**`.fnm` with two fields** — `title` (indexed) and `id_attr` (not indexed):

```
02                          VInt fieldCount = 2
05 74 69 74 6C 65 01        String len=5 "title", flags=0x01 (indexed)
07 69 64 5F 61 74 74 72 00  String len=7 "id_attr", flags=0x00 (not indexed)
```

**`.del` with 10 docs, only doc 2 deleted:**

```
00 00 00 0A                 Int32 docCount = 10
00 00 00 01                 Int32 bitCount = 1
04                          bitmap: 0b0000_0100 → bit 2 set → doc 2 deleted
```

**`String "hi"`** and a **NUL character**:

```
02 68 69                    "hi": VInt charCount=2, 'h' 'i'
01 C0 80                    "\0": VInt charCount=1, NUL encoded C0 80
```

These match the round-trip assertions in `zsl/bytes.rs`, `zsl/fields.rs`, and
`zsl/deletes.rs`.
