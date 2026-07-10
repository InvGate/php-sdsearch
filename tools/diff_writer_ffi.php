<?php
/**
 * WRITER FFI DIFFERENTIAL: ZSL vs `SdSearch\Writer` (native extension).
 *
 * Unlike tools/diff_writer.php (which validates the `stream_writer` example binary),
 * this validates the real FFI boundary the host application will use: `SdSearch\Writer::open/
 * add_document/optimize` via the `.so` extension. Indexes the SAME KB-shaped corpus with (a) ZSL's
 * PHP writer and (b) `SdSearch\Writer`, and with ZSL as the SOLE READER of both (over lock-free
 * copies) compares: term-dict (field,text,docFreq), stored fields per doc, and the doc-set of a
 * query battery. Isolates the writer: any observed difference is the native writer's (via FFI), not
 * the reader's (which is ZSL on both sides) nor the host feed's (there is no DB here).
 *
 * Methodological gotcha: docFreq over a multi-segment index is NOT
 * segmentation-invariant → optimize() BOTH sides BEFORE comparing term-dict.
 *
 * Usage:
 *   cargo build -p sdsearch-php --release
 *   ZEND_LUCENE_PATH=/path/to/zsl/library \
 *   php -n -d extension=mbstring -d extension=iconv \
 *       -d extension=$(pwd)/target/release/libsdsearch.so tools/diff_writer_ffi.php
 */
require __DIR__ . '/zsl_bootstrap.php';

if (!extension_loaded('sdsearch')) {
    fwrite(STDERR, "missing native extension: run with -d extension=<abs>/target/release/libsdsearch.so\n");
    exit(2);
}

$cleanupLocks = function (string $dir): void {
    foreach (array_merge(glob("$dir/*.sti") ?: [], glob("$dir/*lock*") ?: []) as $f) { @unlink($f); }
};

// copies $src -> $dst WITHOUT the lock files (so ZSL can safely re-open an
// index that just had an active writer -- same trick as writer_ffi_smoke.php).
function lockFreeCopy(string $src, string $dst): void {
    if (is_dir($dst)) { array_map('unlink', glob("$dst/*") ?: []); } else { @mkdir($dst, 0777, true); }
    foreach (glob("$src/*") ?: [] as $f) {
        if (!str_contains(basename($f), 'lock')) { copy($f, "$dst/" . basename($f)); }
    }
}

// maps a corpus kind to a Zend_Search_Lucene_Field (== diff_writer.php).
function zslField(string $n, string $v, string $kind) {
    switch ($kind) {
        case 'text':      return Zend_Search_Lucene_Field::Text($n, $v);
        case 'keyword':   return Zend_Search_Lucene_Field::Keyword($n, $v);
        case 'unindexed': return Zend_Search_Lucene_Field::UnIndexed($n, $v);
        default: throw new InvalidArgumentException("unknown kind: $kind");
    }
}

/**
 * FIXED KB-shaped corpus (12 docs). Field names ALREADY carry the `_key`/`_attr` suffix as the
 * host feed would produce them (this tool tests the WRITER, not the schema→suffix translation).
 * The kind per field name is kept CONSISTENT across the whole corpus: title/body=text,
 * id_key/category_id_key/tags_key=keyword, status_id_attr/notes_id_attr=unindexed.
 *
 * Edge cases covered:
 *   B0/B1     multi-word Text.
 *   B2        unicode Text (accents: café, piñata, mañana, niño).
 *   B3        unicode Text + emoji (🎉 🚀 éxito 🙌).
 *   B0/B4/B8  docFreq('body','vpn')=3 across docs.
 *   B0/B1/B2/B3/B9  numeric Keyword `id_key` (includes "0" and a large value).
 *   B5/B11    verbatim MULTI-WORD Keyword `tags_key` = "High Priority Urgent" (a single token,
 *             not tokenized, unlike a Text field with the same text).
 *   B6        UnIndexed `notes_id_attr` (stored, WITHOUT a term).
 *   B7        punctuation + case-folding ("WORK Work woRK" -> a single term 'work').
 *   B9        numeric Keyword "0" and "999999999" (boundaries).
 *   B10       more unicode docFreq (café appears twice in the corpus).
 */
function corpus(): array {
    return [
        ['docid' => 'B0', 'fields' => [
            ['name' => 'title', 'value' => 'VPN Setup Guide', 'kind' => 'text'],
            ['name' => 'body', 'value' => 'Configure the VPN client for remote access to the office network.', 'kind' => 'text'],
            ['name' => 'id_key', 'value' => '1001', 'kind' => 'keyword'],
            ['name' => 'category_id_key', 'value' => '10', 'kind' => 'keyword'],
            ['name' => 'status_id_attr', 'value' => 'published', 'kind' => 'unindexed'],
        ]],
        ['docid' => 'B1', 'fields' => [
            ['name' => 'title', 'value' => 'Password Reset Instructions', 'kind' => 'text'],
            ['name' => 'body', 'value' => 'How to reset your password using the self service portal.', 'kind' => 'text'],
            ['name' => 'id_key', 'value' => '1002', 'kind' => 'keyword'],
            ['name' => 'category_id_key', 'value' => '11', 'kind' => 'keyword'],
            ['name' => 'status_id_attr', 'value' => 'published', 'kind' => 'unindexed'],
        ]],
        ['docid' => 'B2', 'fields' => [
            ['name' => 'title', 'value' => 'Configuración de café y mañana', 'kind' => 'text'],
            ['name' => 'body', 'value' => 'Instalación en español: café, piñata, mañana, niño pequeño.', 'kind' => 'text'],
            ['name' => 'id_key', 'value' => '1003', 'kind' => 'keyword'],
            ['name' => 'category_id_key', 'value' => '12', 'kind' => 'keyword'],
            ['name' => 'status_id_attr', 'value' => 'draft', 'kind' => 'unindexed'],
        ]],
        ['docid' => 'B3', 'fields' => [
            ['name' => 'title', 'value' => 'Celebración final del proyecto 🎉', 'kind' => 'text'],
            ['name' => 'body', 'value' => 'Terminamos el proyecto con éxito 🚀 hoy mismo 🙌.', 'kind' => 'text'],
            ['name' => 'id_key', 'value' => '1004', 'kind' => 'keyword'],
            ['name' => 'category_id_key', 'value' => '13', 'kind' => 'keyword'],
            ['name' => 'status_id_attr', 'value' => 'published', 'kind' => 'unindexed'],
        ]],
        ['docid' => 'B4', 'fields' => [
            ['name' => 'body', 'value' => 'VPN client setup for network access, vpn again.', 'kind' => 'text'],
        ]],
        ['docid' => 'B5', 'fields' => [
            ['name' => 'title', 'value' => 'Escalated ticket handling procedure', 'kind' => 'text'],
            ['name' => 'body', 'value' => 'This ticket requires escalated handling procedures immediately.', 'kind' => 'text'],
            ['name' => 'tags_key', 'value' => 'High Priority Urgent', 'kind' => 'keyword'],
        ]],
        ['docid' => 'B6', 'fields' => [
            ['name' => 'title', 'value' => 'Internal process notes', 'kind' => 'text'],
            ['name' => 'body', 'value' => 'Standard operating procedure for internal review process.', 'kind' => 'text'],
            ['name' => 'notes_id_attr', 'value' => 'internal only - confidential, do not index this text', 'kind' => 'unindexed'],
        ]],
        ['docid' => 'B7', 'fields' => [
            ['name' => 'title', 'value' => 'TEST-CASE: Done!', 'kind' => 'text'],
            ['name' => 'body', 'value' => 'v2.0 build123 test-case done. WORK Work woRK done.', 'kind' => 'text'],
        ]],
        ['docid' => 'B8', 'fields' => [
            ['name' => 'body', 'value' => 'Please submit a vpn access request form today.', 'kind' => 'text'],
        ]],
        ['docid' => 'B9', 'fields' => [
            ['name' => 'id_key', 'value' => '0', 'kind' => 'keyword'],
            ['name' => 'category_id_key', 'value' => '999999999', 'kind' => 'keyword'],
            ['name' => 'tags_key', 'value' => 'Low', 'kind' => 'keyword'],
        ]],
        ['docid' => 'B10', 'fields' => [
            ['name' => 'body', 'value' => 'Café résumé naïve façade piñata café.', 'kind' => 'text'],
        ]],
        ['docid' => 'B11', 'fields' => [
            ['name' => 'title', 'value' => 'Second escalation case', 'kind' => 'text'],
            ['name' => 'body', 'value' => 'Another escalated case requiring urgent handling.', 'kind' => 'text'],
            ['name' => 'tags_key', 'value' => 'High Priority Urgent', 'kind' => 'keyword'],
        ]],
    ];
}

/**
 * Query battery over the fixed corpus. Each entry [name, field, text, kind ('term'|'phrase')].
 * Chosen to hit each edge case: multi-doc docFreq, unicode, case-folding, verbatim multi-word
 * keyword, numeric keyword, multi-token phrase.
 */
function queryBattery(): array {
    return [
        ['vpn-term',        'body',            'vpn',                  'term'],   // B0,B4,B8 -> df=3
        ['unicode-cafe',    'body',            'café',                 'term'],   // B2,B10   -> df=2
        ['case-fold-work',  'body',            'work',                 'term'],   // B7       -> df=1
        ['kw-verbatim-tags','tags_key',        'High Priority Urgent', 'term'],   // B5,B11   -> df=2 (verbatim, not tokenized)
        ['kw-numeric-id',   'id_key',          '1001',                 'term'],   // B0       -> df=1
        ['kw-cat-big',      'category_id_key', '999999999',            'term'],   // B9       -> df=1
        ['phrase-vpn-client','body',           'vpn client',           'phrase'], // B0,B4    -> df=2
        ['text-lower-escalated','body',        'escalated',            'term'],   // B5,B11   -> df=2
    ];
}

// builds a Zend_Search_Lucene_Document from a corpus doc (+ 'docid' marker).
function toZslDocument(array $doc) {
    $d = new Zend_Search_Lucene_Document();
    $d->addField(Zend_Search_Lucene_Field::Keyword('docid', $doc['docid']));
    foreach ($doc['fields'] as $f) {
        $d->addField(zslField($f['name'], $f['value'], $f['kind']));
    }
    return $d;
}

// serializes a corpus doc to the JSON SdSearch\Writer::add_document expects (+ 'docid' marker).
function toWriterJson(array $doc): string {
    $fields = [['name' => 'docid', 'value' => $doc['docid'], 'kind' => 'keyword']];
    foreach ($doc['fields'] as $f) {
        $fields[] = ['name' => $f['name'], 'value' => $f['value'], 'kind' => $f['kind']];
    }
    return json_encode(['fields' => $fields]);
}

// indexes the corpus with ZSL's writer in $dir: create + addDocument* + optimize().
function indexWithZsl(string $dir, array $corpus): void {
    $idx = Zend_Search_Lucene::create($dir);
    foreach ($corpus as $doc) {
        $idx->addDocument(toZslDocument($doc));
    }
    $idx->optimize();
    $idx = null;
}

// indexes the corpus with SdSearch\Writer (native FFI) in $dir: empty ZSL skeleton + add_document* + optimize().
function indexWithNative(string $dir, array $corpus, callable $cleanupLocks): int {
    $idx = Zend_Search_Lucene::create($dir); // empty skeleton: segments_1 + segments.gen
    $idx->commit();
    $idx = null;
    $cleanupLocks($dir);

    $w = new \SdSearch\Writer();
    $w->open($dir);
    foreach ($corpus as $doc) {
        $w->add_document(toWriterJson($doc));
    }
    $cnt = $w->optimize();
    $w = null;
    $cleanupLocks($dir);
    return $cnt;
}

// term-dict: field\x00text => docFreq, ignoring the 'docid' marker. ksort for stable order.
function termDict($idx): array {
    $out = [];
    foreach ($idx->terms() as $t) {
        if ($t->field === 'docid') { continue; }
        $out[$t->field . "\x00" . $t->text] = $idx->docFreq($t);
    }
    ksort($out);
    return $out;
}

// stored fields per doc, keyed by the 'docid' marker (excluded from the sub-arrays). Skips deletes.
function storedByDocid($idx): array {
    $out = [];
    $max = $idx->maxDoc();
    for ($i = 0; $i < $max; $i++) {
        if ($idx->isDeleted($i)) { continue; }
        $doc = $idx->getDocument($i);
        $docid = $doc->getFieldValue('docid');
        $fields = [];
        foreach ($doc->getFieldNames() as $fn) {
            if ($fn === 'docid') { continue; }
            $fld = $doc->getField($fn);
            if (!$fld->isStored) { continue; }
            $fields[$fn] = (string)$fld->value;
        }
        ksort($fields);
        $out[$docid] = $fields;
    }
    ksort($out);
    return $out;
}

// truncated symmetric-diff helper (DIFF detail), == diff_writer.php.
function symDiff(array $a, array $b): array {
    $onlyA = array_diff_key($a, $b);
    $onlyB = array_diff_key($b, $a);
    $dfDiff = [];
    foreach (array_intersect_key($a, $b) as $k => $v) {
        if ($v !== $b[$k]) { $dfDiff[$k] = "z=$v n={$b[$k]}"; }
    }
    return ['onlyZ' => array_keys($onlyA), 'onlyN' => array_keys($onlyB), 'dfDiff' => $dfDiff];
}

function truncList(array $items, int $head = 8, int $tail = 4): string {
    $n = count($items);
    if ($n <= $head + $tail) { return implode(' ', $items); }
    $mid = $n - $head - $tail;
    return implode(' ', array_slice($items, 0, $head)) . " ...($mid more)... " . implode(' ', array_slice($items, -$tail));
}

// builds a ZSL query. kind 'term' = a single term; 'phrase' = multi-word phrase.
function zslQuery(string $field, string $text, string $kind) {
    if ($kind === 'phrase') {
        $ph = new Zend_Search_Lucene_Search_Query_Phrase();
        foreach (preg_split('/\s+/', trim($text)) as $w) {
            $ph->addTerm(new Zend_Search_Lucene_Index_Term($w, $field));
        }
        return $ph;
    }
    return new Zend_Search_Lucene_Search_Query_Term(new Zend_Search_Lucene_Index_Term($text, $field));
}

// doc-set by docid in the order ZSL returns.
function searchDocids($idx, $query): array {
    return array_map(fn($h) => $idx->getDocument($h->id)->getFieldValue('docid'), $idx->find($query));
}

// ============================== main ==============================
$fails = 0;

$corpus = corpus();
printf("corpus: %d docs\n", count($corpus));

$dirZ = sys_get_temp_dir() . '/sdsearch_diffffi_z_' . getmypid();
$dirN = sys_get_temp_dir() . '/sdsearch_diffffi_n_' . getmypid();
foreach ([$dirZ, $dirN] as $d) {
    if (is_dir($d)) { array_map('unlink', glob("$d/*") ?: []); } else { mkdir($d, 0777, true); }
}

echo "== indexing ZSL ($dirZ) ==\n";
indexWithZsl($dirZ, $corpus);
$cleanupLocks($dirZ);

echo "== indexing NATIVE via SdSearch\\Writer FFI ($dirN) ==\n";
$nativeDocCount = indexWithNative($dirN, $corpus, $cleanupLocks);
printf("  native optimize() reported doc_count=%d (expected=%d)\n", $nativeDocCount, count($corpus));
if ($nativeDocCount !== count($corpus)) { $fails++; echo "  [doc_count native] DIFF\n"; } else { echo "  [doc_count native] OK\n"; }

// ZSL reads BOTH indexes from lock-free copies (isolates the writer; the reader is identical on both sides).
$roZ = $dirZ . '_ro';
$roN = $dirN . '_ro';
lockFreeCopy($dirZ, $roZ);
lockFreeCopy($dirN, $roN);
$idxZ = Zend_Search_Lucene::open($roZ);
$idxN = Zend_Search_Lucene::open($roN);

printf("numDocs: zsl=%d nat=%d\n", $idxZ->numDocs(), $idxN->numDocs());
if ($idxZ->numDocs() !== count($corpus) || $idxN->numDocs() !== count($corpus)) {
    $fails++;
    echo "[numDocs] DIFF (expected=" . count($corpus) . ")\n";
} else {
    echo "[numDocs] OK\n";
}

// ---- non-vacuous guard: the corpus must have produced real terms ----
$tdZ = termDict($idxZ);
$tdN = termDict($idxN);
if (count($tdZ) === 0) {
    $fails++;
    echo "[non-vacuous] HARNESS FAILURE: ZSL term-dict is empty (malformed corpus)\n";
} else {
    printf("[non-vacuous] ZSL term-dict has %d terms (>0) OK\n", count($tdZ));
}

// ---- term-dict (field,text,docFreq), post-optimize BOTH sides (methodological gotcha) ----
if ($tdZ === $tdN) {
    printf("[term-dict] OK (%d identical terms, post-optimize)\n", count($tdZ));
} else {
    $fails++;
    $d = symDiff($tdZ, $tdN);
    printf("[term-dict] DIFF: zsl-only=%d nat-only=%d docFreq-different=%d\n",
        count($d['onlyZ']), count($d['onlyN']), count($d['dfDiff']));
    if ($d['onlyZ']) { echo "  zsl-only: " . implode(' ', array_map(fn($k) => str_replace("\x00", ':', $k), array_slice($d['onlyZ'], 0, 20))) . "\n"; }
    if ($d['onlyN']) { echo "  nat-only: " . implode(' ', array_map(fn($k) => str_replace("\x00", ':', $k), array_slice($d['onlyN'], 0, 20))) . "\n"; }
    foreach (array_slice($d['dfDiff'], 0, 20, true) as $k => $v) { echo "  df≠ " . str_replace("\x00", ':', $k) . " ($v)\n"; }
}

// ---- stored fields per doc ----
$stZ = storedByDocid($idxZ);
$stN = storedByDocid($idxN);
if ($stZ === $stN) {
    printf("[stored] OK (%d docs, identical stored fields)\n", count($stZ));
} else {
    $fails++;
    echo "[stored] DIFF:\n";
    $allDocids = array_unique(array_merge(array_keys($stZ), array_keys($stN)));
    sort($allDocids);
    foreach ($allDocids as $docid) {
        $z = $stZ[$docid] ?? null;
        $n = $stN[$docid] ?? null;
        if ($z !== $n) {
            echo "  docid=$docid\n    zsl=" . json_encode($z) . "\n    nat=" . json_encode($n) . "\n";
        }
    }
}

// ---- doc-set of the query battery ----
$battery = queryBattery();
printf("%-24s %-6s %-6s %-8s\n", 'case', 'zsl', 'nat', 'docset');
foreach ($battery as [$name, $field, $text, $kind]) {
    $q = zslQuery($field, $text, $kind);
    $rz = searchDocids($idxZ, $q);
    $rn = searchDocids($idxN, $q);
    $setZ = $rz; sort($setZ);
    $setN = $rn; sort($setN);
    $docsetOk = ($setZ === $setN);
    printf("%-24s %-6d %-6d %-8s\n", $name, count($rz), count($rn), $docsetOk ? 'OK' : 'DIFF');
    if (count($rz) === 0) {
        // non-vacuous guard: if the ZSL ORACLE returns nothing, the query/corpus is badly designed.
        $fails++;
        echo "  [non-vacuous] HARNESS FAILURE: query '$name' matches nothing in the ZSL oracle\n";
    }
    if (!$docsetOk) {
        $fails++;
        echo "  zsl: [" . truncList($rz) . "]\n";
        echo "  nat: [" . truncList($rn) . "]\n";
    }
}

$idxZ = null; $idxN = null;
$cleanupLocks($roZ); $cleanupLocks($roN);
foreach ([$dirZ, $dirN, $roZ, $roN] as $d) { array_map('unlink', glob("$d/*") ?: []); @rmdir($d); }

echo $fails === 0
    ? "\nDIFF-FFI: PASS — SdSearch\\Writer (FFI) ≡ writer ZSL (term-dict + stored + doc-set, post-optimize)\n"
    : "\nDIFF-FFI: FAIL — $fails divergence" . ($fails === 1 ? '' : 's') . " (see DIFF/FAILURE above)\n";
exit($fails === 0 ? 0 : 1);
