<?php
/**
 * WRITER DIFFERENTIAL (Phase 1: ZSL reads both).
 *
 * Indexes the SAME corpus with (a) ZSL's PHP writer and (b) the native IndexWriter (the
 * stream_writer binary), and with ZSL as the SOLE READER of both, compares: term-dict
 * (field,text,docFreq) and doc-set per query. Isolates the writer: any difference is the
 * writer's, not the reader's. Operates on temp dirs; does NOT touch <source-index-dir>
 * (except to COPY for the extracted corpus). Does NOT use the .so extension.
 *
 * Usage: ZEND_LUCENE_PATH=/path/to/zsl/library php tools/diff_writer.php [source-index-dir] [N]
 *   default: sdsearch-core/tests/fixtures/zsl_index_kb (committed fixture) 200
 */
require __DIR__ . '/zsl_bootstrap.php';

$REPO          = dirname(__DIR__);
$EXTRACT_SRC   = $argv[1] ?? "$REPO/sdsearch-core/tests/fixtures/zsl_index_kb";
$EXTRACT_N     = (int)($argv[2] ?? 200);
$STREAM        = "$REPO/target/release/examples/stream_writer";
$CAP           = 4;   // small cap → multi-segment on the native side
$DIFFREAD      = "$REPO/target/release/examples/diff_read";
$hasDiffRead   = is_file($DIFFREAD);

if (!is_file($STREAM)) {
    fwrite(STDERR, "missing binary: $STREAM\n  build: cargo build -p sdsearch-core --release --example stream_writer\n");
    exit(2);
}
if (!$hasDiffRead) {
    fwrite(STDERR, "warning: missing $DIFFREAD → skipping the 2x2 triangulation (Phase 2)\n  build: cargo build -p sdsearch-core --release --example diff_read\n");
}

$cleanupLocks = function (string $dir) {
    foreach (array_merge(glob("$dir/*.sti") ?: [], glob("$dir/*lock*") ?: []) as $f) { @unlink($f); }
};

// maps a corpus kind to a Zend_Search_Lucene_Field.
function zslField(string $n, string $v, string $kind) {
    switch ($kind) {
        case 'text':      return Zend_Search_Lucene_Field::Text($n, $v);
        case 'keyword':   return Zend_Search_Lucene_Field::Keyword($n, $v);
        case 'unindexed': return Zend_Search_Lucene_Field::UnIndexed($n, $v);
        default: throw new InvalidArgumentException("unknown kind: $kind");
    }
}

// indexes the corpus with ZSL's writer in $dir (creates empty index + addDocument*).
function indexWithZsl(string $dir, array $corpus): void {
    $idx = Zend_Search_Lucene::create($dir);
    $idx->setMaxBufferedDocs(4); // several ZSL segments (layout irrelevant to the comparison)
    foreach ($corpus as $doc) {
        $d = new Zend_Search_Lucene_Document();
        $d->addField(Zend_Search_Lucene_Field::Keyword('docid', $doc['docid']));
        foreach ($doc['fields'] as $f) {
            $d->addField(zslField($f['name'], $f['value'], $f['kind']));
        }
        $idx->addDocument($d);
    }
    $idx->commit();
}

// serializes the corpus to the docs.json format stream_writer expects (+ docid keyword).
function nativeDocs(array $corpus): array {
    return array_map(function ($doc) {
        $fields = [['name' => 'docid', 'value' => $doc['docid'], 'kind' => 'keyword', 'stored' => true]];
        foreach ($doc['fields'] as $f) {
            $fields[] = ['name' => $f['name'], 'value' => $f['value'], 'kind' => $f['kind'], 'stored' => true];
        }
        return ['fields' => $fields];
    }, $corpus);
}

// indexes the corpus with the NATIVE writer in $dir: creates an empty ZSL base (IndexWriter needs a
// base generation), then runs stream_writer with a small cap.
function indexWithNative(string $dir, array $corpus, string $streamBin, int $cap, callable $cleanupLocks): void {
    $idx = Zend_Search_Lucene::create($dir); // empty base: writes segments.gen + segments_1
    $idx->commit();
    $idx = null;
    $cleanupLocks($dir);

    $docsJson = "$dir/_corpus.json";
    file_put_contents($docsJson, json_encode(['docs' => nativeDocs($corpus)]));
    $out = []; $rc = 0;
    exec(escapeshellarg($streamBin) . ' ' . escapeshellarg($dir) . ' ' . escapeshellarg($docsJson)
         . ' ' . escapeshellarg((string)$cap) . ' 2>&1', $out, $rc);
    @unlink($docsJson);
    $cleanupLocks($dir);
    if ($rc !== 0) { throw new RuntimeException("stream_writer failed (rc=$rc): " . implode(' ', $out)); }
}

/**
 * Adversarial corpus: each doc hits a byte-faithful writer edge case.
 * docid = stable keyword marker to compare per-doc across indexes.
 * Kind per field name is kept CONSISTENT across the whole corpus: 'body' always
 * text, 'tag'/'status' always keyword, 'note' always unindexed.
 */
function adversarialCorpus(): array {
    return [
        ['docid' => 'A0', 'fields' => [['name' => 'body', 'value' => 'work workflow working worked', 'kind' => 'text']]], // shared-prefix .tis
        ['docid' => 'A1', 'fields' => [['name' => 'body', 'value' => 'work', 'kind' => 'text']]],                          // docFreq('body','work')=2
        ['docid' => 'A2', 'fields' => [['name' => 'body', 'value' => 'café piñata mañana', 'kind' => 'text']]],            // modified-UTF-8
        ['docid' => 'A3', 'fields' => [
            ['name' => 'tag',  'value' => 'High Priority', 'kind' => 'keyword'],  // keyword: 1 token verbatim, pos 0
            ['name' => 'body', 'value' => 'High Priority', 'kind' => 'text'],     // text: tokenizes + lowercase
        ]],
        ['docid' => 'A4', 'fields' => [['name' => 'body', 'value' => str_repeat('lorem ipsum dolor ', 60), 'kind' => 'text']]], // norm: long field
        ['docid' => 'A5', 'fields' => [['name' => 'body', 'value' => 'lorem', 'kind' => 'text']]],                          // norm: short field, same term
        ['docid' => 'A6', 'fields' => [
            ['name' => 'note', 'value' => 'internal only', 'kind' => 'unindexed'], // unindexed: stored, no term
            ['name' => 'body', 'value' => 'vpn client setup', 'kind' => 'text'],
        ]],
        ['docid' => 'A7', 'fields' => [['name' => 'body', 'value' => 'workflow vpn café', 'kind' => 'text']]],              // term crossovers between docs
        ['docid' => 'A8', 'fields' => [['name' => 'body', 'value' => 'naïve résumé façade', 'kind' => 'text']]],            // more accented unicode
        ['docid' => 'A9', 'fields' => [['name' => 'body', 'value' => 'workflow', 'kind' => 'text']]],                       // docFreq('body','workflow')=3
        ['docid' => 'A10', 'fields' => [
            ['name' => 'tag',  'value' => 'Low Priority Ticket', 'kind' => 'keyword'], // multi-word keyword: 1 token verbatim with spaces
            ['name' => 'body', 'value' => 'ticket priority handling', 'kind' => 'text'],
        ]],
        ['docid' => 'A11', 'fields' => [
            ['name' => 'note', 'value' => 'do not index this text', 'kind' => 'unindexed'],
            ['name' => 'body', 'value' => 'printer setup network', 'kind' => 'text'],
        ]],
        ['docid' => 'A12', 'fields' => [['name' => 'body', 'value' => 'v2.0 build123 test-case done.', 'kind' => 'text']]], // numbers/punctuation in tokenization
        ['docid' => 'A13', 'fields' => [['name' => 'body', 'value' => str_repeat('alpha beta gamma ', 10), 'kind' => 'text']]], // norm: medium field (different bucket from A4/A5)
        ['docid' => 'A14', 'fields' => [['name' => 'body', 'value' => 'a i o u', 'kind' => 'text']]],                       // 1-letter tokens (possible stopword/min-len)
        ['docid' => 'A15', 'fields' => [['name' => 'body', 'value' => 'WORK Work woRK WORKFLOW', 'kind' => 'text']]],       // case-folding, more docFreq work/workflow
        ['docid' => 'A16', 'fields' => [['name' => 'body', 'value' => 'hello, world! test-case; done.', 'kind' => 'text']]], // heavy punctuation, token boundaries
        ['docid' => 'A17', 'fields' => [['name' => 'body', 'value' => 'café résumé naïve piñata', 'kind' => 'text']]],      // docFreq>1 over unicode terms
        ['docid' => 'A18', 'fields' => [['name' => 'body', 'value' => 'vpn setup client access', 'kind' => 'text']]],       // more docFreq('body','vpn')
        ['docid' => 'A19', 'fields' => [['name' => 'tag', 'value' => 'High Priority', 'kind' => 'keyword']]],               // docFreq('tag','High Priority')=2
        ['docid' => 'A20', 'fields' => [['name' => 'body', 'value' => 'lorem ipsum', 'kind' => 'text']]],                   // more docFreq('body','lorem') and 'ipsum'
        ['docid' => 'A21', 'fields' => [['name' => 'status', 'value' => 'Open', 'kind' => 'keyword']]],                     // new keyword field, single token
    ];
}

// term-dict: field\x00text => docFreq, ignoring the 'docid' marker. ksort for stable order.
function termDict($idx): array {
    $out = [];
    foreach ($idx->terms() as $t) {          // Zend_Search_Lucene::terms() = all terms
        if ($t->field === 'docid') { continue; }
        $out[$t->field . "\x00" . $t->text] = $idx->docFreq($t);
    }
    ksort($out);
    return $out;
}

// truncated symmetric-diff helper (for the DIFF detail).
function symDiff(array $a, array $b): array {
    $onlyA = array_diff_key($a, $b);
    $onlyB = array_diff_key($b, $a);
    // terms present in both but with different docFreq:
    $dfDiff = [];
    foreach (array_intersect_key($a, $b) as $k => $v) {
        if ($v !== $b[$k]) { $dfDiff[$k] = "z=$v n={$b[$k]}"; }
    }
    return ['onlyZ' => array_keys($onlyA), 'onlyN' => array_keys($onlyB), 'dfDiff' => $dfDiff];
}

// truncates long lists for readable output (head + tail). (== parity_adapter.php)
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

// runs the NATIVE reader over $dir; returns ['doc_sets'=>[name=>[docid,...]], 'term_dict'=>[...]].
function nativeRead(string $bin, string $dir, array $fields, array $queries): array {
    $spec = ['fields' => array_values($fields), 'queries' => $queries];
    $qJson = "$dir/_queries.json";
    file_put_contents($qJson, json_encode($spec));
    $out = []; $rc = 0;
    exec(escapeshellarg($bin) . ' ' . escapeshellarg($dir) . ' ' . escapeshellarg($qJson) . ' 2>&1', $out, $rc);
    @unlink($qJson);
    if ($rc !== 0) { throw new RuntimeException("diff_read failed (rc=$rc): " . implode(' ', $out)); }
    $parsed = json_decode(end($out), true);
    if (!is_array($parsed)) { throw new RuntimeException("diff_read: invalid JSON: " . implode(' ', $out)); }
    return $parsed;
}

// doc-set by docid in the order ZSL returns.
function searchDocids($idx, $query): array {
    return array_map(fn($h) => $idx->getDocument($h->id)->getFieldValue('docid'), $idx->find($query));
}
// docid => score, exactly as ZSL returned it (for orderStatus).
function scoresByDocid($idx, $query): array {
    $out = [];
    foreach ($idx->find($query) as $h) {
        $out[$idx->getDocument($h->id)->getFieldValue('docid')] = $h->score;
    }
    return $out;
}

/**
 * Order "OK where scores separate": consecutive blocks of tied score (on the Z side)
 * only require SET equality; outside the blocks, positional equality. (== parity_adapter.php)
 */
function orderStatus(array $rz, array $rn, array $scoresZ): string {
    if ($rz === $rn) { return 'OK'; }
    if (count($rz) !== count($rn)) { return 'diff'; }
    $n = count($rz); $i = 0;
    while ($i < $n) {
        $j = $i;
        while ($j + 1 < $n && ($scoresZ[$rz[$j + 1]] ?? null) === ($scoresZ[$rz[$i]] ?? null)) { $j++; }
        $setZ = array_slice($rz, $i, $j - $i + 1); sort($setZ);
        $setN = array_slice($rn, $i, $j - $i + 1); sort($setN);
        if ($setZ !== $setN) { return 'diff'; }
        $i = $j + 1;
    }
    return 'OK';
}

/**
 * Extracted corpus: reads stored fields from a COPY of the source index + each field's kind
 * (the Field's isIndexed/isTokenized). docid = value of the 'id' field if present, else a synthetic one.
 * Kind fidelity does NOT affect validity (both writers receive the same spec).
 */
function extractedCorpus(string $srcDir, int $n, callable $cleanupLocks): array {
    $src = $srcDir;
    if (!is_dir($src)) { throw new RuntimeException("index does not exist: $src"); }
    $tmp = sys_get_temp_dir() . "/sdsearch_diff_src_" . getmypid();
    if (is_dir($tmp)) { array_map('unlink', glob("$tmp/*") ?: []); } else { mkdir($tmp, 0777, true); }
    foreach (glob("$src/*") as $f) {
        $b = basename($f);
        if (str_contains($b, 'lock') || str_ends_with($b, '.sti')) { continue; }
        copy($f, "$tmp/$b");
    }
    try {
        $idx = Zend_Search_Lucene::open($tmp);
        $corpus = [];
        $max = min($n, $idx->maxDoc());
        for ($i = 0, $taken = 0; $i < $idx->maxDoc() && $taken < $max; $i++) {
            if ($idx->isDeleted($i)) { continue; }
            $doc = $idx->getDocument($i);
            $fields = [];
            $docid = "X$taken";
            foreach ($doc->getFieldNames() as $fn) {
                $fld = $doc->getField($fn);
                if ($fn === 'id') { $docid = (string)$fld->value; }
                if ($fn === 'docid') { continue; } // avoid collision with our marker
                if (!$fld->isStored) { continue; }  // we can only re-feed stored values
                $kind = !$fld->isIndexed ? 'unindexed' : ($fld->isTokenized ? 'text' : 'keyword');
                $fields[] = ['name' => $fn, 'value' => (string)$fld->value, 'kind' => $kind];
            }
            if ($fields) { $corpus[] = ['docid' => $docid, 'fields' => $fields]; $taken++; }
        }
        $idx = null;
        return $corpus;
    } finally {
        // always runs (happy path AND error path): the COPY in $tmp must never survive.
        $cleanupLocks($tmp);
        array_map('unlink', glob("$tmp/*") ?: []); @rmdir($tmp);
    }
}

/**
 * Battery for the adversarial corpus. Each entry [name, field, text, kind].
 * Chosen to test specific content: shared-prefix (work vs workflow differ in docFreq),
 * unicode, keyword-verbatim vs text-lowercased.
 */
function queryBattery(string $label): array {
    if ($label === 'adversarial') {
        return [
            ['work',        'body', 'work',          'term'],   // docFreq 2 (A0,A1)
            ['workflow',    'body', 'workflow',      'term'],   // docFreq 2 (A0,A7)
            ['unicode-cafe','body', 'café',          'term'],   // A2,A7
            ['kw-verbatim', 'tag',  'High Priority', 'term'],   // keyword: NOT lowercased → matches A3
            ['text-lower',  'body', 'high',          'term'],   // text: lowercased → matches A3
            ['phrase-vpn',  'body', 'vpn client',    'phrase'], // A6
        ];
    }
    // extracted corpus (KB): real fields 'title'/'content' (NOT 'description' — it does not exist
    // in this index). Terms confirmed against the real run (docFreq>0 on both sides):
    // title:vpn=2, content:windows=16, content:instalar=14.
    return [
        ['t-vpn',      'title',   'vpn',      'term'],
        ['c-windows',  'content', 'windows',  'term'],
        ['c-instalar', 'content', 'instalar', 'term'],
    ];
}

// --- main Phase 1: term-dict parity over the adversarial corpus ---
$fails = 0;
// SEPARATE counter for divergences in the cross(diag) column: it is NOT a real writer-differential
// (diff_read runs build_query all-fields; the ZSL battery uses field-scoped Term/Phrase) → reported
// as a diagnostic, never adds to $fails nor to the exit code.
$crossDiag = 0;

function runCorpus(string $label, array $corpus, string $stream, int $cap, callable $cleanupLocks, int &$fails, int &$crossDiag, ?string $diffReadBin = null): void {
    $tmpZ = sys_get_temp_dir() . "/sdsearch_diff_z_{$label}_" . getmypid();
    $tmpN = sys_get_temp_dir() . "/sdsearch_diff_n_{$label}_" . getmypid();
    foreach ([$tmpZ, $tmpN] as $d) {
        if (is_dir($d)) { array_map('unlink', glob("$d/*") ?: []); } else { mkdir($d, 0777, true); }
    }
    echo "== differential writer: corpus=$label (zsl=$tmpZ nat=$tmpN) ==\n";
    indexWithZsl($tmpZ, $corpus);
    $cleanupLocks($tmpZ);
    indexWithNative($tmpN, $corpus, $stream, $cap, $cleanupLocks);

    $idxZ = Zend_Search_Lucene::open($tmpZ);
    $idxN = Zend_Search_Lucene::open($tmpN);
    printf("  numDocs: zsl=%d nat=%d\n", $idxZ->numDocs(), $idxN->numDocs());

    // Phase 2 (triangulation) FIRST: needs the RAW native segments (the native reader
    // reading the multi-segment index Rust wrote). Runs before the optimize below.
    $battery = queryBattery($label);
    if ($diffReadBin && $battery) {
        // cells: ZSL-reads is already in $rz/$rn above; here we add NAT-reads over both indexes.
        $fields = [];
        foreach ($battery as [$name, $field, $text, $kind]) { $fields[$field] = true; }
        // the native reader does all-fields text search → we use each case's raw text.
        $queries = array_map(fn($c) => ['name' => $c[0], 'text' => $c[2]], $battery);
        $natOnZ = nativeRead($diffReadBin, $tmpZ, array_keys($fields), $queries)['doc_sets'];
        $natOnN = nativeRead($diffReadBin, $tmpN, array_keys($fields), $queries)['doc_sets'];

        printf("  -- 2x2 triangulation (by docid, doc-set) --\n");
        printf("  %-16s %-10s %-10s %-12s\n", 'case', 'Zr/Zw≡Nw', 'Nr/Zw≡Nw', 'cross(diag)');
        $crossDiagLocal = 0;
        foreach ($battery as [$name, $field, $text, $kind]) {
            // ZSL-reads both (recompute for clarity; cheap):
            $zOnZ = searchDocids($idxZ, zslQuery($field, $text, $kind));
            $zOnN = searchDocids($idxN, zslQuery($field, $text, $kind));
            $nOnZ = $natOnZ[$name] ?? [];
            $nOnN = $natOnN[$name] ?? [];
            $set = function (array $a) { sort($a); return $a; };
            $zslWriterDiff = ($set($zOnZ) === $set($zOnN));   // ZSL-reads: Zwriter ≡ Nwriter (REAL writer-differential)
            $natWriterDiff = ($set($nOnZ) === $set($nOnN));   // NAT-reads: Zwriter ≡ Nwriter (REAL writer-differential)
            // DIAGNOSTIC, not writer-differential: compares ZSL-reads vs NAT-reads over the native index,
            // but diff_read runs build_query all-fields while the battery uses field-scoped Term/Phrase
            // → a divergence here is a reader query-semantics artifact, not the writer's.
            $crossOk = ($set($zOnN) === $set($nOnN));
            $writerOk = $zslWriterDiff && $natWriterDiff;
            if (!$writerOk) { $fails++; }
            if (!$crossOk) { $crossDiag++; $crossDiagLocal++; }
            printf("  %-16s %-10s %-10s %-12s\n", $name,
                $zslWriterDiff ? 'OK' : 'DIFF', $natWriterDiff ? 'OK' : 'DIFF', $crossOk ? 'OK' : 'DIFF');
            if (!$writerOk) {
                echo "     Zr/Zw:[" . truncList($zOnZ) . "] Zr/Nw:[" . truncList($zOnN) . "]\n";
                echo "     Nr/Zw:[" . truncList($nOnZ) . "] Nr/Nw:[" . truncList($nOnN) . "]\n";
            } elseif (!$crossOk) {
                echo "     [cross(diag)] ZSL-read vs NAT-read (over the native index) differ — artifact of\n";
                echo "       reader query semantics (diff_read=all-fields vs battery=Term/Phrase\n";
                echo "       field-scoped), NOT a writer divergence.\n";
                echo "     Zr/Nw:[" . truncList($zOnN) . "] Nr/Nw:[" . truncList($nOnN) . "]\n";
            }
        }
        if ($crossDiagLocal > 0) {
            printf("  [cross(diag)] %d/%d cases differ (query-semantics artifact, does not count as a writer failure)\n",
                $crossDiagLocal, count($battery));
        }
    }

    // NORMALIZE SEGMENTATION before comparing CONTENT: docFreq/find over an unoptimized
    // multi-segment index is NOT segmentation-invariant in ZSL (ZSL itself over-counts
    // high-frequency terms until they are merged; the host application's indexing loop always
    // calls optimize() once it finishes indexing). Comparing writer equivalence requires merging
    // both to 1 segment first; otherwise different segmentations are compared (ZSL auto-merges,
    // the native side does not yet) and term-dict/query-set give false positives.
    $idxZ->optimize();
    $idxN->optimize();
    $idxZ = null; $idxN = null;
    $cleanupLocks($tmpZ); $cleanupLocks($tmpN);
    $idxZ = Zend_Search_Lucene::open($tmpZ);
    $idxN = Zend_Search_Lucene::open($tmpN);

    $tdZ = termDict($idxZ);
    $tdN = termDict($idxN);
    if ($tdZ === $tdN) {
        printf("  [term-dict] OK (%d identical terms, post-optimize)\n", count($tdZ));
    } else {
        $fails++;
        $d = symDiff($tdZ, $tdN);
        printf("  [term-dict] DIFF: zsl-only=%d nat-only=%d docFreq-different=%d\n",
            count($d['onlyZ']), count($d['onlyN']), count($d['dfDiff']));
        if ($d['onlyZ']) { echo "    zsl-only: " . implode(' ', array_map(fn($k) => str_replace("\x00", ':', $k), array_slice($d['onlyZ'], 0, 12))) . "\n"; }
        if ($d['onlyN']) { echo "    nat-only: " . implode(' ', array_map(fn($k) => str_replace("\x00", ':', $k), array_slice($d['onlyN'], 0, 12))) . "\n"; }
        foreach (array_slice($d['dfDiff'], 0, 12, true) as $k => $v) { echo "    df≠ " . str_replace("\x00", ':', $k) . " ($v)\n"; }
    }

    if ($battery) {
        printf("  %-16s %-6s %-6s %-8s %-8s\n", 'case', 'zsl', 'nat', 'docset', 'order');
        foreach ($battery as [$name, $field, $text, $kind]) {
            $qZ = zslQuery($field, $text, $kind);
            $rz = searchDocids($idxZ, $qZ);
            $rn = searchDocids($idxN, $qZ);
            $scoresZ = scoresByDocid($idxZ, $qZ);
            $setZ = $rz; sort($setZ);
            $setN = $rn; sort($setN);
            $docsetOk = ($setZ === $setN);
            $orderOk = $docsetOk ? (orderStatus($rz, $rn, $scoresZ) === 'OK') : false;
            printf("  %-16s %-6d %-6d %-8s %-8s\n", $name, count($rz), count($rn),
                $docsetOk ? 'OK' : 'DIFF', $orderOk ? 'OK' : 'diff');
            if (!$docsetOk) {
                $fails++;
                echo "     zsl: [" . truncList($rz) . "]\n";
                echo "     nat: [" . truncList($rn) . "]\n";
            }
        }
    }

    $idxZ = null; $idxN = null;
    $cleanupLocks($tmpZ); $cleanupLocks($tmpN);
    foreach ([$tmpZ, $tmpN] as $d) { array_map('unlink', glob("$d/*") ?: []); @rmdir($d); }
}

/**
 * DELETE differential: indexes the SAME corpus with both writers, deletes the
 * SAME doc by position (gid = global_doc_id, the 0-based numbering of
 * `IndexWriter::delete_document` / `Zend_Search_Lucene::delete()` over insertion order —
 * NOT the value of the stored 'docid' field) on BOTH sides (ZSL via `delete()`, native via
 * `stream_writer delete`), then a MANDATORY optimize() on both sides (docFreq is not
 * segmentation-invariant in an unoptimized ZSL index, so both must be merged before comparing)
 * and compares term-dict (field,text,docFreq) + doc-sets.
 *
 * Chosen victim: position 1 of the adversarial corpus = docid 'A1' (body="work"). It is discriminating
 * because 'body:work' has docFreq=3 in the full corpus (A0, A1 and A15 — "WORK Work woRK
 * WORKFLOW" case-folds to a single token 'work', docFreq counts DOCS not occurrences); after deleting
 * A1 it must drop to 2 (A0, A15) on BOTH writers. If the delete did not take effect in some writer,
 * that writer would still report docFreq=3 and 'A1' would still appear in the doc-set of the
 * 'work' query — both are compared explicitly below.
 */
function runDeleteCase(array $corpus, string $stream, int $cap, callable $cleanupLocks, int &$fails): void {
    $label = 'delete';
    $victimPos = 1; // docid 'A1' (body="work")
    $victimDocid = $corpus[$victimPos]['docid'];

    $tmpZ = sys_get_temp_dir() . "/sdsearch_diff_delz_" . getmypid();
    $tmpN = sys_get_temp_dir() . "/sdsearch_diff_deln_" . getmypid();
    foreach ([$tmpZ, $tmpN] as $d) {
        if (is_dir($d)) { array_map('unlink', glob("$d/*") ?: []); } else { mkdir($d, 0777, true); }
    }
    echo "== differential writer: corpus=$label (zsl=$tmpZ nat=$tmpN, victim=$victimDocid@gid$victimPos) ==\n";
    indexWithZsl($tmpZ, $corpus);
    $cleanupLocks($tmpZ);
    indexWithNative($tmpN, $corpus, $stream, $cap, $cleanupLocks);

    $numBefore = count($corpus);

    // ---- delete the SAME doc (by gid) on BOTH sides ----
    $idxZ = Zend_Search_Lucene::open($tmpZ);
    $idxZ->delete($victimPos);
    $idxZ->commit();
    $idxZ = null;
    $cleanupLocks($tmpZ);

    $out = []; $rc = 0;
    exec(escapeshellarg($stream) . ' delete ' . escapeshellarg($tmpN) . ' ' . escapeshellarg((string)$victimPos)
         . ' 2>&1', $out, $rc);
    if ($rc !== 0) { throw new RuntimeException("stream_writer delete failed (rc=$rc): " . implode(' ', $out)); }
    $cleanupLocks($tmpN);

    $idxZ = Zend_Search_Lucene::open($tmpZ);
    $idxN = Zend_Search_Lucene::open($tmpN);
    printf("  numDocs after delete: zsl=%d nat=%d (expected=%d)\n", $idxZ->numDocs(), $idxN->numDocs(), $numBefore - 1);
    $numDocsOk = ($idxZ->numDocs() === $numBefore - 1) && ($idxN->numDocs() === $numBefore - 1);
    if (!$numDocsOk) { $fails++; echo "  [numDocs] DIFF\n"; } else { echo "  [numDocs] OK\n"; }
    $idxZ = null; $idxN = null;
    $cleanupLocks($tmpZ); $cleanupLocks($tmpN);

    // ---- optimize BOTH sides (MANDATORY, optimize-before-compare rule) ----
    $idxZ = Zend_Search_Lucene::open($tmpZ);
    $idxN = Zend_Search_Lucene::open($tmpN);
    $idxZ->optimize();
    $idxN->optimize();
    $idxZ = null; $idxN = null;
    $cleanupLocks($tmpZ); $cleanupLocks($tmpN);
    $idxZ = Zend_Search_Lucene::open($tmpZ);
    $idxN = Zend_Search_Lucene::open($tmpN);

    $numDocsOkPostOpt = ($idxZ->numDocs() === $numBefore - 1) && ($idxN->numDocs() === $numBefore - 1);
    if (!$numDocsOkPostOpt) { $fails++; echo "  [numDocs post-optimize] DIFF\n"; } else { echo "  [numDocs post-optimize] OK\n"; }

    // ---- term-dict (field,text,docFreq), post-optimize ----
    $tdZ = termDict($idxZ);
    $tdN = termDict($idxN);
    if ($tdZ === $tdN) {
        printf("  [term-dict] OK (%d identical terms, post-optimize)\n", count($tdZ));
    } else {
        $fails++;
        $d = symDiff($tdZ, $tdN);
        printf("  [term-dict] DIFF: zsl-only=%d nat-only=%d docFreq-different=%d\n",
            count($d['onlyZ']), count($d['onlyN']), count($d['dfDiff']));
        if ($d['onlyZ']) { echo "    zsl-only: " . implode(' ', array_map(fn($k) => str_replace("\x00", ':', $k), array_slice($d['onlyZ'], 0, 12))) . "\n"; }
        if ($d['onlyN']) { echo "    nat-only: " . implode(' ', array_map(fn($k) => str_replace("\x00", ':', $k), array_slice($d['onlyN'], 0, 12))) . "\n"; }
        foreach (array_slice($d['dfDiff'], 0, 12, true) as $k => $v) { echo "    df≠ " . str_replace("\x00", ':', $k) . " ($v)\n"; }
    }
    // explicit discriminator: docFreq('body','work') must have dropped from 3 (A0,A1,A15) to 2
    // (A0,A15) on BOTH sides after deleting A1.
    $workDf = $tdZ['body' . "\x00" . 'work'] ?? null;
    $workDfOk = ($workDf === 2) && (($tdN['body' . "\x00" . 'work'] ?? null) === 2);
    if (!$workDfOk) { $fails++; }
    printf("  [docFreq(body,work) post-delete] %s (zsl=%s nat=%s want=2)\n",
        $workDfOk ? 'OK' : 'DIFF', json_encode($workDf), json_encode($tdN['body' . "\x00" . 'work'] ?? null));

    // ---- doc-set: the victim must be ABSENT from both for a query that used to return it ----
    $q = zslQuery('body', 'work', 'term');
    $rz = searchDocids($idxZ, $q);
    $rn = searchDocids($idxN, $q);
    $setZ = $rz; sort($setZ);
    $setN = $rn; sort($setN);
    $docsetOk = ($setZ === $setN) && !in_array($victimDocid, $rz, true) && !in_array($victimDocid, $rn, true);
    printf("  [doc-set 'work' without %s] %s zsl=[%s] nat=[%s]\n", $victimDocid,
        $docsetOk ? 'OK' : 'DIFF', implode(' ', $rz), implode(' ', $rn));
    if (!$docsetOk) { $fails++; }

    $idxZ = null; $idxN = null;
    $cleanupLocks($tmpZ); $cleanupLocks($tmpN);
    foreach ([$tmpZ, $tmpN] as $d) { array_map('unlink', glob("$d/*") ?: []); @rmdir($d); }
}

/**
 * OPTIMIZE differential: indexes the SAME multi-segment corpus with both writers,
 * deletes the SAME doc by gid on both, runs the NATIVE optimize() (native merge) vs ZSL's optimize(),
 * and compares term-dict (field,text,docFreq) + doc-sets. Proves the native merge ≡ ZSL's
 * by CONTENT, including doc-id renumbering (the deleted victim absent + docFreq
 * collapsed). The native side does NOT use ZSL's optimize(): it uses `stream_writer optimize`.
 */
function runOptimizeCase(array $corpus, string $stream, int $cap, callable $cleanupLocks, int &$fails): void {
    $victimPos = 1; // docid 'A1' (body="work") — same discriminator as runDeleteCase
    $victimDocid = $corpus[$victimPos]['docid'];

    $tmpZ = sys_get_temp_dir() . "/sdsearch_diff_optz_" . getmypid();
    $tmpN = sys_get_temp_dir() . "/sdsearch_diff_optn_" . getmypid();
    foreach ([$tmpZ, $tmpN] as $d) {
        if (is_dir($d)) { array_map('unlink', glob("$d/*") ?: []); } else { mkdir($d, 0777, true); }
    }
    echo "== differential OPTIMIZE native vs ZSL: (zsl=$tmpZ nat=$tmpN, victim=$victimDocid@gid$victimPos) ==\n";
    indexWithZsl($tmpZ, $corpus);
    $cleanupLocks($tmpZ);
    indexWithNative($tmpN, $corpus, $stream, $cap, $cleanupLocks);

    $numBefore = count($corpus);

    // delete the same doc (gid) on both sides
    $idxZ = Zend_Search_Lucene::open($tmpZ);
    $idxZ->delete($victimPos);
    $idxZ->commit();
    $idxZ = null;
    $cleanupLocks($tmpZ);
    $out = []; $rc = 0;
    exec(escapeshellarg($stream) . ' delete ' . escapeshellarg($tmpN) . ' ' . escapeshellarg((string)$victimPos) . ' 2>&1', $out, $rc);
    if ($rc !== 0) { throw new RuntimeException("stream_writer delete failed (rc=$rc): " . implode(' ', $out)); }
    $cleanupLocks($tmpN);

    // OPTIMIZE: ZSL with optimize(), NATIVE with `stream_writer optimize` (native merge, NOT ZSL)
    $idxZ = Zend_Search_Lucene::open($tmpZ);
    $idxZ->optimize();
    $idxZ = null;
    $cleanupLocks($tmpZ);
    $out = []; $rc = 0;
    exec(escapeshellarg($stream) . ' optimize ' . escapeshellarg($tmpN) . ' 2>&1', $out, $rc);
    if ($rc !== 0) { throw new RuntimeException("stream_writer optimize failed (rc=$rc): " . implode(' ', $out)); }
    $cleanupLocks($tmpN);

    $idxZ = Zend_Search_Lucene::open($tmpZ);
    $idxN = Zend_Search_Lucene::open($tmpN);
    printf("  numDocs post-optimize: zsl=%d nat=%d (expected=%d)\n", $idxZ->numDocs(), $idxN->numDocs(), $numBefore - 1);
    if ($idxZ->numDocs() !== $numBefore - 1 || $idxN->numDocs() !== $numBefore - 1) {
        $fails++; echo "  [numDocs post-optimize] DIFF\n";
    } else { echo "  [numDocs post-optimize] OK\n"; }

    // term-dict (field,text,docFreq): the native merge ≡ ZSL's
    $tdZ = termDict($idxZ);
    $tdN = termDict($idxN);
    if ($tdZ === $tdN) {
        printf("  [term-dict] OK (%d identical terms, native merge ≡ ZSL)\n", count($tdZ));
    } else {
        $fails++;
        $d = symDiff($tdZ, $tdN);
        printf("  [term-dict] DIFF: zsl-only=%d nat-only=%d docFreq-different=%d\n",
            count($d['onlyZ']), count($d['onlyN']), count($d['dfDiff']));
        if ($d['onlyZ']) { echo "    zsl-only: " . implode(' ', array_map(fn($k) => str_replace("\x00", ':', $k), array_slice($d['onlyZ'], 0, 12))) . "\n"; }
        if ($d['onlyN']) { echo "    nat-only: " . implode(' ', array_map(fn($k) => str_replace("\x00", ':', $k), array_slice($d['onlyN'], 0, 12))) . "\n"; }
        foreach (array_slice($d['dfDiff'], 0, 12, true) as $k => $v) { echo "    df≠ " . str_replace("\x00", ':', $k) . " ($v)\n"; }
    }
    // discriminator: docFreq(body,work) 3 -> 2 on both; victim absent from the doc-set
    $workDfOk = (($tdZ['body' . "\x00" . 'work'] ?? null) === 2) && (($tdN['body' . "\x00" . 'work'] ?? null) === 2);
    if (!$workDfOk) { $fails++; }
    printf("  [docFreq(body,work) post-optimize] %s\n", $workDfOk ? 'OK' : 'DIFF');

    $q = zslQuery('body', 'work', 'term');
    $rz = searchDocids($idxZ, $q); $rn = searchDocids($idxN, $q);
    $setZ = $rz; sort($setZ); $setN = $rn; sort($setN);
    $docsetOk = ($setZ === $setN) && !in_array($victimDocid, $rz, true) && !in_array($victimDocid, $rn, true);
    printf("  [doc-set 'work' without %s] %s zsl=[%s] nat=[%s]\n", $victimDocid,
        $docsetOk ? 'OK' : 'DIFF', implode(' ', $rz), implode(' ', $rn));
    if (!$docsetOk) { $fails++; }

    $idxZ = null; $idxN = null;
    $cleanupLocks($tmpZ); $cleanupLocks($tmpN);
    foreach ([$tmpZ, $tmpN] as $d) { array_map('unlink', glob("$d/*") ?: []); @rmdir($d); }
}

runCorpus('adversarial', adversarialCorpus(), $STREAM, $CAP, $cleanupLocks, $fails, $crossDiag, $hasDiffRead ? $DIFFREAD : null);

$extracted = extractedCorpus($EXTRACT_SRC, $EXTRACT_N, $cleanupLocks);
printf("extracted corpus '%s': %d docs\n", $EXTRACT_SRC, count($extracted));
runCorpus('extracted', $extracted, $STREAM, $CAP, $cleanupLocks, $fails, $crossDiag, $hasDiffRead ? $DIFFREAD : null);

runDeleteCase(adversarialCorpus(), $STREAM, $CAP, $cleanupLocks, $fails);

runOptimizeCase(adversarialCorpus(), $STREAM, $CAP, $cleanupLocks, $fails);

// TRIANGULATION status = ONLY the writer-differential columns (Zr/Zw≡Nw, Nr/Zw≡Nw). cross(diag)
// is reported separately and NEVER fails the exit code: it is a reader query-semantics artifact
// (diff_read=all-fields vs battery=field-scoped Term/Phrase), not a writer divergence.
echo $fails === 0
    ? "\nDIFFERENTIAL: PASS — native writer ≡ ZSL writer" . ($hasDiffRead ? " + TRIANGULATION OK (writer-differential columns; cross(diag) reported separately)" : " (Phase 1; build diff_read for the 2x2 matrix)") . "\n"
    : "\nDIFFERENTIAL: FAIL — $fails writer divergence" . ($fails === 1 ? '' : 's') . " (see DIFF above)\n";
if ($hasDiffRead && $crossDiag > 0) {
    echo "TRIANGULATION cross(diag): $crossDiag diagnostic divergence" . ($crossDiag === 1 ? '' : 's')
        . " (ZSL-read vs NAT-read over the native index; query-semantics artifact, NOT writer) — do not affect the exit code.\n";
}
exit($fails === 0 ? 0 : 1);
