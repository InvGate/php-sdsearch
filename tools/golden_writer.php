<?php
/**
 * GOLDEN INTERCHANGE GATE.
 *
 * Tests the key requirement: the native writer appends ONE byte-faithful segment to a real
 * ZSL index and ZSL can (1) RE-READ it and (2) CONTINUE indexing on top (optimize/merge).
 *
 * Flow (operates on a COPY; does NOT touch <source-index-dir>):
 *   1. copy <source-index-dir> to a tmp; open with ZSL, measure numDocs + a witness term.
 *   2. run the NATIVE append (target/release/examples/append_writer) over the tmp.
 *   3. ZSL reopens the shared dir -> assert: numDocs+N, new term found N times,
 *      new doc's stored correct, old content intact.
 *   4. ZSL optimize() (merge over the native segment) -> re-assert everything.
 *
 * Usage: ZEND_LUCENE_PATH=/path/to/zsl/library php tools/golden_writer.php [source-index-dir]
 *   default source-index-dir: sdsearch-core/tests/fixtures/zsl_index_kb (committed fixture)
 */
require __DIR__ . '/zsl_bootstrap.php';

$REPO  = dirname(__DIR__);
$SRC   = $argv[1] ?? "$REPO/sdsearch-core/tests/fixtures/zsl_index_kb";
$BIN   = "$REPO/target/release/examples/append_writer";
$STREAM = "$REPO/target/release/examples/stream_writer";
$DIFFREAD = "$REPO/target/release/examples/diff_read";
$N     = 3;               // docs to append
$MARK  = 'zqxwriter';     // unique term that does not exist in the index

foreach ([$BIN => 'append_writer', $STREAM => 'stream_writer', $DIFFREAD => 'diff_read'] as $bin => $ex) {
    if (!is_file($bin)) {
        fwrite(STDERR, "missing native binary: $bin\n  build: cargo build -p sdsearch-core --release --example $ex\n");
        exit(2);
    }
}

// ---- helpers ----
$fails = 0;
$check = function (string $what, $got, $want) use (&$fails) {
    $ok = $got === $want;
    printf("  [%s] %-46s got=%s want=%s\n", $ok ? 'OK ' : 'DIFF', $what,
           json_encode($got), json_encode($want));
    if (!$ok) { $fails++; }
};
$cleanupLocks = function (string $dir) {
    foreach (array_merge(glob("$dir/*.sti") ?: [], glob("$dir/*lock*") ?: []) as $f) { @unlink($f); }
};
$termCount = function ($idx, string $field, string $text): int {
    $q = new Zend_Search_Lucene_Search_Query_Term(new Zend_Search_Lucene_Index_Term($text, $field));
    return count($idx->find($q));
};

// ---- 1) copy source index to tmp ----
$src = $SRC;
if (!is_dir($src)) { fwrite(STDERR, "index does not exist: $src\n"); exit(2); }
$tmp = sys_get_temp_dir() . "/sdsearch_golden_" . getmypid();
if (is_dir($tmp)) { array_map('unlink', glob("$tmp/*") ?: []); } else { mkdir($tmp, 0777, true); }
foreach (glob("$src/*") as $f) {
    $b = basename($f);
    if (str_contains($b, 'lock') || str_ends_with($b, '.sti')) { continue; }
    copy($f, "$tmp/$b");
}

echo "== golden interchange: $src (copy in $tmp) ==\n";

// baseline: numDocs + a witness field/term (the first term in the stream)
$idx = Zend_Search_Lucene::open($tmp);
$before = $idx->numDocs();
$idx->resetTermsStream();
$probe = $idx->currentTerm();            // first existing term
$probeField = $probe->field;
$probeText  = $probe->text;
$probeBefore = $termCount($idx, $probeField, $probeText);
printf("  baseline: numDocs=%d, witness=%s:%s (docFreq=%d)\n", $before, $probeField, $probeText, $probeBefore);
$idx = null;
$cleanupLocks($tmp);

// ---- 2) batch to append: unique term MARK in the witness field + a stored keyword ----
$docs = ['docs' => []];
for ($i = 0; $i < $N; $i++) {
    $docs['docs'][] = ['fields' => [
        ['name' => $probeField, 'value' => "$MARK unique$i", 'kind' => 'text', 'stored' => true],
        ['name' => 'gid',       'value' => "G$i",            'kind' => 'keyword', 'stored' => true],
    ]];
}
$docsJson = "$tmp/_batch.json";
file_put_contents($docsJson, json_encode($docs));

$out = [];
$rc = 0;
exec(escapeshellarg($BIN) . ' ' . escapeshellarg($tmp) . ' ' . escapeshellarg($docsJson) . ' 2>&1', $out, $rc);
echo "  native append: rc=$rc " . implode(' ', $out) . "\n";
if ($rc !== 0) { fwrite(STDERR, "native append failed\n"); exit(1); }
@unlink($docsJson);
$cleanupLocks($tmp);

// ---- 3) ZSL RE-READS the native segment ----
echo "-- phase A: ZSL re-reads the native segment --\n";
$idx = Zend_Search_Lucene::open($tmp);
$check('numDocs += N', $idx->numDocs(), $before + $N);
$check("new term '$MARK' found N times", $termCount($idx, $probeField, $MARK), $N);
$check('old content intact (witness)', $termCount($idx, $probeField, $probeText), $probeBefore);

// stored of the new doc(s): the value we wrote must come back verbatim
$q = new Zend_Search_Lucene_Search_Query_Term(new Zend_Search_Lucene_Index_Term($MARK, $probeField));
$hits = $idx->find($q);
$storedOk = count($hits) === $N;
$seen = [];
foreach ($hits as $h) {
    $d = $idx->getDocument($h->id);
    $seen[$d->getFieldValue($probeField)] = ($d->getFieldValue('gid'));
}
for ($i = 0; $i < $N; $i++) {
    $storedOk = $storedOk && (($seen["$MARK unique$i"] ?? null) === "G$i");
}
$check('stored (text + keyword) round-trip', $storedOk, true);
$idx = null;
$cleanupLocks($tmp);

// ---- 4) ZSL CONTINUES indexing: optimize() merges the native segment ----
echo "-- phase B: ZSL optimize() over the native segment (merge/handoff) --\n";
$idx = Zend_Search_Lucene::open($tmp);
$idx->optimize();
$idx = null;
$cleanupLocks($tmp);

$idx = Zend_Search_Lucene::open($tmp);
$check('post-optimize numDocs', $idx->numDocs(), $before + $N);
$check("post-optimize new term '$MARK'", $termCount($idx, $probeField, $MARK), $N);
$check('post-optimize old content', $termCount($idx, $probeField, $probeText), $probeBefore);
$idx = null;

// ---- 5) phase C: native STREAMING build → multiple segments in one commit ----
echo "-- phase C: streaming writer (small cap → multi-segment) + ZSL re-reads + optimize --\n";
$tmp2 = sys_get_temp_dir() . "/sdsearch_golden_stream_" . getmypid();
if (is_dir($tmp2)) { array_map('unlink', glob("$tmp2/*") ?: []); } else { mkdir($tmp2, 0777, true); }
foreach (glob("$src/*") as $f) {
    $b = basename($f);
    if (str_contains($b, 'lock') || str_ends_with($b, '.sti')) { continue; }
    copy($f, "$tmp2/$b");
}

$NS = 5;              // docs to stream
$CAP = 2;            // small buffer → ceil(5/2)=3 segments in one commit
$MARK2 = 'zqxstream';
$docs2 = ['docs' => []];
for ($i = 0; $i < $NS; $i++) {
    $docs2['docs'][] = ['fields' => [
        ['name' => $probeField, 'value' => "$MARK2 s$i", 'kind' => 'text', 'stored' => true],
    ]];
}
$docs2Json = "$tmp2/_stream.json";
file_put_contents($docs2Json, json_encode($docs2));

$out2 = []; $rc2 = 0;
exec(escapeshellarg($STREAM) . ' ' . escapeshellarg($tmp2) . ' ' . escapeshellarg($docs2Json)
     . ' ' . escapeshellarg((string)$CAP) . ' 2>&1', $out2, $rc2);
echo "  native stream: rc=$rc2 " . implode(' ', $out2) . "\n";
if ($rc2 !== 0) { fwrite(STDERR, "native stream failed\n"); exit(1); }
$rep2 = json_decode(end($out2), true);
$check('stream wrote multiple segments', ($rep2['segments'] ?? 0) >= 2, true);
@unlink($docs2Json);
$cleanupLocks($tmp2);

$idx = Zend_Search_Lucene::open($tmp2);
$check('stream: numDocs += NS', $idx->numDocs(), $before + $NS);
$check("stream: term '$MARK2' found NS times", $termCount($idx, $probeField, $MARK2), $NS);
$idx->optimize(); // merge of the N native segments
$idx = null;
$cleanupLocks($tmp2);

$idx = Zend_Search_Lucene::open($tmp2);
$check('stream post-optimize numDocs', $idx->numDocs(), $before + $NS);
$check("stream post-optimize term '$MARK2'", $termCount($idx, $probeField, $MARK2), $NS);
$idx = null;

array_map('unlink', glob("$tmp2/*") ?: []); @rmdir($tmp2);

// ---- 6) phase D: native delete → ZSL RE-READS the native .del ----
// gid = global_doc_id from IndexWriter::delete_document, the SAME 0-based numbering as
// Zend_Search_Lucene::delete($id) (position over Σ maxDoc of the base segments, in
// segments_N order) — NOT the value of a stored field. We delete doc gid=0 (the first doc of
// the KB index, 'id_key'=2, title "Problems with file permissions in MySQL"); it is discriminating:
// - before the delete, a find() by Term(id_key, "2") returns it (id_key is unique per doc);
// - if the native .del were not materialized or ZSL did not re-read it, numDocs() would stay at 63 and
//   that find() would keep returning the doc.
echo "-- phase D: native delete (gid=0) + ZSL re-reads .del + optimize --\n";
$tmp3 = sys_get_temp_dir() . "/sdsearch_golden_delete_" . getmypid();
if (is_dir($tmp3)) { array_map('unlink', glob("$tmp3/*") ?: []); } else { mkdir($tmp3, 0777, true); }
foreach (glob("$src/*") as $f) {
    $b = basename($f);
    if (str_contains($b, 'lock') || str_ends_with($b, '.sti')) { continue; }
    copy($f, "$tmp3/$b");
}

$victimGid = 0;
$idx = Zend_Search_Lucene::open($tmp3);
$before3 = $idx->numDocs();
$victimQuery = new Zend_Search_Lucene_Search_Query_Term(
    new Zend_Search_Lucene_Index_Term($idx->getDocument($victimGid)->getFieldValue('id_key'), 'id_key')
);
$hitsBefore = $idx->find($victimQuery);
$check('phase D baseline: victim findable before deleting', count($hitsBefore) >= 1, true);
$idx = null;
$cleanupLocks($tmp3);

$out3 = []; $rc3 = 0;
exec(escapeshellarg($STREAM) . ' delete ' . escapeshellarg($tmp3) . ' ' . escapeshellarg((string)$victimGid)
     . ' 2>&1', $out3, $rc3);
echo "  native delete: rc=$rc3 " . implode(' ', $out3) . "\n";
if ($rc3 !== 0) { fwrite(STDERR, "native delete failed\n"); exit(1); }
$cleanupLocks($tmp3);

$idx = Zend_Search_Lucene::open($tmp3);
$check('phase D: numDocs after native delete', $idx->numDocs(), $before3 - 1);
$check('phase D: victim absent from find after delete', count($idx->find($victimQuery)), 0);
$idx = null;
$cleanupLocks($tmp3);

// ZSL optimize() collapses the native .del when merging -> re-assert.
$idx = Zend_Search_Lucene::open($tmp3);
$idx->optimize();
$idx = null;
$cleanupLocks($tmp3);

$idx = Zend_Search_Lucene::open($tmp3);
$check('phase D post-optimize: numDocs', $idx->numDocs(), $before3 - 1);
$check('phase D post-optimize: victim still absent', count($idx->find($victimQuery)), 0);
$idx = null;

array_map('unlink', glob("$tmp3/*") ?: []); @rmdir($tmp3);

// ---- 7) phase E: multi-segment + NATIVE delete → NATIVE optimize() → ZSL re-reads ----
// Tests the native merge: we write several segments with the streaming writer, delete a doc,
// run the NATIVE optimize() (collapses everything to 1 compacted segment) and ZSL must RE-READ it
// (correct numDocs, new term found, victim absent) and be able to optimize() on top (no-op).
echo "-- phase E: multi-seg + delete + NATIVE optimize() + ZSL re-reads --\n";
$tmp4 = sys_get_temp_dir() . "/sdsearch_golden_optimize_" . getmypid();
if (is_dir($tmp4)) { array_map('unlink', glob("$tmp4/*") ?: []); } else { mkdir($tmp4, 0777, true); }
foreach (glob("$src/*") as $f) {
    $b = basename($f);
    if (str_contains($b, 'lock') || str_ends_with($b, '.sti')) { continue; }
    copy($f, "$tmp4/$b");
}

$NE = 5; $CAPE = 2; $MARKE = 'zqxoptimize';
$docsE = ['docs' => []];
for ($i = 0; $i < $NE; $i++) {
    $docsE['docs'][] = ['fields' => [
        ['name' => $probeField, 'value' => "$MARKE e$i", 'kind' => 'text', 'stored' => true],
    ]];
}
$docsEJson = "$tmp4/_optimize.json";
file_put_contents($docsEJson, json_encode($docsE));
$outE = []; $rcE = 0;
exec(escapeshellarg($STREAM) . ' ' . escapeshellarg($tmp4) . ' ' . escapeshellarg($docsEJson)
     . ' ' . escapeshellarg((string)$CAPE) . ' 2>&1', $outE, $rcE);
echo "  native stream: rc=$rcE " . implode(' ', $outE) . "\n";
if ($rcE !== 0) { fwrite(STDERR, "native stream (phase E) failed\n"); exit(1); }
$repE = json_decode(end($outE), true);
$check('phase E: stream multi-segment', ($repE['segments'] ?? 0) >= 2, true);
@unlink($docsEJson);
$cleanupLocks($tmp4);

// delete doc gid=0 (KB base) with the native side
$outD = []; $rcD = 0;
exec(escapeshellarg($STREAM) . ' delete ' . escapeshellarg($tmp4) . ' 0 2>&1', $outD, $rcD);
echo "  native delete: rc=$rcD " . implode(' ', $outD) . "\n";
if ($rcD !== 0) { fwrite(STDERR, "native delete (phase E) failed\n"); exit(1); }
$cleanupLocks($tmp4);

// NATIVE OPTIMIZE: collapses base + 5 new - 1 deleted, multi-segment, into 1 segment
$outO = []; $rcO = 0;
exec(escapeshellarg($STREAM) . ' optimize ' . escapeshellarg($tmp4) . ' 2>&1', $outO, $rcO);
echo "  native optimize: rc=$rcO " . implode(' ', $outO) . "\n";
if ($rcO !== 0) { fwrite(STDERR, "native optimize (phase E) failed\n"); exit(1); }
$repO = json_decode(end($outO), true);
$check('phase E: optimize doc_count', $repO['doc_count'] ?? -1, $before + $NE - 1);
$cleanupLocks($tmp4);

// ZSL RE-READS the segment MERGED by the native side
$idx = Zend_Search_Lucene::open($tmp4);
$check('phase E: ZSL re-reads numDocs post native optimize', $idx->numDocs(), $before + $NE - 1);
$check("phase E: term '$MARKE' found NE times", $termCount($idx, $probeField, $MARKE), $NE);
$idx->optimize(); // ZSL optimize on top of the native merged segment (must be a safe no-op)
$idx = null;
$cleanupLocks($tmp4);

$idx = Zend_Search_Lucene::open($tmp4);
$check('phase E: post ZSL-optimize numDocs', $idx->numDocs(), $before + $NE - 1);
$check("phase E: post ZSL-optimize term '$MARKE'", $termCount($idx, $probeField, $MARKE), $NE);
$idx = null;

array_map('unlink', glob("$tmp4/*") ?: []); @rmdir($tmp4);

// ---- 8) FLIP phase: explicit bidirectional interchange, same dir, NO reindex ----
// (1) native writes witness 'zqflip' -> flip (closes everything, nobody holds the index open) ->
//     ZSL opens and reads without reindexing (numDocs+1, term present).
// (2) ZSL writes witness 'zqflip2' -> flip -> native (diff_read) opens and reads BOTH witnesses.
// Non-vacuous: each witness is verified ABSENT before its write and PRESENT (==1) after,
// via $check (adds to $fails). If the interchange did not cross engines, one of these would diverge.
echo "-- phase FLIP: bidirectional interchange native<->ZSL, same dir, no reindex (B1) --\n";
$tmp5 = sys_get_temp_dir() . "/sdsearch_golden_flip_" . getmypid();
if (is_dir($tmp5)) { array_map('unlink', glob("$tmp5/*") ?: []); } else { mkdir($tmp5, 0777, true); }
foreach (glob("$src/*") as $f) {
    $b = basename($f);
    if (str_contains($b, 'lock') || str_ends_with($b, '.sti')) { continue; }
    copy($f, "$tmp5/$b");
}

$MARK_FLIP  = 'zqflip';
$MARK_FLIP2 = 'zqflip2';

// baseline: MARK_FLIP must be absent before the native side writes it.
$idx5 = Zend_Search_Lucene::open($tmp5);
$before5 = $idx5->numDocs();
$check("FLIP baseline: '$MARK_FLIP' absent before writing (native)", $termCount($idx5, $probeField, $MARK_FLIP), 0);
$idx5 = null;
$cleanupLocks($tmp5);

// (1) NATIVE writes a witness doc (stream_writer, default cap).
$docsFlip = ['docs' => [['fields' => [
    ['name' => $probeField, 'value' => $MARK_FLIP, 'kind' => 'text', 'stored' => true],
]]]];
$docsFlipJson = "$tmp5/_flip1.json";
file_put_contents($docsFlipJson, json_encode($docsFlip));
$outF1 = []; $rcF1 = 0;
exec(escapeshellarg($STREAM) . ' ' . escapeshellarg($tmp5) . ' ' . escapeshellarg($docsFlipJson) . ' 2>&1', $outF1, $rcF1);
echo "  native append (flip 1/2): rc=$rcF1 " . implode(' ', $outF1) . "\n";
if ($rcF1 !== 0) { fwrite(STDERR, "flip: native append failed\n"); exit(1); }
@unlink($docsFlipJson);
$cleanupLocks($tmp5); // "flip": end of the native process, nobody holds the index open

// (2) ZSL opens and reads what the native side wrote, WITHOUT reindex: numDocs+1 and witness present.
$idx5 = Zend_Search_Lucene::open($tmp5);
$check('FLIP native→ZSL: numDocs += 1', $idx5->numDocs(), $before5 + 1);
$check("FLIP native→ZSL: '$MARK_FLIP' present (==1)", $termCount($idx5, $probeField, $MARK_FLIP), 1);
// MARK_FLIP2 must be absent before ZSL writes it.
$check("FLIP baseline: '$MARK_FLIP2' absent before writing (ZSL)", $termCount($idx5, $probeField, $MARK_FLIP2), 0);

// (3) ZSL writes another witness doc and closes ("flip" back toward the native side).
$dFlip = new Zend_Search_Lucene_Document();
$dFlip->addField(Zend_Search_Lucene_Field::Text($probeField, $MARK_FLIP2));
$idx5->addDocument($dFlip);
$idx5->commit();
$idx5 = null;
$cleanupLocks($tmp5); // nobody holds the index open

// (4) NATIVE (diff_read) opens and reads: must see BOTH witnesses, without reindex.
$queriesFlip = ['fields' => [$probeField], 'queries' => []];
$queriesFlipJson = "$tmp5/_flip_queries.json";
file_put_contents($queriesFlipJson, json_encode($queriesFlip));
$outF2 = []; $rcF2 = 0;
exec(escapeshellarg($DIFFREAD) . ' ' . escapeshellarg($tmp5) . ' ' . escapeshellarg($queriesFlipJson) . ' 2>&1', $outF2, $rcF2);
echo "  native diff_read (flip 2/2): rc=$rcF2\n";
if ($rcF2 !== 0) { fwrite(STDERR, "flip: native diff_read failed: " . implode(' ', $outF2) . "\n"); exit(1); }
@unlink($queriesFlipJson);
$repF2 = json_decode(end($outF2), true);
$termDictF = $repF2['term_dict'] ?? [];
$check("FLIP ZSL→native: '$MARK_FLIP' still present (==1)", $termDictF["$probeField\0$MARK_FLIP"] ?? 0, 1);
$check("FLIP ZSL→native: '$MARK_FLIP2' present (==1)", $termDictF["$probeField\0$MARK_FLIP2"] ?? 0, 1);
$cleanupLocks($tmp5);

array_map('unlink', glob("$tmp5/*") ?: []); @rmdir($tmp5);

// ---- cleanup + verdict ----
array_map('unlink', glob("$tmp/*") ?: []);
@rmdir($tmp);

echo $fails === 0
    ? "\nGOLDEN: PASS — ZSL re-reads and merges the native segment; native optimize() collapses multi-seg+deletes (interchange proven)\n"
    : "\nGOLDEN: FAIL — $fails checks diverge\n";
exit($fails === 0 ? 0 : 1);
