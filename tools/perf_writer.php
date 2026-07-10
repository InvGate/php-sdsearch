<?php
/**
 * WRITE-path PERF: index N docs with the NATIVE writer vs ZSL's PHP writer.
 *
 * All three index EXACTLY the same N docs over a COPY of the same base index
 * (does not touch <source-index-dir>). Measures wall-time and peak memory.
 *   - ZSL: open + addDocument×N + commit (its buffered writer, maxBufferedDocs=10).
 *   - native-batch: append_documents (ONE segment) via target/release/examples/append_writer.
 *   - native-stream: IndexWriter open + add_document×N + commit with a cap, bounded
 *     memory, via target/release/examples/stream_writer (ceil(N/cap) segments in one commit).
 *
 * Usage: ZEND_LUCENE_PATH=/path/to/zsl/library php tools/perf_writer.php [N] [source-index-dir] [cap]
 *   N defaults to 2000; source-index-dir defaults to sdsearch-core/tests/fixtures/zsl_index_kb
 *   (committed fixture); cap defaults to 1000.
 */
require __DIR__ . '/zsl_bootstrap.php';

$REPO  = dirname(__DIR__);
$N     = (int)($argv[1] ?? 2000);
$SRC   = $argv[2] ?? "$REPO/sdsearch-core/tests/fixtures/zsl_index_kb";
$BIN   = "$REPO/target/release/examples/append_writer";
$STREAM = "$REPO/target/release/examples/stream_writer";
$CAP    = (int)($argv[3] ?? 1000);

foreach ([$BIN => 'append_writer', $STREAM => 'stream_writer'] as $bin => $ex) {
    if (!is_file($bin)) {
        fwrite(STDERR, "missing native binary: $bin\n  build: cargo build -p sdsearch-core --release --example $ex\n");
        exit(2);
    }
}

// ---- generate N deterministic docs (title ~5 tokens, body ~40 tokens, id keyword) ----
$pool = ['printer', 'network', 'vpn', 'login', 'email', 'server', 'crash', 'slow', 'reset',
         'password', 'access', 'error', 'update', 'install', 'config', 'backup', 'restore',
         'timeout', 'license', 'upgrade', 'firewall', 'router', 'disk', 'memory', 'cpu'];
$np = count($pool);
$docs = [];
for ($i = 0; $i < $N; $i++) {
    $title = "ticket $i " . $pool[$i % $np] . ' ' . $pool[($i * 3) % $np] . " issue$i";
    $bodyWords = [];
    for ($j = 0; $j < 40; $j++) { $bodyWords[] = $pool[($i * 7 + $j * 5) % $np]; }
    $docs[] = [
        'title' => $title,
        'body'  => implode(' ', $bodyWords) . " ref$i",
        'id'    => "REC-$i",
    ];
}

$copyBase = function (string $dst) use ($SRC) {
    if (!is_dir($SRC)) { fwrite(STDERR, "does not exist: $SRC\n"); exit(2); }
    if (is_dir($dst)) { array_map('unlink', glob("$dst/*") ?: []); } else { mkdir($dst, 0777, true); }
    foreach (glob("$SRC/*") as $f) {
        $b = basename($f);
        if (str_contains($b, 'lock') || str_ends_with($b, '.sti')) { continue; }
        copy($f, "$dst/$b");
    }
};
$cleanupLocks = function (string $d) {
    foreach (array_merge(glob("$d/*.sti") ?: [], glob("$d/*lock*") ?: []) as $f) { @unlink($f); }
};

echo "== perf write: N=$N docs, base=$SRC ==\n";

// ---- ZSL ----
$tmpz = sys_get_temp_dir() . '/sdsearch_perfw_zsl_' . getmypid();
$copyBase($tmpz);
$memBefore = memory_get_peak_usage(true);
$t0 = microtime(true);
$idx = Zend_Search_Lucene::open($tmpz);
foreach ($docs as $d) {
    $doc = new Zend_Search_Lucene_Document();
    $doc->addField(Zend_Search_Lucene_Field::text('title', $d['title']));
    $doc->addField(Zend_Search_Lucene_Field::text('body', $d['body']));
    $doc->addField(Zend_Search_Lucene_Field::keyword('id', $d['id']));
    $idx->addDocument($doc);
}
$idx->commit();
$zslMs = (microtime(true) - $t0) * 1000.0;
$zslDocs = $idx->numDocs();
$idx = null;
$zslMemMb = (memory_get_peak_usage(true) - $memBefore) / 1048576.0;
$cleanupLocks($tmpz);
array_map('unlink', glob("$tmpz/*") ?: []); @rmdir($tmpz);

// ---- native ----
$tmpn = sys_get_temp_dir() . '/sdsearch_perfw_native_' . getmypid();
$copyBase($tmpn);
$spec = ['docs' => array_map(fn($d) => ['fields' => [
    ['name' => 'title', 'value' => $d['title'], 'kind' => 'text',    'stored' => true],
    ['name' => 'body',  'value' => $d['body'],  'kind' => 'text',    'stored' => true],
    ['name' => 'id',    'value' => $d['id'],    'kind' => 'keyword', 'stored' => true],
]], $docs)];
$docsJson = "$tmpn/_batch.json";
file_put_contents($docsJson, json_encode($spec));

$out = []; $rc = 0;
exec(escapeshellarg($BIN) . ' ' . escapeshellarg($tmpn) . ' ' . escapeshellarg($docsJson) . ' 2>&1', $out, $rc);
if ($rc !== 0) { fwrite(STDERR, "native append failed: " . implode("\n", $out) . "\n"); exit(1); }
$rep = json_decode(end($out), true);
$nativeMs = $rep['elapsed_ms'] ?? -1;
$nativeRssMb = ($rep['peak_rss_kb'] ?? 0) / 1024.0;

// verify the native side's correctness with ZSL (re-reads the native copy)
@unlink($docsJson); $cleanupLocks($tmpn);
$idx = Zend_Search_Lucene::open($tmpn);
$nativeDocs = $idx->numDocs();
$idx = null;
array_map('unlink', glob("$tmpn/*") ?: []); @rmdir($tmpn);

// ---- native streaming (bounded memory, rebuild shape) ----
$tmps = sys_get_temp_dir() . '/sdsearch_perfw_stream_' . getmypid();
$copyBase($tmps);
$specJson = "$tmps/_batch.json";
file_put_contents($specJson, json_encode($spec));
$outs = []; $rcs = 0;
exec(escapeshellarg($STREAM) . ' ' . escapeshellarg($tmps) . ' ' . escapeshellarg($specJson)
     . ' ' . escapeshellarg((string)$CAP) . ' 2>&1', $outs, $rcs);
if ($rcs !== 0) { fwrite(STDERR, "native stream failed: " . implode("\n", $outs) . "\n"); exit(1); }
$reps = json_decode(end($outs), true);
$streamMs = $reps['elapsed_ms'] ?? -1;
$streamRssMb = ($reps['peak_rss_kb'] ?? 0) / 1024.0;
$streamSegs = $reps['segments'] ?? 0;
@unlink($specJson); $cleanupLocks($tmps);
$idx = Zend_Search_Lucene::open($tmps);
$streamDocs = $idx->numDocs();
$idx = null;
array_map('unlink', glob("$tmps/*") ?: []); @rmdir($tmps);

// ---- report ----
$speedup       = $nativeMs > 0 ? $zslMs / $nativeMs : 0;
$streamSpeedup = $streamMs > 0 ? $zslMs / $streamMs : 0;
echo "\n";
printf("  %-14s %12s %12s %14s %10s\n", '', 'wall (ms)', 'docs', 'mem/rss (MB)', 'segments');
printf("  %-14s %12.1f %12d %14.1f %10s\n", 'ZSL', $zslMs, $zslDocs, $zslMemMb, '-');
printf("  %-14s %12.1f %12d %14.1f %10s\n", 'native-batch', $nativeMs, $nativeDocs, $nativeRssMb, '1');
printf("  %-14s %12.1f %12d %14.1f %10d\n", 'native-stream', $streamMs, $streamDocs, $streamRssMb, $streamSegs);
printf("\n  speedup batch:  %.1fx   speedup stream: %.1fx  (cap=%d, N=%d docs)\n",
       $speedup, $streamSpeedup, $CAP, $N);
$allOk = ($zslDocs === $nativeDocs) && ($zslDocs === $streamDocs);
echo $allOk
    ? "  correctness: OK (ZSL=$zslDocs, batch=$nativeDocs, stream=$streamDocs)\n"
    : "  correctness: DIFF (zsl=$zslDocs batch=$nativeDocs stream=$streamDocs)\n";
