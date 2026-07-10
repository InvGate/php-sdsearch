<?php
/**
 * Perf baseline with Zend_Search_Lucene over the SAME index and the SAME queries
 * as perf_native.rs, to compare native vs ZSL latency.
 * Usage: ZEND_LUCENE_PATH=/path/to/zsl/library SDSEARCH_PERF_INDEX=/path php tools/perf_zsl.php
 */
require __DIR__ . '/zsl_bootstrap.php';

$dir = getenv('SDSEARCH_PERF_INDEX');
if ($dir === false) { fwrite(STDERR, "SDSEARCH_PERF_INDEX no seteada\n"); exit(0); }

$idx = Zend_Search_Lucene::open($dir);

// same text sub-tree as build_query (the fuzzy-text sub-query)
$F = fn($t, $s, $p, $f = null) => new Zend_Search_Lucene_Search_Query_Fuzzy(new Zend_Search_Lucene_Index_Term($t, $f), $s, $p);
$W = fn($p, $f = null) => new Zend_Search_Lucene_Search_Query_Wildcard(new Zend_Search_Lucene_Index_Term($p, $f));
$buildText = function (string $text) use ($F, $W) {
    $b = new Zend_Search_Lucene_Search_Query_Boolean();
    $words = array_values(array_filter(explode(' ', $text), fn($w) => trim($w) !== ''));
    if (count($words) > 1) { foreach ($words as $w) { $b->addSubquery($F($w, 0.5, 3), null); } }
    $b->addSubquery($F($text, 0.5, 3), null);
    $b->addSubquery($W($text . '*'), null);
    $b->addSubquery(Zend_Search_Lucene_Search_QueryParser::parse($text), null);
    $top = new Zend_Search_Lucene_Search_Query_Boolean();
    $top->addSubquery($b, true);
    return $top;
};

// WARNING: keep this list IDENTICAL to `queries` in sdsearch-core/examples/perf_native.rs,
// and lowercase/space-separated — the Rust side does lowercase + split_whitespace,
// this PHP side does not; if they diverge, the latency numbers compare different queries.
$queries = ['felipe', 'roa', 'juan rodriguez', 'andrés barrios'];
$iters = 20;
echo "query,p50_ms,p95_ms,hits\n";
foreach ($queries as $text) {
    $q = $buildText($text);
    $samples = [];
    $hits = 0;
    for ($i = 0; $i < $iters; $i++) {
        $t = microtime(true);
        $r = $idx->find($q);
        $samples[] = (microtime(true) - $t) * 1000.0;
        $hits = count($r);
    }
    sort($samples);
    $p = fn($ql) => $samples[(int)round((count($samples) - 1) * $ql)];
    printf("%s,%.3f,%.3f,%d\n", json_encode($text), $p(0.50), $p(0.95), $hits);
}
