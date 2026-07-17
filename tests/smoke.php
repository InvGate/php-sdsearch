<?php
declare(strict_types=1);

// smoke test: the extension loads and exposes sdsearch_version()
if (!\extension_loaded('sdsearch')) {
    \fwrite(\STDERR, "FAIL: sdsearch extension not loaded\n");
    exit(1);
}
$v = \sdsearch_version();
if (!\is_string($v) || $v === '') {
    \fwrite(\STDERR, "FAIL: sdsearch_version() returned invalid value\n");
    exit(1);
}
\fwrite(\STDOUT, "OK sdsearch_version=$v\n");

// smoke test: Engine::hybrid_query returns a JSON array against a real fixture index.
$indexDir = __DIR__ . '/../sdsearch-core/tests/fixtures/zsl_index_multiseg';
$engine = new \SdSearch\Engine();
$hybrid = $engine->hybrid_query($indexDir, \json_encode(['text' => 'vpn']));
$decoded = \json_decode($hybrid, true);
if (!\is_array($decoded)) {
    \fwrite(\STDERR, "FAIL: hybrid_query did not return a JSON array\n");
    exit(1);
}
\fwrite(\STDOUT, "hybrid_query OK: " . \count($decoded) . " hits\n");

exit(0);
