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

// smoke test: Engine::hybrid_query with explicit "prf"/"hybrid" keys — catches a mistyped
// serde field name at the PHP boundary (e.g. k/depth/top_k), since sdsearch-php is cdylib
// (0 unit tests) and this is the only place the explicit-keys path is exercised.
$hybridExplicit = $engine->hybrid_query($indexDir, \json_encode([
    'text' => 'vpn',
    'prf' => ['top_k' => 3, 'num_terms' => 5],
    'hybrid' => ['k' => 10, 'depth' => 5],
]));
$decodedExplicit = \json_decode($hybridExplicit, true);
if (!\is_array($decodedExplicit)) {
    \fwrite(\STDERR, "FAIL: hybrid_query with explicit prf/hybrid keys did not return a JSON array\n");
    exit(1);
}
\fwrite(\STDOUT, "hybrid_query explicit-keys OK: " . \count($decodedExplicit) . " hits\n");

// smoke test: Engine::search with "synonyms" => true marshals through serde without
// panicking. This doesn't assert on the expansion changing hits (the fixture's "vpn" term
// has no bundled synonym counterpart) — it just proves the DTO field round-trips at the
// PHP boundary, mirroring the explicit-keys check above for hybrid_query.
$synonyms = $engine->search($indexDir, \json_encode(['text' => 'vpn', 'synonyms' => true]));
$decodedSynonyms = \json_decode($synonyms, true);
if (!\is_array($decodedSynonyms)) {
    \fwrite(\STDERR, "FAIL: search with synonyms=true did not return a JSON array\n");
    exit(1);
}
\fwrite(\STDOUT, "search synonyms OK: " . \count($decodedSynonyms) . " hits\n");

exit(0);
