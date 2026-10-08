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
if (!\is_array($decodedSynonyms) || !\is_array($decodedSynonyms['hits'] ?? null)) {
    \fwrite(\STDERR, "FAIL: search with synonyms=true did not return the {hits,...} envelope\n");
    exit(1);
}
\fwrite(\STDOUT, "search synonyms OK: " . \count($decodedSynonyms['hits']) . " hits\n");

// smoke test: exact_match y boolean_tree marshallan por serde y llegan al motor. Igual que
// el chequeo de synonyms de arriba, esto no asegura un ranking: prueba que las dos llaves
// nuevas hacen round-trip por el borde PHP (sdsearch-php es cdylib, 0 unit tests) y que el
// árbol acota de verdad — AND NOT no puede devolver más hits que el OR de las mismas hojas.
// Términos elegidos contra la fixture real: "vpn" matchea 2 docs ("alpha vpn guide",
// "gamma vpn tutorial"); "guide" matchea solo 1 de esos ("alpha vpn guide"), un subconjunto
// propio de "vpn" — así OR(vpn,guide)=2 y AND(vpn, NOT guide)=1 son distintos de verdad.
$exact = \json_decode($engine->search($indexDir, \json_encode([
    'text' => 'vpn',
    'exact_match' => true,
])), true);
if (!\is_array($exact['hits'] ?? null)) {
    \fwrite(\STDERR, "FAIL: search with exact_match=true did not return the {hits,...} envelope\n");
    exit(1);
}
\fwrite(\STDOUT, "search exact_match OK: " . \count($exact['hits']) . " hits\n");

$or = \json_decode($engine->search($indexDir, \json_encode([
    'boolean_tree' => ['type' => 'or', 'children' => [
        ['type' => 'term', 'phrase' => 'vpn'],
        ['type' => 'term', 'phrase' => 'guide'],
    ]],
    'limit' => 50,
])), true);
$andNot = \json_decode($engine->search($indexDir, \json_encode([
    'boolean_tree' => ['type' => 'and', 'children' => [
        ['type' => 'term', 'phrase' => 'vpn'],
        ['type' => 'not', 'child' => ['type' => 'term', 'phrase' => 'guide']],
    ]],
    'limit' => 50,
])), true);
if (!\is_array($or['hits'] ?? null) || !\is_array($andNot['hits'] ?? null)) {
    \fwrite(\STDERR, "FAIL: search with boolean_tree did not return the {hits,...} envelope\n");
    exit(1);
}
if (\count($andNot['hits']) >= \count($or['hits'])) {
    \fwrite(\STDERR, "FAIL: boolean_tree AND-NOT did not return fewer hits than the OR of the same leaves\n");
    exit(1);
}
\fwrite(\STDOUT, "search boolean_tree OK: or=" . \count($or['hits'])
    . " andNot=" . \count($andNot['hits']) . "\n");

// smoke test: hybrid_query rejects boolean_tree (and, by the same code path, exact_match).
// PRF relaxes the base query into a "should" clause, so a NOT inside the tree stops being a
// hard filter — hybrid_query/semantic_query must throw rather than silently soften it.
$rejected = false;
try {
    $engine->hybrid_query($indexDir, \json_encode([
        'boolean_tree' => ['type' => 'term', 'phrase' => 'vpn'],
    ]));
} catch (\Throwable $e) {
    $rejected = true;
}
if (!$rejected) {
    \fwrite(\STDERR, "FAIL: hybrid_query with boolean_tree did not throw\n");
    exit(1);
}
\fwrite(\STDOUT, "hybrid_query boolean_tree rejection OK\n");

// smoke test: fuzzy_similarity / fuzzy_prefix_len reach the engine. "mysgl" reaches the
// "mysql" doc only through the fuzzy leaf (similarity 0.8 with the default prefix of 3), so
// raising the threshold past it, or the prefix past the shared "mys", must drop the hit; a
// threshold outside [0, 1) must throw instead of silently switching fuzzy off.
$fuzzyHits = static fn (array $extra): int => \count(\json_decode($engine->search(
    $indexDir,
    \json_encode(['text' => 'mysgl', 'wildcard_min_prefix' => 0] + $extra)
), true)['hits']);
if ($fuzzyHits([]) !== 1 || $fuzzyHits(['fuzzy_similarity' => 0.9]) !== 0
    || $fuzzyHits(['fuzzy_prefix_len' => 4]) !== 0) {
    \fwrite(\STDERR, "FAIL: fuzzy_similarity/fuzzy_prefix_len did not reach the engine\n");
    exit(1);
}
$rejected = false;
try {
    $fuzzyHits(['fuzzy_similarity' => 1.0]);
} catch (\Throwable $e) {
    $rejected = true;
}
if (!$rejected) {
    \fwrite(\STDERR, "FAIL: fuzzy_similarity 1.0 did not throw\n");
    exit(1);
}
\fwrite(\STDOUT, "search fuzzy knobs OK\n");

exit(0);
