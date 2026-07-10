<?php
/**
 * Dump the ZSL "oracle" (ZSL's own output) for a built index into the JSON the
 * sdsearch-core parity tests consume. Needs a local Zend Search Lucene
 * (set ZEND_LUCENE_PATH). Operates on a COPY of <source-index-dir>.
 *
 * Usage: ZEND_LUCENE_PATH=/path/to/zsl/library \
 *        php tools/gen_zsl_fixture.php <index-name> <source-index-dir> <fixture-dir> [full|queries]
 *
 * Examples (reproduce the committed fixtures):
 *   php tools/gen_zsl_fixture.php incidents      indexes/incidents      sdsearch-core/tests/fixtures/zsl_index    full
 *   php tools/gen_zsl_fixture.php knowledgebase  indexes/knowledgebase  sdsearch-core/tests/fixtures/zsl_index_kb queries
 *
 * - 'full'    -> heavy oracle: docs(stored) + terms + doc_freq + postings + queries (for the
 *                byte-level tests: stored/terms/postings). Use with small indexes.
 * - 'queries' -> light oracle: num_docs + queries (for search-parity tests).
 *
 * CAVEAT: the committed `zsl_index/` fixture (incidents, 4 docs) is a FROZEN SNAPSHOT: it was
 * generated against a 4-doc demo source index that has since been replaced by a much larger one,
 * so re-running `incidents full` against the current source will NOT reproduce the committed
 * fixture (it would produce a huge index and break the unit tests' hardcoded values). The KB
 * fixture (`zsl_index_kb/`, 63 docs) IS reproducible against its committed source. To create a new
 * fixture at scale, use 'queries' with a fresh dir, not 'full' on a large source.
 *
 * ZSL gotchas handled here: intalophp shim (hex markers), DS constant, ZF1 autoloader; opens a
 * COPY (ZSL writes .sti + locks on open); .sti/locks are stripped from the fixture.
 */
require __DIR__ . '/zsl_bootstrap.php';

$indexName  = $argv[1] ?? die("usage: gen_zsl_fixture.php <index-name> <source-index-dir> <fixture-dir> [full|queries]\n");
$srcDir     = $argv[2] ?? die("missing <source-index-dir>\n");
$fixtureDir = $argv[3] ?? die("missing <fixture-dir>\n");
$mode       = $argv[4] ?? 'queries';

// per-index curated queries (chosen so term/wildcard/fuzzy/phrase DIVERGE)
$T = fn($t, $f = 'title') => new Zend_Search_Lucene_Search_Query_Term(new Zend_Search_Lucene_Index_Term($t, $f));
$W = fn($p, $f = 'title') => new Zend_Search_Lucene_Search_Query_Wildcard(new Zend_Search_Lucene_Index_Term($p, $f));
$F = fn($t, $s, $p, $f = 'title') => new Zend_Search_Lucene_Search_Query_Fuzzy(new Zend_Search_Lucene_Index_Term($t, $f), $s, $p);
$P = function (array $words, $f = 'title') {
    $q = new Zend_Search_Lucene_Search_Query_Phrase();
    foreach ($words as $w) { $q->addTerm(new Zend_Search_Lucene_Index_Term($w, $f)); }
    return $q;
};
$queryDefs = [
    'incidents' => [
        'term:title:new'          => $T('new'),
        'wildcard:title:ne*'      => $W('ne*'),
        'fuzzy:title:new:0.5:3'   => $F('new', 0.5, 3),
    ],
    'knowledgebase' => [
        'term:title:vpn'          => $T('vpn'),
        'term:title:mysql'        => $T('mysql'),
        'wildcard:title:manu*'    => $W('manu*'),
        'wildcard:title:set*'     => $W('set*'),
        'fuzzy:title:mysgl:0.5:3' => $F('mysgl', 0.5, 3),
        'phrase:title:how to'     => $P(['how', 'to']),
        'phrase:title:set up'     => $P(['set', 'up']),
    ],
];

// 1) copy the real index (without locks/.sti) into the fixture and open the COPY
$src = $srcDir;
if (!is_dir($src)) { die("index does not exist: $src\n"); }
if (!is_dir($fixtureDir)) { mkdir($fixtureDir, 0777, true); }
array_map('unlink', glob("$fixtureDir/*") ?: []);
foreach (glob("$src/*") as $f) {
    $b = basename($f);
    if (str_contains($b, 'lock') || str_ends_with($b, '.sti')) { continue; }
    copy($f, "$fixtureDir/$b");
}
$idx = Zend_Search_Lucene::open($fixtureDir);

// 2) queries → oracle (ZSL output, in rank order)
$run = function (Zend_Search_Lucene_Search_Query $q) use ($idx) {
    $out = [];
    foreach ($idx->find($q) as $hit) { $out[] = ['id' => $hit->id]; }
    return $out;
};
$queries = [];
foreach (($queryDefs[$indexName] ?? []) as $key => $q) { $queries[$key] = $run($q); }

$oracle = [
    'index'        => $indexName,
    'num_docs'     => $idx->numDocs(),
    'query_params' => ['fuzzy_similarity' => 0.5, 'fuzzy_prefix' => 3, 'wildcard_min_prefix' => 0],
];

// 3) 'full' mode: stored fields + term dict + postings (for byte-level tests)
if ($mode === 'full') {
    $docs = [];
    for ($i = 0; $i < $idx->count(); $i++) {
        if ($idx->isDeleted($i)) { continue; }
        $d = $idx->getDocument($i);
        $s = [];
        foreach ($d->getFieldNames() as $fn) { $s[$fn] = $d->getFieldValue($fn); }
        $docs[] = ['internal_id' => $i, 'stored' => $s];
    }
    $terms = [];
    $docFreq = [];
    $postings = [];
    $idx->resetTermsStream();
    while (($t = $idx->currentTerm()) !== null) {
        $terms[$t->field][] = $t->text;
        $docFreq[$t->field][$t->text] = $idx->docFreq($t);
        $pl = [];
        foreach ($idx->termPositions($t) as $doc => $ps) {
            $pl[] = ['doc' => $doc, 'freq' => count($ps), 'positions' => array_values($ps)];
        }
        $postings[$t->field][$t->text] = $pl;
        $idx->nextTerm();
    }
    $oracle += ['fields' => array_keys($terms), 'docs' => $docs, 'terms' => $terms,
                'doc_freq' => $docFreq, 'postings' => $postings];
}
$oracle['queries'] = $queries;

// 4) write oracle (next to the fixture dir) and clean up open artifacts
$oraclePath = dirname($fixtureDir) . '/zsl_expected' . ($indexName === 'knowledgebase' ? '_kb' : '') . '.json';
file_put_contents($oraclePath, json_encode($oracle, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE));
array_map('unlink', array_merge(glob("$fixtureDir/*.sti") ?: [], glob("$fixtureDir/*lock*") ?: []));

echo "fixture: $fixtureDir (" . count(glob("$fixtureDir/*")) . " files)\n";
echo "oracle:  $oraclePath  (mode=$mode, num_docs={$oracle['num_docs']})\n";
foreach ($queries as $k => $v) {
    echo sprintf("  %-26s -> {%s}\n", $k, implode(',', array_map(fn($h) => $h['id'], $v)));
}
