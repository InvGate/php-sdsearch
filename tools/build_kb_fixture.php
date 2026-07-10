<?php
/**
 * Build the synthetic KB fixture from tools/corpus/kb_corpus.json using a local
 * Zend Search Lucene (ZEND_LUCENE_PATH), optimize it to one segment, write it to
 * the fixture dir, dump the query oracle, and print the structural constants the
 * Rust tests pin. Regenerates a provably PII-free zsl_index_kb fixture.
 *
 * Usage: ZEND_LUCENE_PATH=/path/to/zsl/library php tools/build_kb_fixture.php
 */
require __DIR__ . '/zsl_bootstrap.php';

$root       = dirname(__DIR__);
$corpus     = json_decode(file_get_contents($root . '/tools/corpus/kb_corpus.json'), true);
$fixtureDir = $root . '/sdsearch-core/tests/fixtures/zsl_index_kb';
$oraclePath = $root . '/sdsearch-core/tests/fixtures/zsl_expected_kb.json';

// fresh build dir
$build = sys_get_temp_dir() . '/kb_build_' . getmypid();
@mkdir($build, 0777, true);
$idx = Zend_Search_Lucene::create($build);

// field-type mapping mirrors the host application's KB schema
$addField = function (Zend_Search_Lucene_Document $doc, string $name, string $value): void {
    if (str_ends_with($name, '_key')) {
        $doc->addField(Zend_Search_Lucene_Field::keyword($name, $value));
    } elseif (str_ends_with($name, '_attr')) {
        $doc->addField(Zend_Search_Lucene_Field::unIndexed($name, $value));
    } else {
        $doc->addField(Zend_Search_Lucene_Field::text($name, $value));
    }
};
foreach ($corpus as $row) {
    $doc = new Zend_Search_Lucene_Document();
    foreach ($row as $name => $value) { $addField($doc, $name, (string)$value); }
    $idx->addDocument($doc);
}
$idx->optimize();
$idx->commit();

// copy built index (minus locks/.sti) into the fixture dir
if (!is_dir($fixtureDir)) { mkdir($fixtureDir, 0777, true); }
array_map('unlink', glob("$fixtureDir/*") ?: []);
foreach (glob("$build/*") as $f) {
    $b = basename($f);
    if (str_contains($b, 'lock') || str_ends_with($b, '.sti')) { continue; }
    copy($f, "$fixtureDir/$b");
}
array_map('unlink', glob("$build/*") ?: []); @rmdir($build);

$open = Zend_Search_Lucene::open($fixtureDir);
$T = fn($t, $f='title') => new Zend_Search_Lucene_Search_Query_Term(new Zend_Search_Lucene_Index_Term($t, $f));
$W = fn($p, $f='title') => new Zend_Search_Lucene_Search_Query_Wildcard(new Zend_Search_Lucene_Index_Term($p, $f));
$F = fn($t,$s,$p,$f='title') => new Zend_Search_Lucene_Search_Query_Fuzzy(new Zend_Search_Lucene_Index_Term($t, $f), $s, $p);
$P = function(array $ws,$f='title'){ $q=new Zend_Search_Lucene_Search_Query_Phrase(); foreach($ws as $w){$q->addTerm(new Zend_Search_Lucene_Index_Term($w,$f));} return $q; };

// Dump the title term dictionary with doc frequencies, to CHOOSE a divergent query set.
$termFreq = [];
$open->resetTermsStream();
while (($t = $open->currentTerm()) !== null) {
    if ($t->field === 'title') { $termFreq[$t->text] = $open->docFreq($t); }
    $open->nextTerm();
}
arsort($termFreq);

// CHOSEN QUERIES — fill these in from the term dump (Step 3): keys are the same
// strings the Rust test will use. Each must return non-empty; the two term
// queries must return DIFFERENT doc-sets (for the assert_ne! divergence check).
$queryDefs = [
    'term:title:vpn'          => $T('vpn'),
    'term:title:laptop'       => $T('laptop'),
    'wildcard:title:print*'   => $W('print*'),
    'fuzzy:title:passwrd:0.5:3' => $F('passwrd', 0.5, 3),
    'phrase:title:setting up' => $P(['setting', 'up']),
    'phrase:title:cloud storage' => $P(['cloud', 'storage']),
];
$queries = [];
foreach ($queryDefs as $k => $q) {
    $queries[$k] = array_map(fn($hit) => ['id' => $hit->id], iterator_to_array($open->find($q)));
}
file_put_contents($oraclePath, json_encode([
    'index' => 'knowledgebase',
    'num_docs' => $open->numDocs(),
    'query_params' => ['fuzzy_similarity' => 0.5, 'fuzzy_prefix' => 3, 'wildcard_min_prefix' => 0],
    'queries' => $queries,
], JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE));
array_map('unlink', array_merge(glob("$fixtureDir/*.sti") ?: [], glob("$fixtureDir/*lock*") ?: []));

// print the structural constants the Rust tests pin + the chosen-query results
$cfs = glob("$fixtureDir/_*.cfs");
$seg = $cfs ? basename($cfs[0], '.cfs') : '(none)';
$genFiles = glob("$fixtureDir/segments_*");
echo "=== TEST CONSTANTS (copy into the Rust re-baseline) ===\n";
echo "num_docs:        {$open->numDocs()}\n";
echo "segment name:    {$seg}\n";
echo "segments.gen ->  " . implode(',', array_map('basename', $genFiles)) . "\n";
echo "=== TITLE TERM DICTIONARY (text -> docFreq, for choosing queries) ===\n";
foreach ($termFreq as $text => $df) { echo sprintf("  %-20s %d\n", $text, $df); }
echo "=== CHOSEN QUERIES (query -> result doc-ids) ===\n";
foreach ($queries as $k => $v) { echo sprintf("  %-30s -> %d docs {%s}\n", $k, count($v), implode(',', array_map(fn($h)=>$h['id'],$v))); }
echo "NOTE: read generation / name_counter / records_range via the sdsearch parser tests.\n";
