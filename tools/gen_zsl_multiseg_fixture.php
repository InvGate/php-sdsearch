<?php
/**
 * Generates a MULTI-SEGMENT ZSL index (incremental adds without optimize) with a few
 * deletes, and dumps the oracle (ZSL's own output) for the parity tests.
 *
 * Does NOT copy an existing index: creates a new one, controlling flushes to force
 * >= 2 segments (ZSL creates one segment per commit() with buffered docs; with fewer than
 * mergeFactor (10) segments there is no auto-merge).
 *
 * Needs a local Zend Search Lucene (set ZEND_LUCENE_PATH). Doesn't read any external corpus —
 * it builds the multi-segment index from scratch, so there is nothing to decouple beyond ZSL
 * itself.
 *
 * Usage: ZEND_LUCENE_PATH=/path/to/zsl/library \
 *        php tools/gen_zsl_multiseg_fixture.php [fixture-dir]
 *
 * CAVEAT (boolean-oracle transcription): $buildQuery/$buildText below
 * RE-TRANSCRIBE Zend Lucene's boolean query builder by hand and share the same simplifications as
 * build_query (Rust) — they are not a direct call into a real host-application backend. In
 * particular:
 *   - they do NOT escape '-', ',', ':' in the text/values (unlike the real preprocessing
 *     such a backend would do before assembling the query);
 *   - they do NOT apply the "_key" suffix to the where/in fields — fields are passed
 *     literal/verbatim, already suffixed by hand in the fixture;
 *   - they do NOT model the empty-result fallback of the search layer nor the min_score
 *     filter that the real search layer applies.
 * So these parity tests only validate the surface exercised here (plain ASCII,
 * explicit _key fields, single index). The PHP integration layer
 * MUST re-validate build_query against the real host boolean query builder + search layer
 * before relying on this parity for production.
 */
require __DIR__ . '/zsl_bootstrap.php';

$fixtureDir = $argv[1] ?? 'sdsearch-core/tests/fixtures/zsl_index_multiseg';

// clean dir
if (is_dir($fixtureDir)) { array_map('unlink', glob("$fixtureDir/*") ?: []); }
else { mkdir($fixtureDir, 0777, true); }

$idx = Zend_Search_Lucene::create($fixtureDir);
$idx->setMaxBufferedDocs(1000); // no buffer-driven auto-flush; we control it with commit()

// fields: title (text, all-fields), id_key/lang_key/cat_key (keyword)
$mk = function ($idKey, $title, $lang, $cat) {
    $d = new Zend_Search_Lucene_Document();
    $d->addField(Zend_Search_Lucene_Field::keyword('id_key', (string)$idKey));
    $d->addField(Zend_Search_Lucene_Field::text('title', $title));
    $d->addField(Zend_Search_Lucene_Field::keyword('lang_key', $lang));
    $d->addField(Zend_Search_Lucene_Field::keyword('cat_key', (string)$cat));
    return $d;
};

// batch 1 -> segment 1
$idx->addDocument($mk(100, 'alpha vpn guide', 'es', 1));
$idx->addDocument($mk(101, 'beta mysql setup', 'es', 2));
$idx->commit();
// batch 2 -> segment 2 (vpn crosses the segment boundary)
$idx->addDocument($mk(102, 'gamma vpn tutorial', 'en', 1));
$idx->addDocument($mk(103, 'delta backup notes', 'en', 3));
$idx->commit();
// batch 3 -> segment 3
$idx->addDocument($mk(104, 'epsilon how to restore', 'es', 2));
$idx->addDocument($mk(105, 'zeta how to reset', 'en', 1));
$idx->commit();

// delete a doc from a middle segment ("delta backup notes", id_key 103)
foreach (range(0, $idx->maxDoc() - 1) as $i) {
    if (!$idx->isDeleted($i) && $idx->getDocument($i)->getFieldValue('id_key') === '103') {
        $idx->delete($i);
    }
}
$idx->commit();

// --- oracle ---
$run = function (Zend_Search_Lucene_Search_Query $q) use ($idx) {
    $out = [];
    foreach ($idx->find($q) as $hit) { $out[] = ['id' => $hit->id]; }
    return $out;
};
$T = fn($t, $f = 'title') => new Zend_Search_Lucene_Search_Query_Term(new Zend_Search_Lucene_Index_Term($t, $f));
$W = fn($p, $f = 'title') => new Zend_Search_Lucene_Search_Query_Wildcard(new Zend_Search_Lucene_Index_Term($p, $f));
$F = fn($t, $s, $p, $f = 'title') => new Zend_Search_Lucene_Search_Query_Fuzzy(new Zend_Search_Lucene_Index_Term($t, $f), $s, $p);
$P = function (array $ws, $f = 'title') {
    $q = new Zend_Search_Lucene_Search_Query_Phrase();
    foreach ($ws as $w) { $q->addTerm(new Zend_Search_Lucene_Index_Term($w, $f)); }
    return $q;
};

$queries = [
    'term:title:vpn'        => $T('vpn'),      // crosses segments 1 and 2
    'term:title:mysql'      => $T('mysql'),    // only segment 1
    'term:title:backup'     => $T('backup'),   // in a deleted doc -> must return empty
    'wildcard:title:re*'    => $W('re*'),      // restore/reset
    'fuzzy:title:mysgl:0.5:3' => $F('mysgl', 0.5, 3),
    'phrase:title:how to'   => $P(['how', 'to']),
];
$out = ['index' => 'multiseg', 'num_docs' => $idx->numDocs()];
$out['queries'] = array_map($run, $queries);

// boolean: transcription of Zend Lucene's boolean query builder for the surface the host application uses.
// helper: fuzzy text sub-boolean
$buildText = function (string $text) use ($F, $W) {
    $b = new Zend_Search_Lucene_Search_Query_Boolean();
    $words = array_values(array_filter(explode(' ', $text), fn($w) => trim($w) !== ''));
    if (count($words) > 1) {
        foreach ($words as $w) { $b->addSubquery($F($w, 0.5, 3, null), null); }
    }
    $b->addSubquery($F($text, 0.5, 3, null), null);
    $b->addSubquery($W($text . '*', null), null);
    $b->addSubquery(Zend_Search_Lucene_Search_QueryParser::parse($text), null);
    return $b;
};
$buildQuery = function (string $text, array $where, array $in) use ($buildText, $T) {
    $top = new Zend_Search_Lucene_Search_Query_Boolean();
    if ($text !== '') { $top->addSubquery($buildText($text), true); }
    foreach ($where as [$field, $values, $sign]) {
        $fq = new Zend_Search_Lucene_Search_Query_Boolean();
        foreach ($values as $v) {
            $fq->addSubquery($T($v, $field), null);
        }
        $top->addSubquery($fq, $sign);
    }
    // the IN-clause merge: ALL 'in' groups collapse into a single MultiTerm (global OR), required once.
    if (!empty($in)) {
        $mq = new Zend_Search_Lucene_Search_Query_MultiTerm();
        foreach ($in as [$field, $values]) {
            foreach ($values as $v) { $mq->addTerm(new Zend_Search_Lucene_Index_Term($v, $field), null); }
        }
        $top->addSubquery($mq, true);
    }
    return $top;
};

$boolQueries = [
    'text-only:vpn'          => $buildQuery('vpn', [], []),
    'text+where:vpn|lang=es' => $buildQuery('vpn', [['lang_key', ['es'], null]], []),
    'text+in:vpn|cat=1,2'    => $buildQuery('vpn', [], [['cat_key', ['1', '2']]]),
    // multi-field in: distinguishes global-OR (ZSL) from AND-of-ORs. cat=3 matches no live doc,
    // lang=en matches doc2/doc5 => with vpn (must) the MultiTerm(cat=3 OR lang=en) leaves {doc2}.
    'text+in-multi:vpn|cat=3&lang=en' => $buildQuery('vpn', [], [['cat_key', ['3']], ['lang_key', ['en']]]),
    'where-mustnot:how|lang!=en' => $buildQuery('how', [['lang_key', ['en'], false]], []),
    'text-multiword:how to' => $buildQuery('how to', [], []),
];
$out['bool_queries'] = array_map($run, $boolQueries);

// stored map (to check per-base routing)
$docs = [];
for ($i = 0; $i < $idx->maxDoc(); $i++) {
    if ($idx->isDeleted($i)) { continue; }
    $d = $idx->getDocument($i);
    $docs[] = ['id' => $i, 'stored' => [
        'id_key'   => $d->getFieldValue('id_key'),
        'title'    => $d->getFieldValue('title'),
        'lang_key' => $d->getFieldValue('lang_key'),
        'cat_key'  => $d->getFieldValue('cat_key'),
    ]];
}
$out['docs'] = $docs;

$oraclePath = dirname($fixtureDir) . '/zsl_expected_multiseg.json';
file_put_contents($oraclePath, json_encode($out, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE));
// clean up open artifacts (locks/.sti); keep segments/.cfs/.del
array_map('unlink', array_merge(glob("$fixtureDir/*.sti") ?: [], glob("$fixtureDir/*lock*") ?: []));

$cfs = count(glob("$fixtureDir/*.cfs"));
echo "fixture: $fixtureDir ($cfs .cfs files, num_docs={$out['num_docs']})\n";
echo "oracle:  $oraclePath\n";
if ($cfs < 2) { fwrite(STDERR, "WARNING: <2 .cfs — failed to force multi-segment\n"); exit(1); }
