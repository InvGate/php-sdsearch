<?php
/**
 * Smoke of the write FFI: build path of `SdSearch\Writer`
 * (open/add_document/commit) from PHP against the native `libsdsearch.so` extension.
 *
 * Tests the key requirement: an empty ZSL index created by ZSL can be built by the
 * native writer via FFI (without going through example binaries) and ZSL can re-read the result.
 *
 * Usage:
 *   ZEND_LUCENE_PATH=/path/to/zsl/library \
 *   php -d extension=<abs>/target/release/libsdsearch.so tools/writer_ffi_smoke.php
 */
require __DIR__ . '/zsl_bootstrap.php';

if (!extension_loaded('sdsearch')) {
    fwrite(STDERR, "load the extension with -d extension=<abs>/libsdsearch.so\n");
    exit(2);
}

function check($label, $got, $exp) {
    if ($got != $exp) {
        fwrite(STDERR, "FAIL $label: got=" . var_export($got, true) . " exp=" . var_export($exp, true) . "\n");
        exit(1);
    }
    echo "OK $label ($got)\n";
}

$idxDir = sys_get_temp_dir() . '/sdsearch_wffi_' . getmypid();
@mkdir($idxDir, 0777, true);

// 1) empty index via ZSL
$z = Zend_Search_Lucene::create($idxDir);
$z->commit();
unset($z);

// 2) native build path
$w = new \SdSearch\Writer();
$w->open($idxDir);
$w->add_document(json_encode(['fields' => [
    ['name' => 'title', 'value' => 'vpn setup guide', 'kind' => 'text'],
    ['name' => 'id_key', 'value' => '165', 'kind' => 'keyword'],
    ['name' => 'status_id_attr', 'value' => '6', 'kind' => 'unindexed'],
]]));
$w->add_document(json_encode(['fields' => [
    ['name' => 'title', 'value' => 'reset your password', 'kind' => 'text'],
    ['name' => 'id_key', 'value' => '166', 'kind' => 'keyword'],
]]));
$cnt = $w->commit();
check('commit doc_count', $cnt, 2);

// 3) ZSL re-reads (lock-free copy)
$ro = $idxDir . '_ro';
@mkdir($ro, 0777, true);
foreach (glob("$idxDir/*") as $f) {
    if (!str_contains(basename($f), 'lock')) {
        copy($f, "$ro/" . basename($f));
    }
}
$zr = Zend_Search_Lucene::open($ro);
check('ZSL numDocs', $zr->numDocs(), 2);
$hits = $zr->find('title:vpn');
check('find(title:vpn) count', count($hits), 1);
check('find(title:vpn) id_key', $hits ? $hits[0]->getDocument()->getFieldValue('id_key') : '-', '165');

// --- incremental: rebuild of doc id_key=165 ---
$w2 = new \SdSearch\Writer();
$w2->open($idxDir);
$docId = $w2->find_doc_id('id', '165');           // resolves id_key:165 -> internal doc-id
echo "find_doc_id(id,165)=$docId\n";
check('find_doc_id(id,165) >= 0', $docId >= 0, true);
$miss = $w2->find_doc_id('id', '999999');         // nonexistent
check('find_doc_id(id,999999)', $miss, -1);
if ($docId >= 0) { $w2->delete_document($docId); }
$w2->add_document(json_encode(['fields'=>[
    ['name'=>'title','value'=>'vpn setup guide v2','kind'=>'text'],
    ['name'=>'id_key','value'=>'165','kind'=>'keyword'],
]]));
$w2->optimize();                                   // collapses deletes + re-add into 1 segment

// verify with ZSL: still 2 docs (deleted old 165, added the new one)
$ro2 = $idxDir . '_ro2'; @mkdir($ro2, 0777, true);
foreach (glob("$idxDir/*") as $f) { if (!str_contains(basename($f),'lock')) copy($f, "$ro2/".basename($f)); }
$zr2 = Zend_Search_Lucene::open($ro2);
check('ZSL numDocs post-rebuild', $zr2->numDocs(), 2);
check('find(title:v2) count', count($zr2->find('title:v2')), 1);
array_map('unlink', glob("$ro2/*") ?: []); @rmdir($ro2);

// cleanup
array_map('unlink', glob("$idxDir/*") ?: []);
@rmdir($idxDir);
array_map('unlink', glob("$ro/*") ?: []);
@rmdir($ro);

echo "BUILD SMOKE OK\n";
