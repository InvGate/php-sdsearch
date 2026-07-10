<?php
/**
 * Shared bootstrap for the Zend Search Lucene (ZSL) oracle tools.
 *
 * ZSL is NOT bundled in this repo. To run any tool that needs it, install a
 * Zend Search Lucene tree locally and point ZEND_LUCENE_PATH at the directory
 * that contains `Zend/Search/Lucene.php` (i.e. the ZF1 `library/` root).
 *
 * If ZEND_LUCENE_PATH is unset, the requiring tool no-ops (exit 0) so that CI
 * and casual runs stay green — the ZSL tools are opt-in, local-only.
 */

$zslRoot = getenv('ZEND_LUCENE_PATH') ?: '';
if ($zslRoot === '' || !is_file(rtrim($zslRoot, '/\\') . '/Zend/Search/Lucene.php')) {
    fwrite(STDERR, "ZEND_LUCENE_PATH not set (or missing Zend/Search/Lucene.php) — "
        . "skipping. Point it at a local Zend Search Lucene library/ root to run this tool.\n");
    exit(0);
}
$zslRoot = rtrim($zslRoot, '/\\');

if (!function_exists('zsl_env_path')) {
    function zsl_env_path(): string {
        return rtrim(getenv('ZEND_LUCENE_PATH'), '/\\');
    }
}

if (!defined('DS')) {
    define('DS', DIRECTORY_SEPARATOR);
}
if (!function_exists('intalophp')) {
    // ZSL hex-marker shim; on 64-bit intval(base16) matches the emitted bytes.
    function intalophp($str) { return intval($str, 16); }
}
set_include_path(get_include_path() . PATH_SEPARATOR . $zslRoot);
require_once 'Zend/Loader/Autoloader.php';
Zend_Loader_Autoloader::getInstance();
Zend_Search_Lucene_Analysis_Analyzer::setDefault(
    new Zend_Search_Lucene_Analysis_Analyzer_Common_Utf8Num_CaseInsensitive()
);
Zend_Search_Lucene_Search_Query_Wildcard::setMinPrefixLength(0);
