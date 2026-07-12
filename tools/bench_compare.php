<?php
/**
 * Cross-engine benchmark: runs ONE measurement (engine × workload × size) in a CLEAN process
 * and prints ONE JSON line, so the orchestrator (benches/run.sh) can isolate each run's memory.
 *
 * Compares OUR PHP extension (SdSearch\Writer / SdSearch\Engine) against Zend Search Lucene on
 * the same deterministic corpus and the same queries. Wall time + two memory numbers:
 *   - php_peak_mb: memory_get_peak_usage(true). For `zend` this is the real footprint; for
 *     `sdsearch` it MISSES the Rust heap (allocated outside the Zend memory manager) — so it
 *     understates sdsearch. Reported for context, not as the cross-engine metric.
 *   - rss_peak_mb: process peak RSS (VmHWM from /proc/self/status). Includes BOTH engines' real
 *     footprint (Rust heap shows up here), so this is the comparable cross-engine number. The
 *     exact Rust HEAP (not RSS) comes from the native bench (examples/bench_engine.rs).
 *
 * Usage:
 *   php tools/bench_compare.php <engine> <workload> <N> [iters] [index_dir]
 *     engine    ∈ {sdsearch, zend}
 *     workload  ∈ {build, rebuild, churn, search}
 *     N         index size (docs added on top of the ~20-doc KB base)
 *     iters     search only: sampled query iterations (default 50)
 *     index_dir build/churn/search: the persistent index dir (build writes it; churn/search read
 *               it). rebuild ignores it (uses its own temp dir).
 *
 * `zend` requires ZEND_LUCENE_PATH (see zsl_bootstrap.php); without it the tool no-ops (exit 0),
 * which the orchestrator records as "skipped". `sdsearch` requires the extension to be loaded
 * (`php -d extension=target/release/libsdsearch.so ...`).
 *
 * The corpus generator and the three query tokens MUST stay byte-identical to
 * sdsearch-core/examples/bench_engine.rs, or the two engines compare different work.
 */

$REPO = dirname(__DIR__);
$engine   = $argv[1] ?? '';
$workload = $argv[2] ?? '';
$N        = (int)($argv[3] ?? 1000);
$iters    = (int)($argv[4] ?? 50);
$indexDir = $argv[5] ?? (sys_get_temp_dir() . "/sdsearch_bench_persist_{$engine}_{$N}");

if (!in_array($engine, ['sdsearch', 'zend'], true)) {
    fwrite(STDERR, "engine must be sdsearch|zend\n");
    exit(2);
}
if (!in_array($workload, ['build', 'rebuild', 'churn', 'search'], true)) {
    fwrite(STDERR, "workload must be build|rebuild|churn|search\n");
    exit(2);
}

if ($engine === 'zend') {
    // no-ops (exit 0) if ZEND_LUCENE_PATH is unset → orchestrator records a clean skip.
    require __DIR__ . '/zsl_bootstrap.php';
} else {
    if (!class_exists('SdSearch\\Writer') || !class_exists('SdSearch\\Engine')) {
        fwrite(STDERR, "sdsearch extension not loaded — run with: php -d extension="
            . "$REPO/target/release/libsdsearch.so ...\n");
        exit(2);
    }
}

// ---- planted-term corpus (byte-identical to examples/bench_engine.rs) ----
const COMMON  = 'widetoken';     // in EVERY doc  → "many"  (doc_freq == N)
const RARE    = 'sparsetoken';   // in RARE_K docs → "few"   (doc_freq == RARE_K)
const MISSING = 'absenttoken';   // never emitted → "none"  (doc_freq == 0)
const RARE_K  = 5;
const POOL = ['printer', 'network', 'vpn', 'login', 'email', 'server', 'crash', 'slow', 'reset',
    'password', 'access', 'error', 'update', 'install', 'config', 'backup', 'restore',
    'timeout', 'license', 'upgrade', 'firewall', 'router', 'disk', 'memory', 'cpu'];

function is_rare(int $i, int $n): bool {
    $step = max(1, intdiv($n, RARE_K));
    return $i % $step === 0 && intdiv($i, $step) < RARE_K;
}

/** one deterministic doc (title/body/id), matching gen_one() in bench_engine.rs. title/body
 *  draw ONLY from the fixed POOL (bounded vocabulary); the only unique-per-doc term is `id`. */
function gen_one(int $i, int $n, string $prefix): array {
    $np = count(POOL);
    $title = 'ticket ' . POOL[$i % $np] . ' ' . POOL[($i * 3) % $np] . ' ' . POOL[($i * 7) % $np];
    $bodyWords = [];
    for ($j = 0; $j < 40; $j++) { $bodyWords[] = POOL[($i * 7 + $j * 5) % $np]; }
    $body = implode(' ', $bodyWords) . ' ' . COMMON;
    if (is_rare($i, $n)) { $body .= ' ' . RARE; }
    return ['title' => $title, 'body' => $body, 'id' => "$prefix-$i"];
}

/** process peak RSS (VmHWM) in KB; 0 if unavailable. */
function rss_peak_kb(): int {
    $s = @file_get_contents('/proc/self/status');
    if ($s !== false && preg_match('/^VmHWM:\s+(\d+)/m', $s, $m)) { return (int)$m[1]; }
    return 0;
}

$KB_BASE = "$REPO/sdsearch-core/tests/fixtures/zsl_index_kb";

/** copies the KB fixture (or any index dir) to $dst, skipping lock/.sti files. */
function copy_index(string $src, string $dst): void {
    if (!is_dir($src)) { fwrite(STDERR, "missing base: $src\n"); exit(2); }
    if (is_dir($dst)) { array_map('unlink', glob("$dst/*") ?: []); } else { mkdir($dst, 0777, true); }
    foreach (glob("$src/*") as $f) {
        $b = basename($f);
        if (str_contains($b, 'lock') || str_ends_with($b, '.sti')) { continue; }
        copy($f, "$dst/$b");
    }
}
function cleanup_dir(string $d): void {
    foreach (glob("$d/*") ?: [] as $f) { @unlink($f); }
    @rmdir($d);
}

// ---- per-engine add / delete / search primitives (generate docs on the fly) ----

/** adds docs numbered start..start+count into $dir; returns nothing. Streaming feed (no array). */
function add_docs(string $engine, $writer, int $start, int $count, int $n, string $prefix): void {
    for ($i = $start; $i < $start + $count; $i++) {
        $d = gen_one($i, $n, $prefix);
        if ($engine === 'zend') {
            $doc = new Zend_Search_Lucene_Document();
            $doc->addField(Zend_Search_Lucene_Field::text('title', $d['title']));
            $doc->addField(Zend_Search_Lucene_Field::text('body', $d['body']));
            $doc->addField(Zend_Search_Lucene_Field::keyword('id', $d['id']));
            $writer->addDocument($doc);
        } else {
            $writer->add_document(json_encode(['fields' => [
                ['name' => 'title', 'value' => $d['title'], 'kind' => 'text'],
                ['name' => 'body',  'value' => $d['body'],  'kind' => 'text'],
                ['name' => 'id',    'value' => $d['id'],    'kind' => 'keyword'],
            ]]));
        }
    }
}

/** opens a writer over $dir (Zend index handle or SdSearch\Writer). */
function open_writer(string $engine, string $dir) {
    if ($engine === 'zend') { return Zend_Search_Lucene::open($dir); }
    $w = new SdSearch\Writer();
    $w->open($dir);
    return $w;
}

/** Runs the free-text query $token at paging depth $limit (0 = unlimited) and returns the hit
 *  count. Uses the SAME fuzzy+wildcard+parser shape on both engines (mirrors query.rs
 *  text_subquery / perf_zsl.php buildText) and reopens the index per call, exactly as
 *  SdSearch\Engine::search does — so both engines measure the same per-request cost.
 *  NOTE: Zend's find() has no top-K limit (it always scores + returns all), so for zend the
 *  $limit only affects the returned count, not the work — top-20 and top-100 latencies coincide. */
function query_count(string $engine, string $dir, string $token, int $limit): int {
    if ($engine === 'sdsearch') {
        $eng = new SdSearch\Engine();
        $json = $eng->search($dir, json_encode(['text' => $token, 'limit' => $limit]));
        return count(json_decode($json, true) ?: []);
    }
    $F = fn($t, $s, $p, $f = null) => new Zend_Search_Lucene_Search_Query_Fuzzy(new Zend_Search_Lucene_Index_Term($t, $f), $s, $p);
    $W = fn($p, $f = null) => new Zend_Search_Lucene_Search_Query_Wildcard(new Zend_Search_Lucene_Index_Term($p, $f));
    $b = new Zend_Search_Lucene_Search_Query_Boolean();
    $b->addSubquery($F($token, 0.5, 3), null);
    $b->addSubquery($W($token . '*'), null);
    $b->addSubquery(Zend_Search_Lucene_Search_QueryParser::parse($token), null);
    $top = new Zend_Search_Lucene_Search_Query_Boolean();
    $top->addSubquery($b, true);
    $idx = Zend_Search_Lucene::open($dir);
    $hits = $idx->find($top);
    return $limit > 0 ? min($limit, count($hits)) : count($hits);
}

/**
 * Commits the open writer INSIDE the timed region and returns the resulting LIVE doc count for
 * sdsearch (read from document_count() — its contract is "live base + buffered − deletes" = the
 * post-commit total — while the writer is still open, so no reopen is timed), or null for zend
 * (whose count needs a reopen, done by the caller AFTER stopping the clock). Zend is never
 * loaded in the sdsearch path, so the branch is unambiguous.
 */
function finish_writer(string $engine, $w): ?int {
    if ($engine === 'sdsearch') {
        $c = $w->document_count();
        $w->commit();
        return $c;
    }
    $w->commit();
    return null;
}

/** Like finish_writer(), but OPTIMIZES to a single segment (production shape). Used by build
 *  (churn/search setup) and rebuild. SdSearch\Writer::optimize() commits + merges and consumes
 *  the writer; Zend commits then optimizes its still-open index handle. */
function finish_writer_optimized(string $engine, $w): ?int {
    if ($engine === 'sdsearch') {
        $c = $w->document_count();
        $w->optimize();
        return $c;
    }
    $w->commit();
    $w->optimize();
    return null;
}

function emit(array $row): void { echo json_encode($row) . "\n"; }

// ---- workloads ----

switch ($workload) {
    case 'build': {
        // UNMEASURED setup: create a persistent N-doc index for churn/search to reuse.
        copy_index($KB_BASE, $indexDir);
        $w = open_writer($engine, $indexDir);
        add_docs($engine, $w, 0, $N, $N, 'REC');
        $dc = finish_writer_optimized($engine, $w); // production shape: single segment
        if ($dc === null) { $dc = Zend_Search_Lucene::open($indexDir)->numDocs(); }
        emit(['engine' => $engine, 'workload' => 'build', 'n' => $N,
              'index_dir' => $indexDir, 'doc_count' => $dc]);
        break;
    }

    case 'rebuild': {
        $dir = sys_get_temp_dir() . "/sdsearch_bench_rebuild_{$engine}_{$N}_" . getmypid();
        copy_index($KB_BASE, $dir);
        $t0 = microtime(true);
        $w = open_writer($engine, $dir);
        add_docs($engine, $w, 0, $N, $N, 'REC');
        $dc = finish_writer_optimized($engine, $w); // production rebuild ends optimized
        $ms = (microtime(true) - $t0) * 1000.0;
        if ($dc === null) { $dc = Zend_Search_Lucene::open($dir)->numDocs(); }
        emit(['engine' => $engine, 'workload' => 'rebuild', 'n' => $N, 'ms' => round($ms, 3),
              'php_peak_mb' => round(memory_get_peak_usage(true) / 1048576, 2),
              'rss_peak_mb' => round(rss_peak_kb() / 1024, 2), 'doc_count' => $dc]);
        cleanup_dir($dir);
        break;
    }

    case 'churn': {
        // requires a pre-built index at $indexDir; copy to scratch so re-runs are idempotent.
        $scratch = sys_get_temp_dir() . "/sdsearch_bench_churn_{$engine}_{$N}_" . getmypid();
        copy_index($indexDir, $scratch);
        $onePct = max(1, intdiv($N, 100));
        // $before only when a Zend reader is available (zend path); sdsearch path has no Zend.
        $before = class_exists('Zend_Search_Lucene') ? Zend_Search_Lucene::open($scratch)->numDocs() : null;
        $t0 = microtime(true);
        $w = open_writer($engine, $scratch);
        for ($gid = 0; $gid < $onePct; $gid++) {
            if ($engine === 'zend') { $w->delete($gid); } else { $w->delete_document($gid); }
        }
        add_docs($engine, $w, 0, $onePct, $N, 'CHURN');
        $after = finish_writer($engine, $w);
        $ms = (microtime(true) - $t0) * 1000.0;
        if ($after === null) { $after = Zend_Search_Lucene::open($scratch)->numDocs(); }
        emit(['engine' => $engine, 'workload' => 'churn', 'n' => $N, 'pct1' => $onePct,
              'ms' => round($ms, 3),
              'php_peak_mb' => round(memory_get_peak_usage(true) / 1048576, 2),
              'rss_peak_mb' => round(rss_peak_kb() / 1024, 2),
              'doc_count_before' => $before, 'doc_count' => $after]);
        cleanup_dir($scratch);
        break;
    }

    case 'search': {
        // requires a pre-built index at $indexDir (read-only). Times two realistic paging depths
        // (top-20, top-100); reports the TRUE hit count per class (one unlimited call, unmeasured).
        $classes = ['many' => COMMON, 'few' => RARE, 'none' => MISSING];
        $out = ['engine' => $engine, 'workload' => 'search', 'n' => $N, 'iters' => $iters];
        foreach ($classes as $label => $token) {
            $entry = ['hits' => query_count($engine, $indexDir, $token, 0)]; // true freq (unlimited)
            foreach ([20, 100] as $lim) {
                query_count($engine, $indexDir, $token, $lim); // warm-up (not sampled)
                $samples = [];
                for ($k = 0; $k < $iters; $k++) {
                    $t0 = microtime(true);
                    query_count($engine, $indexDir, $token, $lim);
                    $samples[] = (microtime(true) - $t0) * 1000.0;
                }
                sort($samples);
                $p = fn($q) => $samples[(int)round((count($samples) - 1) * $q)];
                $entry["top$lim"] = ['p50_ms' => round($p(0.50), 4), 'p95_ms' => round($p(0.95), 4)];
            }
            $out[$label] = $entry;
        }
        $out['php_peak_mb'] = round(memory_get_peak_usage(true) / 1048576, 2);
        $out['rss_peak_mb'] = round(rss_peak_kb() / 1024, 2);
        emit($out);
        break;
    }
}
