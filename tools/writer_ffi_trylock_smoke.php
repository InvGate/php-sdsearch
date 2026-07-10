<?php
// Smoke: Writer::try_open returns false when another Writer already holds the lock, true if free.
// Run with: php -n -d extension=$(pwd)/target/release/libsdsearch.so tools/writer_ffi_trylock_smoke.php <index_dir_copy>
$dir = $argv[1] ?? null;
if (!$dir || !is_dir($dir)) { fwrite(STDERR, "usage: ... tools/writer_ffi_trylock_smoke.php <index_dir>\n"); exit(2); }

$a = new SdSearch\Writer();
$okA = $a->try_open($dir);
if ($okA !== true) { fwrite(STDERR, "FAIL: A should take the lock (got ".var_export($okA, true).")\n"); exit(1); }

$b = new SdSearch\Writer();
$okB = $b->try_open($dir);
if ($okB !== false) { fwrite(STDERR, "FAIL: B should see the lock taken → false (got ".var_export($okB, true).")\n"); exit(1); }

unset($a); // releases the lock (Drop)
$c = new SdSearch\Writer();
$okC = $c->try_open($dir);
if ($okC !== true) { fwrite(STDERR, "FAIL: C should take the lock after A releases it (got ".var_export($okC, true).")\n"); exit(1); }

echo "TRYLOCK SMOKE OK\n";
exit(0);
