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
exit(0);
