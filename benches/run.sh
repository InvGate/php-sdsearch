#!/usr/bin/env bash
#
# sdsearch benchmark orchestrator: runs the full {engine × size × workload} matrix, each
# measurement in a FRESH process (so peak-RSS is not cross-contaminated), collects one JSON
# line per run into benches/results.jsonl, and renders benches/RESULTS.md.
#
# Engines:
#   native   — the Rust core via examples/bench_engine (reports EXACT heap_peak_kb + rss).
#   sdsearch — our PHP extension (SdSearch\Writer / SdSearch\Engine); RSS is the comparable mem.
#   zend     — Zend_Search_Lucene (requires ZEND_LUCENE_PATH). Pure-PHP, so it is SLOW to index;
#              guarded by a timeout (+ an address-space cap on very large sizes). If a guarded
#              run does not finish it is recorded as "skipped", not fatal.
#
# Workloads: rebuild (build N docs from the KB base), churn (delete 1% + add 1%), search
# (many/few/none result classes). See benches/README.md for the exact corpus + query definitions.
#
# Usage:
#   ./benches/run.sh [--quick] [--sizes "1000 10000 ..."] [--workloads "rebuild churn search"]
#                    [--engines "native sdsearch zend"] [--iters N] [--no-build] [--append]
#
# Env knobs (defaults in brackets):
#   ZEND_LUCENE_PATH        path to a ZSL library/ root; unset => zend engine is skipped.
#   BENCH_ZEND_MAX_DOCS     [100000] zend is skipped for sizes strictly above this.
#   BENCH_ZEND_TIMEOUT      [900]    per-zend-run wall-clock budget (seconds); over => skipped.
#   BENCH_ZEND_MEM_MB       [0]      address-space cap (MB) applied to zend runs at/above
#                                    BENCH_ZEND_ULIMIT_FROM; 0 disables the cap.
#   BENCH_ZEND_ULIMIT_FROM  [500000] size at/above which BENCH_ZEND_MEM_MB is enforced.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

SIZES="1000 10000 50000 100000 500000"
WORKLOADS="rebuild churn search"
ENGINES="native sdsearch zend"
ITERS=50
DO_BUILD=1
APPEND=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)      SIZES="1000 10000"; shift ;;
        --sizes)      SIZES="$2"; shift 2 ;;
        --workloads)  WORKLOADS="$2"; shift 2 ;;
        --engines)    ENGINES="$2"; shift 2 ;;
        --iters)      ITERS="$2"; shift 2 ;;
        --no-build)   DO_BUILD=0; shift ;;
        --append)     APPEND=1; shift ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

BENCH_ZEND_MAX_DOCS="${BENCH_ZEND_MAX_DOCS:-100000}"
BENCH_ZEND_TIMEOUT="${BENCH_ZEND_TIMEOUT:-900}"
BENCH_ZEND_MEM_MB="${BENCH_ZEND_MEM_MB:-0}"
BENCH_ZEND_ULIMIT_FROM="${BENCH_ZEND_ULIMIT_FROM:-500000}"

RESULTS="$REPO/benches/results.jsonl"
NATIVE_BIN="$REPO/target/release/examples/bench_engine"
EXT_SO="$REPO/target/release/libsdsearch.so"
COMPARE="$REPO/tools/bench_compare.php"
SCRATCH="${TMPDIR:-/tmp}/sdsearch_bench_$$"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

log() { echo "[bench] $*" >&2; }

# ---- build (release) ----
if [[ "$DO_BUILD" == 1 ]]; then
    log "building native example + PHP extension (release)…"
    cargo build -p sdsearch-core --release --example bench_engine >&2 || { log "native build failed"; exit 1; }
    cargo build -p sdsearch-php --release >&2 || { log "extension build failed"; exit 1; }
fi
[[ -x "$NATIVE_BIN" ]] || { log "missing $NATIVE_BIN (drop --no-build or build it)"; exit 1; }

# ---- environment detection ----
# Only add -d extension if the .so is not already loaded by php.ini (avoids a "already loaded"
# warning that would pollute stdout).
PHP_EXT=()
if php -m 2>/dev/null | grep -qi '^sdsearch$'; then
    log "sdsearch extension already loaded by php.ini"
elif [[ -f "$EXT_SO" ]]; then
    PHP_EXT=(-d "extension=$EXT_SO")
    log "loading sdsearch extension from $EXT_SO"
fi

ZEND_OK=0
if [[ -n "${ZEND_LUCENE_PATH:-}" && -f "${ZEND_LUCENE_PATH%/}/Zend/Search/Lucene.php" ]]; then
    ZEND_OK=1
    log "Zend oracle: $ZEND_LUCENE_PATH"
else
    log "ZEND_LUCENE_PATH unset/invalid — zend engine will be recorded as skipped"
fi

[[ "$APPEND" == 1 ]] || : > "$RESULTS"

# appends a raw JSON line (already emitted by a bench) to results.jsonl.
emit_line() { printf '%s\n' "$1" >> "$RESULTS"; }
# appends a synthetic skip record.
emit_skip() {
    emit_line "{\"engine\":\"$1\",\"workload\":\"$2\",\"n\":$3,\"status\":\"skipped\",\"reason\":\"$4\"}"
    log "SKIP $1/$2 n=$3 ($4)"
}
# runs a command, extracts its LAST JSON object line ({...}) and records it; logs on failure.
capture() {
    local engine="$1" workload="$2" n="$3"; shift 3
    local out rc line
    out="$("$@" 2>/dev/null)"; rc=$?
    line="$(printf '%s\n' "$out" | grep -E '^\{.*\}$' | tail -1)"
    if [[ $rc -ne 0 || -z "$line" ]]; then
        emit_skip "$engine" "$workload" "$n" "exit=$rc/no-json"
        return
    fi
    emit_line "$line"
    log "OK   $engine/$workload n=$n"
}

php_run() { php "${PHP_EXT[@]}" "$COMPARE" "$@"; }

# runs a zend measurement under the timeout (+ optional address-space cap); records skip on any
# non-zero/timeout/OOM exit. Returns the captured JSON via capture().
zend_run() {
    local workload="$1" n="$2"; shift 2   # remaining args: extra bench_compare args
    local cap_kb=0
    if [[ "$BENCH_ZEND_MEM_MB" -gt 0 && "$n" -ge "$BENCH_ZEND_ULIMIT_FROM" ]]; then
        cap_kb=$((BENCH_ZEND_MEM_MB * 1024))
    fi
    capture zend "$workload" "$n" bash -c '
        [[ "$1" -gt 0 ]] && ulimit -v "$1"
        shift
        exec timeout "$1"s "${@:2}"
    ' _ "$cap_kb" "$BENCH_ZEND_TIMEOUT" php "$COMPARE" zend "$workload" "$n" "$@"
}

for n in $SIZES; do
    log "===== size N=$n ====="

    for engine in $ENGINES; do
        case "$engine" in
        native)
            # rebuild/churn run TWICE: a TIME pass (tracking off → honest ms/rss, the engine floor)
            # and a HEAP pass (tracking on → accurate heap, ms taxed & ignored). search runs the
            # TIME pass only (its heap is not a headline). The report merges the two passes.
            for w in $WORKLOADS; do
                case "$w" in
                    search)
                        capture native search "$n" env BENCH_TRACK_HEAP=0 "$NATIVE_BIN" search "$n" "$ITERS" ;;
                    *)
                        capture native "$w" "$n" env BENCH_TRACK_HEAP=0 "$NATIVE_BIN" "$w" "$n"   # time
                        capture native "$w" "$n" env BENCH_TRACK_HEAP=1 "$NATIVE_BIN" "$w" "$n" ;; # heap
                esac
            done
            ;;

        sdsearch)
            pidx="$SCRATCH/sd_$n"
            need_persist=0
            for w in $WORKLOADS; do [[ "$w" == churn || "$w" == search ]] && need_persist=1; done
            [[ "$need_persist" == 1 ]] && php_run sdsearch build "$n" 0 "$pidx" >/dev/null 2>&1
            for w in $WORKLOADS; do
                case "$w" in
                    rebuild) capture sdsearch rebuild "$n" php "${PHP_EXT[@]}" "$COMPARE" sdsearch rebuild "$n" ;;
                    churn)   capture sdsearch churn   "$n" php "${PHP_EXT[@]}" "$COMPARE" sdsearch churn "$n" 0 "$pidx" ;;
                    search)  capture sdsearch search  "$n" php "${PHP_EXT[@]}" "$COMPARE" sdsearch search "$n" "$ITERS" "$pidx" ;;
                esac
            done
            rm -rf "$pidx"
            ;;

        zend)
            if [[ "$ZEND_OK" != 1 ]]; then
                for w in $WORKLOADS; do emit_skip zend "$w" "$n" "ZEND_LUCENE_PATH unset"; done
                continue
            fi
            if [[ "$n" -gt "$BENCH_ZEND_MAX_DOCS" ]]; then
                for w in $WORKLOADS; do emit_skip zend "$w" "$n" "n>BENCH_ZEND_MAX_DOCS=$BENCH_ZEND_MAX_DOCS"; done
                continue
            fi
            zidx="$SCRATCH/zend_$n"
            need_persist=0
            for w in $WORKLOADS; do [[ "$w" == churn || "$w" == search ]] && need_persist=1; done
            if [[ "$need_persist" == 1 ]]; then
                # build under the same guard; if it fails, churn/search for this size are skipped.
                if ! zend_build_out="$(timeout "${BENCH_ZEND_TIMEOUT}s" php "$COMPARE" zend build "$n" 0 "$zidx" 2>/dev/null)"; then
                    log "zend build n=$n failed/timed out — churn/search skipped"
                fi
            fi
            for w in $WORKLOADS; do
                case "$w" in
                    rebuild) zend_run rebuild "$n" ;;
                    churn)   [[ -d "$zidx" ]] && zend_run churn "$n" 0 "$zidx" || emit_skip zend churn "$n" "no prebuilt index" ;;
                    search)  [[ -d "$zidx" ]] && zend_run search "$n" "$ITERS" "$zidx" || emit_skip zend search "$n" "no prebuilt index" ;;
                esac
            done
            rm -rf "$zidx"
            ;;

        *) log "unknown engine: $engine"; exit 2 ;;
        esac
    done
done

log "rendering report → benches/RESULTS.md"
php "$REPO/tools/bench_report.php" "$RESULTS" > "$REPO/benches/RESULTS.md" \
    && log "done. results: benches/results.jsonl  report: benches/RESULTS.md" \
    || log "report rendering failed (results.jsonl is intact)"
