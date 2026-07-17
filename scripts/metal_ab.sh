#!/bin/bash
# Deterministic process-level A/B for one DS4 Metal runtime knob.
#
# It runs the same release DS4Demo twice, forces greedy/non-speculative decode,
# records bounded full-vocabulary logits after timing has ended, and compares
# both numerical correctness and prefill/decode throughput.
set -euo pipefail

usage() {
    echo "uso: scripts/metal_ab.sh <gguf> <prompt-file> <DS4_KNOB> [base-value] [candidate-value] [max-new] [out-dir]" >&2
    echo "esempio: scripts/metal_ab.sh model.gguf prompt.txt DS4_ADAPTIVE_SPLITK 0 1 8" >&2
    exit 64
}

[ "$#" -ge 3 ] || usage
CALLER_PWD="$PWD"
GGUF="$1"
PROMPT="$2"
KNOB="$3"
BASE_VALUE="${4:-0}"
CANDIDATE_VALUE="${5:-1}"
MAX_NEW="${6:-8}"
OUT_ARG="${7:-metal-ab-$(date '+%Y%m%d-%H%M%S')}"

case "$GGUF" in /*) ;; *) GGUF="$CALLER_PWD/$GGUF" ;; esac
case "$PROMPT" in /*) ;; *) PROMPT="$CALLER_PWD/$PROMPT" ;; esac
case "$OUT_ARG" in /*) OUT="$OUT_ARG" ;; *) OUT="$CALLER_PWD/$OUT_ARG" ;; esac

case "$KNOB" in
    DS4_[A-Z0-9_]*) ;;
    *) echo "knob non valido: $KNOB (atteso DS4_[A-Z0-9_]+)" >&2; exit 64 ;;
esac
case "$MAX_NEW" in *[!0-9]*|"") echo "max-new deve essere un intero positivo" >&2; exit 64 ;; esac
[ "$MAX_NEW" -ge 2 ] || { echo "max-new deve essere >=2" >&2; exit 64; }
[ -f "$GGUF" ] || { echo "GGUF non trovato: $GGUF" >&2; exit 66; }
[ -f "$PROMPT" ] || { echo "prompt file non trovato: $PROMPT" >&2; exit 66; }
command -v python3 >/dev/null 2>&1 || { echo "python3 non trovato" >&2; exit 69; }

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"
BIN="$ROOT/.build/release/DS4Demo"
COMPARE="$ROOT/scripts/metal_ab_compare.py"
mkdir -p "$OUT"

# `xcode-select` may point at the standalone CommandLineTools, whose Swift/Metal
# SDK combination is too old for this package. Respect an explicit caller
# override; otherwise prefer the full Xcode toolchain when it is installed.
if [ -z "${DEVELOPER_DIR:-}" ] && [ -d /Applications/Xcode.app/Contents/Developer ]; then
    export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
fi

if [ "${DS4_AB_SKIP_BUILD:-0}" != "1" ]; then
    echo "[build] swift build -c release --product DS4Demo"
    swift build -c release --product DS4Demo
fi
[ -x "$BIN" ] || { echo "binario release mancante: $BIN" >&2; exit 69; }

WARMUP="${DS4_AB_WARMUP:-2}"
case "$WARMUP" in *[!0-9]*|"") echo "DS4_AB_WARMUP deve essere un intero" >&2; exit 64 ;; esac
if [ "$WARMUP" -ge "$MAX_NEW" ]; then WARMUP=$((MAX_NEW - 1)); fi
TRACE_FRAMES="${DS4_AB_TRACE_FRAMES:-$((MAX_NEW + 1))}"
case "$TRACE_FRAMES" in *[!0-9]*|"") echo "DS4_AB_TRACE_FRAMES deve essere un intero" >&2; exit 64 ;; esac
[ "$TRACE_FRAMES" -ge 1 ] || { echo "DS4_AB_TRACE_FRAMES deve essere >=1" >&2; exit 64; }
if [ "$TRACE_FRAMES" -gt 64 ]; then TRACE_FRAMES=64; fi
DIAG="${DS4_AB_DIAG:-0}"
USAGE_FILE="${DS4_AB_USAGE_FILE:-off}"
ATOL="${DS4_AB_ATOL:-1e-4}"
RTOL="${DS4_AB_RTOL:-1e-4}"
ORDER="${DS4_AB_ORDER:-baseline-first}"

{
    printf 'date: %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')"
    printf 'git: %s\n' "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    printf 'developer_dir: %s\n' "${DEVELOPER_DIR:-$(xcode-select -p 2>/dev/null || echo unknown)}"
    printf 'host: %s\n' "$(uname -m)"
    printf 'cpu: %s\n' "$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
    printf 'ram_bytes: %s\n' "$(sysctl -n hw.memsize 2>/dev/null || echo unknown)"
    printf 'gguf: %s\n' "$GGUF"
    printf 'prompt: %s\n' "$PROMPT"
    printf 'knob: %s\nbase: %s\ncandidate: %s\n' "$KNOB" "$BASE_VALUE" "$CANDIDATE_VALUE"
    printf 'max_new: %s\nwarmup: %s\ntrace_frames: %s\norder: %s\n' "$MAX_NEW" "$WARMUP" "$TRACE_FRAMES" "$ORDER"
    printf '\nInherited DS4 environment (controlled values below override it):\n'
    env | LC_ALL=C sort | grep '^DS4_' || true
} > "$OUT/meta.txt"

run_case() {
    local name="$1"
    local value="$2"
    local prefix="$OUT/$name"
    local log_file="$OUT/$name.log"
    echo "[$name] $KNOB=$value"
    env \
        "DS4_AB_TRACE=$prefix" \
        "DS4_AB_TRACE_FRAMES=$TRACE_FRAMES" \
        "DS4_DIAG=$DIAG" \
        "DS4_WARMUP=$WARMUP" \
        "DS4_USAGE_FILE=$USAGE_FILE" \
        DS4_DEMO_TEMPERATURE=0 \
        DS4_DEMO_TOP_K=0 \
        DS4_DEMO_TOP_P=1 \
        DS4_DEMO_MIN_P=0 \
        DS4_DEMO_REPEAT_PENALTY=1 \
        DS4_SPEC_K=0 \
        "$KNOB=$value" \
        "$BIN" "$GGUF" "$MAX_NEW" "@$PROMPT" > "$log_file" 2>&1 || {
            local rc=$?
            echo "[$name] FALLITO (exit $rc); ultime righe:" >&2
            tail -20 "$log_file" >&2
            exit "$rc"
        }
    grep -E "^DS4Demo: prefill |^DS4Demo: REGIME |^DS4Demo: [0-9]+ tokens in |^DS4Demo: A/B logit trace scritta" "$log_file" || true
}

case "$ORDER" in
    baseline-first)
        run_case baseline "$BASE_VALUE"
        run_case candidate "$CANDIDATE_VALUE"
        ;;
    candidate-first)
        run_case candidate "$CANDIDATE_VALUE"
        run_case baseline "$BASE_VALUE"
        ;;
    *) echo "DS4_AB_ORDER deve essere baseline-first o candidate-first" >&2; exit 64 ;;
esac

set +e
python3 "$COMPARE" \
    --baseline "$OUT/baseline" \
    --candidate "$OUT/candidate" \
    --baseline-log "$OUT/baseline.log" \
    --candidate-log "$OUT/candidate.log" \
    --label "$KNOB: $BASE_VALUE -> $CANDIDATE_VALUE" \
    --atol "$ATOL" --rtol "$RTOL" > "$OUT/report.md"
COMPARE_RC=$?
set -e

printf '\n'
sed -n '1,160p' "$OUT/report.md"
printf '\nReport completo: %s\n' "$OUT/report.md"
printf 'Log e trace: %s\n' "$OUT"
exit "$COMPARE_RC"
