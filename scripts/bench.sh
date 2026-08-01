#!/bin/sh
# Benchmark automatico dei knob prefill/decode: esegue la matrice di run con
# la configurazione veloce di base e raccoglie le righe chiave di ognuna in un
# UNICO report — un comando, nessun copia-incolla di knob.
#
# Uso:
#   scripts/bench.sh <gguf> <file-prompt> [report.txt] [caso1 caso2 ...]
#
# Esempio:
#   scripts/bench.sh ~/Downloads/ds4-main/gguf/DeepSeek-V4-*.gguf \
#                    ~/Downloads/ds4-main/README.md
#
# Senza casi espliciti esegue TUTTA la matrice (~9 run, ~6-7 min l'una).
# Casi disponibili: base hot-eviction reuse-eviction mm union256 chunk1024 route64 nsg2 nsg8 slots24 sharedq4
set -u
GGUF="${1:?uso: bench.sh <gguf> <file-prompt> [report] [casi...]}"
PROMPT="${2:?serve il file prompt (usato come @file)}"
OUT="${3:-bench-report.txt}"
shift 2
[ $# -gt 0 ] && shift   # scarta il nome report se presente
CASES="${*:-base hot-eviction reuse-eviction mm union256 chunk1024 route64 nsg2 nsg8 slots24 sharedq4}"
BIN=.build/release/DS4Demo
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

[ -x "$BIN" ] || { echo "manca $BIN — esegui: swift build -c release"; exit 1; }
: > "$OUT"
echo "bench $(date '+%Y-%m-%d %H:%M') — $(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m), RAM $(($(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824)) GB" | tee -a "$OUT"
echo "gguf: $GGUF" >> "$OUT"
echo "" >> "$OUT"

run_case() {
    name="$1"; shift
    echo "== $name ($*)" | tee -a "$OUT"
    # I knob di base sono la configurazione veloce misurata; gli argomenti
    # extra del caso vengono DOPO e quindi la sovrascrivono.
    env DS4_DIAG=1 DS4_EXPERT_CACHE_SLOTS=16 DS4_EXPERT_PREAD=1 DS4_PREAD_SPLIT=3 \
        DS4_DENSE_STREAM=1 DS4_DENSE_Q4=1 DS4_MLOCK=1 "$@" \
        "$BIN" "$GGUF" 8 "@$PROMPT" > "$TMP" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "  FALLITA (exit $rc) — ultime righe:" | tee -a "$OUT"
        tail -8 "$TMP" | sed 's/^/  /' | tee -a "$OUT"
        echo "" >> "$OUT"
        return
    fi
    grep -E "^DS4Demo: prompt dal file|^DS4Demo: prefill " "$TMP" >> "$OUT"
    awk '/^Profilo prefill/{f=1} f{print "  " $0} f&&/totale/{f=0}' "$TMP" >> "$OUT"
    # Risposta (prova di sanita'/parita'): prima riga generata.
    sed -n 's/^Risposta: \(.*\)/  Risposta: \1/p' "$TMP" | head -1 >> "$OUT"
    grep -E "^DS4Demo: [0-9]+ tokens in " "$TMP" >> "$OUT"
    awk '/^Profilo decode/{f=1} f&&/route\/attn|gather IO|experts|totale|cache expert/{print "  " $0} f&&/totale/{f=0}' "$TMP" >> "$OUT"
    echo "" >> "$OUT"
}

for c in $CASES; do
    case "$c" in
        base)      run_case base ;;
        hot-eviction) run_case hot-eviction DS4_EXPERT_CACHE_HOT_EVICTION=1 ;;
        reuse-eviction) run_case reuse-eviction DS4_EXPERT_CACHE_REUSE_EVICTION=1 ;;
        mm)        run_case mm DS4_PREFILL_MM=1 ;;
        union256)  run_case union256 DS4_PREFILL_UNION=256 ;;
        chunk1024) run_case chunk1024 DS4_PREFILL_CHUNK=1024 ;;
        route64)   run_case route64 DS4_PREFILL_ROUTE_BATCH=64 ;;
        nsg2)      run_case nsg2 DS4_Q8_NSG=2 ;;
        nsg8)      run_case nsg8 DS4_Q8_NSG=8 ;;
        slots24)   run_case slots24 DS4_EXPERT_CACHE_SLOTS=24 ;;
        sharedq4)  run_case sharedq4 DS4_SHARED_Q4=1 ;;
        *)         echo "caso sconosciuto: $c (disponibili: base hot-eviction reuse-eviction mm union256 chunk1024 route64 nsg2 nsg8 slots24 sharedq4)" ;;
    esac
done

echo "Report completo in: $OUT"
