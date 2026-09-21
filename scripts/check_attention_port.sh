#!/bin/bash
# Link against an existing SwiftPM build; never build the package or load a model.
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    echo "usage: bash scripts/check_attention_port.sh <SwiftPM-products-dir> [output-dir] [compile|validate|benchmark]" >&2
    echo "example: bash scripts/check_attention_port.sh /private/tmp/ds4-attention-build/release /private/tmp/ds4-attention-checks validate" >&2
    exit 64
fi

CHECK_SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
CHECK_PRODUCTS="$(CDPATH= cd -- "$1" && pwd)"
CHECK_OUTPUT="${2:-/private/tmp/ds4-attention-checks}"
CHECK_MODE="${3:-validate}"
case "$CHECK_MODE" in compile|validate|benchmark) ;; *) echo "unknown mode: $CHECK_MODE" >&2; exit 64 ;; esac

mkdir -p "$CHECK_OUTPUT/module-cache"
CHECK_OUTPUT="$(CDPATH= cd -- "$CHECK_OUTPUT" && pwd)"
CHECK_OBJECTS=()
if [ -e "$CHECK_PRODUCTS/Modules/DS4Metal.swiftmodule" ]; then
    CHECK_MODULES="$CHECK_PRODUCTS/Modules"
    [ -d "$CHECK_PRODUCTS/DS4Core.build" ] || { echo "DS4Core objects missing" >&2; exit 66; }
    [ -d "$CHECK_PRODUCTS/DS4Metal.build" ] || { echo "DS4Metal objects missing" >&2; exit 66; }
    while IFS= read -r -d '' CHECK_OBJECT; do
        CHECK_OBJECTS+=("$CHECK_OBJECT")
    done < <(find "$CHECK_PRODUCTS/DS4Core.build" "$CHECK_PRODUCTS/DS4Metal.build" -type f -name '*.o' ! -name '*.swiftmodule.o' -print0)
elif [ -d "$CHECK_PRODUCTS/DS4Metal.swiftmodule" ] && [ -d "$CHECK_PRODUCTS/DS4Core.swiftmodule" ] && \
     [ -f "$CHECK_PRODUCTS/DS4Metal.o" ] && [ -f "$CHECK_PRODUCTS/DS4Core.o" ]; then
    # SwiftPM's Swift Build backend emits target-level relocatable objects
    # beside architecture-specific module directories in Products/Release.
    CHECK_MODULES="$CHECK_PRODUCTS"
    CHECK_OBJECTS=("$CHECK_PRODUCTS/DS4Core.o" "$CHECK_PRODUCTS/DS4Metal.o")
else
    echo "DS4Core/DS4Metal modules and objects missing from products directory" >&2
    exit 66
fi
[ "${#CHECK_OBJECTS[@]}" -gt 0 ] || { echo "No SwiftPM object files found" >&2; exit 66; }

swiftc -O -parse-as-library -I "$CHECK_MODULES" \
    -module-cache-path "$CHECK_OUTPUT/module-cache" \
    "$CHECK_SCRIPT_DIR/check_attention_port.swift" "${CHECK_OBJECTS[@]}" \
    -framework Metal -framework MetalPerformanceShaders \
    -o "$CHECK_OUTPUT/check_attention_port"
echo "Compiled $CHECK_OUTPUT/check_attention_port"
[ "$CHECK_MODE" != compile ] || exit 0

CHECK_BENCH_ARGS=()
if [ "$CHECK_MODE" = benchmark ]; then CHECK_BENCH_ARGS=(--benchmark --pairs 8 --repeats 8); fi
# Bash 3.2 treats an empty array as unset under nounset; the guarded expansion
# also keeps validation-only mode working with macOS's bundled /bin/bash.
# GraphContext caches NSG once per process. Separate processes are required to
# exercise the actual production knobs without exposing mutable runtime internals.
env DS4_Q8_NSG=4 DS4_DENSE_Q4_NSG=4 \
    "$CHECK_OUTPUT/check_attention_port" --suite decode ${CHECK_BENCH_ARGS[@]+"${CHECK_BENCH_ARGS[@]}"} \
    | tee "$CHECK_OUTPUT/decode-nsg4.log"
"$CHECK_OUTPUT/check_attention_port" --suite prefill ${CHECK_BENCH_ARGS[@]+"${CHECK_BENCH_ARGS[@]}"} \
    | tee "$CHECK_OUTPUT/prefill.log"
# Alternative NSGs validate parity only: do not multiply timing runs and thermal
# drift. Default NSG=4 above is the sole timed decode configuration.
for CHECK_NSG in 1 2 8; do
    env DS4_Q8_NSG="$CHECK_NSG" DS4_DENSE_Q4_NSG="$CHECK_NSG" \
        "$CHECK_OUTPUT/check_attention_port" --suite decode \
        | tee "$CHECK_OUTPUT/decode-nsg$CHECK_NSG.log"
done
echo "Logs: $CHECK_OUTPUT"
