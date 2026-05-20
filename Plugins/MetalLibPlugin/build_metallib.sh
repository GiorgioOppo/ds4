#!/bin/bash
# Compiles every *.metal file in $1 into a single default.metallib placed in $2.
# Invoked by MetalLibPlugin as a prebuildCommand; its output directory is
# scanned by SwiftPM and the metallib is bundled as a resource of DeepSeekKit.
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: build_metallib.sh <kernels-dir> <output-dir>" >&2
    exit 2
fi

kernels_dir=$1
output_dir=$2

mkdir -p "$output_dir"

# Stage .air files in a scratch subdir so SwiftPM doesn't pick them up as
# resources (only default.metallib should be bundled).
air_dir="$output_dir/.air"
rm -rf "$air_dir"
mkdir -p "$air_dir"

shopt -s nullglob
metal_files=( "$kernels_dir"/*.metal )
if [ "${#metal_files[@]}" -eq 0 ]; then
    echo "no .metal files in $kernels_dir" >&2
    exit 1
fi

air_files=()
# Source embedding + line tables make the Metal Shader Debugger work
# (Xcode → Debug → Metal → step through kernels), but the App Store
# validator rejects any .metallib that ships debug info or source
# ("Found Metal shader source code", error 90659 / Transporter
# 90258). SwiftPM `prebuildCommand` plugins do NOT reliably inherit
# Xcode's `CONFIGURATION` env var across xcodebuild/archive flows —
# we used to strip on `CONFIGURATION=Release` and it slipped through
# anyway on at least one archive that reached the App Store. Default
# is now "strip" so every release path is safe by construction;
# developers who want the shader debugger opt in explicitly with
# `DEEPSEEK_METALLIB_DEBUG=1` (e.g. via a scheme env var, or
# `DEEPSEEK_METALLIB_DEBUG=1 swift build`).
metal_debug_flags=()
if [ "${DEEPSEEK_METALLIB_DEBUG:-0}" = "1" ]; then
    metal_debug_flags=( -gline-tables-only -frecord-sources )
fi

for metal in "${metal_files[@]}"; do
    name=$(basename "$metal" .metal)
    air="$air_dir/$name.air"
    xcrun -sdk macosx metal "${metal_debug_flags[@]}" -c "$metal" -o "$air"
    air_files+=( "$air" )
done

xcrun -sdk macosx metallib "${air_files[@]}" -o "$output_dir/default.metallib"

rm -rf "$air_dir"
