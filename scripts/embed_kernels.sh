#!/bin/sh
# Regenerate Sources/DS4Metal/Runtime/KernelSources.swift from metal/*.metal — embeds the
# kernel sources in the binary so MetalRuntime() needs no on-disk kernel folder.
# metal/ stays the source of truth; this file is generated. Keep the order in
# sync with MetalRuntime.kernelFiles.
set -e
cd "$(dirname "$0")/.."
out=Sources/DS4Metal/Runtime/KernelSources.swift
order="flash_attn dense moe dsv4_hc unary dsv4_kv dsv4_rope dsv4_misc argsort cpy concat get_rows sum_rows softmax repeat glu norm bin set_rows"
{
  echo "// AUTO-GENERATED from metal/*.metal — do not edit by hand."
  echo "// Regenerate with: make embed-kernels  (scripts/embed_kernels.sh)."
  echo "// Embeds the kernel sources in the binary so the Metal runtime needs no"
  echo "// on-disk kernel folder (works in SwiftPM, the .xcodeproj, and a shipped .app)."
  echo ""
  echo "extension MetalRuntime {"
  echo "    static let embeddedKernels: [String: String] = ["
  for name in $order; do
    echo "        \"$name\": ###\"\"\""
    cat "metal/$name.metal"
    echo "\"\"\"###,"
  done
  echo "    ]"
  echo "}"
} > "$out"
echo "wrote $out"
