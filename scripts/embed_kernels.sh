#!/bin/sh
# Regenerate Sources/DS4Metal/Runtime/Generated/KernelSources.swift from the vendored kernels —
# embeds the kernel sources in the binary so MetalRuntime() needs no on-disk
# kernel folder. metal/ stays the source of truth; this file is generated.
# Kernels are grouped by architecture: metal/common/ (shared quant tables and
# helpers — FIRST in the order so its symbols are visible to every backend),
# metal/deepseek/ (DeepSeek V4 plus the shared generic ops) and metal/glm5.2/
# (GLM 5.2). The embedded key stays the file basename. Keep the order in sync
# with MetalRuntime.kernelFiles.
set -e
cd "$(dirname "$0")/.."
out=Sources/DS4Metal/Runtime/Generated/KernelSources.swift
# glm52_quant DEVE precedere glm52_moe (helper di dot condivisi); gli altri
# file GLM sono auto-contenuti. laguna_kv DEVE precedere laguna_attention
# (struct args del prefill condivisa).
order="quant_tables flash_attn dense moe dsv4_hc unary dsv4_kv dsv4_rope dsv4_misc glm52_router glm52_quant glm52_kv glm52_indexer glm52_attention glm52_moe glm52_rope glm52_misc laguna_quant laguna_rope laguna_kv laguna_attention argsort cpy concat get_rows sum_rows softmax repeat glu norm bin set_rows"

# Resolve a kernel name to its file: architecture subdirectories first, then
# the flat legacy location.
kernel_path() {
  for candidate in "metal/common/$1.metal" "metal/deepseek/$1.metal" "metal/glm5.2/$1.metal" "metal/laguna/$1.metal" "metal/$1.metal"; do
    if [ -f "$candidate" ]; then
      echo "$candidate"
      return 0
    fi
  done
  echo "embed_kernels.sh: kernel $1.metal not found under metal/" >&2
  return 1
}

{
  echo "// AUTO-GENERATED from the vendored metal/ kernels — do not edit by hand."
  echo "// Regenerate with: make embed-kernels  (scripts/embed_kernels.sh)."
  echo "// Embeds the kernel sources in the binary so the Metal runtime needs no"
  echo "// on-disk kernel folder (works in SwiftPM, the .xcodeproj, and a shipped .app)."
  echo ""
  echo "extension MetalRuntime {"
  echo "    static let embeddedKernels: [String: String] = ["
  for name in $order; do
    path=$(kernel_path "$name")
    echo "        \"$name\": ###\"\"\""
    cat "$path"
    echo "\"\"\"###,"
  done
  echo "    ]"
  echo "}"
} > "$out"
echo "wrote $out"
