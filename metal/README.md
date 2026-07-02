# Metal Kernels

Metal kernel sources (`.metal`) are the **source of truth** for GPU execution.
They are embedded into the binary through:

```sh
make embed-kernels
```

That command runs `scripts/embed_kernels.sh` and regenerates
`Sources/DS4Metal/Runtime/KernelSources.swift`, so the runtime does not require a
kernel directory on disk. The same embedded source path works under SwiftPM, the
generated `.xcodeproj`, and a shipped `.app`.

Main files by runtime weight:

- `moe.metal`: MoE matvec kernels for all supported quantization formats.
- `flash_attn.metal`: attention kernels.
- `dense.metal`: dense projection helpers.
- `dsv4_misc.metal`, `dsv4_hc.metal`, `dsv4_kv.metal`, `dsv4_rope.metal`:
  DeepSeek-V4-specific helpers.
- utility kernels for normalization, softmax, argsort, unary operations, and
  related glue.

## Workflow

1. Edit the `.metal` source.
2. Run `make embed-kernels`.
3. Update or add the Swift wrapper under `Sources/DS4Metal/Kernels/` when the
   kernel signature changes.
4. Keep the file order synchronized with `MetalRuntime.kernelFiles`.
