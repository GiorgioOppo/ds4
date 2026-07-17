# Metal Kernels

Metal kernel sources (`.metal`) are the **source of truth** for GPU execution.
The complete runtime/wrapper workflow and numerical-validation policy are in
[`docs/BACKEND-METAL.md`](../docs/BACKEND-METAL.md).
They are embedded into the binary through:

```sh
make embed-kernels
```

That command runs `scripts/embed_kernels.sh` and regenerates
`Sources/DS4Metal/Runtime/Generated/KernelSources.swift`, so the runtime does not require a
kernel directory on disk. The same embedded source path works under SwiftPM, the
generated `.xcodeproj`, and a shipped `.app`.

## Layout

Kernels are grouped by architecture, one directory each:

- [`deepseek/`](deepseek/): the DeepSeek V4 kernels plus the shared generic ops
  (norm, softmax, argsort, copy/cast, …) that the DeepSeek graph drives today.
- [`glm5.2/`](glm5.2/): the GLM 5.2 (`glm-dsa`) kernels.

Metal still compiles ONE library from the concatenation of every file, so
kernel names remain globally unique across directories. The embedded key and
the `MetalRuntime.kernelFiles` entry stay the file basename; the on-disk loader
(`MetalRuntime(metalDir:)`) searches `MetalRuntime.kernelSubdirectories` and
also accepts the flat legacy layout.

Main files by runtime weight:

- `deepseek/moe.metal`: MoE matvec kernels for all supported quantization formats.
- `deepseek/flash_attn.metal`: attention kernels.
- `deepseek/dense.metal`: dense projection helpers.
- `deepseek/dsv4_misc.metal`, `dsv4_hc.metal`, `dsv4_kv.metal`, `dsv4_rope.metal`:
  DeepSeek-V4-specific helpers.
- `glm5.2/glm52.metal`: GLM 5.2 router and compact-DSA primitives.
- utility kernels for normalization, softmax, argsort, unary operations, and
  related glue.

## Workflow

1. Edit the `.metal` source.
2. Run `make embed-kernels`.
3. Update or add the Swift wrapper under `Sources/DS4Metal/Kernels/` when the
   kernel signature changes.
4. Keep the file order synchronized with `MetalRuntime.kernelFiles`.
5. Run the kernel and graph tests described in
   [`Tests/METAL-TESTS.md`](../Tests/METAL-TESTS.md).

## Kernel audit status (reviewed 2026-07-13)

The original low-level audit covered all 19 kernel files and their Swift
dispatch wrappers. The status was rechecked against the current tree on
2026-07-13. Fixed items are retained here so old reports are not mistaken for
open defects:

- **Fixed:** iq2_xxs shared-table loading no longer races or goes out of bounds
  with `nsg=4`.
- **Fixed:** `fp8_kv_quantize` now guards `in_nope` when `n_nope % 64 != 0`.
- **Fixed:** `kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16` now has the barrier
  required before reusing `sb` on the next k-iteration, and its Swift dispatch
  allocates the required 16 KiB of threadgroup memory.
- **Guarded:** router geometry is hardwired to 256 experts and scale 1.5;
  `StreamingDecoder` refuses incompatible shapes before dispatch.

The remaining entries are open but dormant on supported production geometry,
unless a bullet explicitly says otherwise. Re-audit and add tests before making
a currently unreachable combination live:

- **Open/dormant:** `moe.metal`
  `kernel_mul_mv_{table,addr}_q4_K_sum6_f32`: invalid expert id
  in ANY slot does `return` (dst never written, other slots' contributions
  lost) instead of `continue` like the q2_K variant; the addr variant also
  misses the `addr == 0` check the q2_K one has.
- **Open/dormant:** `dense.metal` matvec implementations (and `moe.metal`
  mul_mv): the last partial threadgroup reads all nr0 weight rows unguarded —
  OOB reads if outDim is not a multiple of nr0(*nsg). All production dims are
  multiples.
- **Open/dormant:** `dense.metal` `switch (args.nr0)` without default: nr0
  outside {2,4} is a silent no-op (stale dst).
- **Open/dormant:** `flash_attn.metal` vec kernel `ss4` zero-init writes 512 B
  per simdgroup into a 256 B region (overlap is zeros-before-barrier today —
  benign, but any layout change turns it into corruption). The non-vec kernels
  (not dispatched from Swift) use `ushort` for `iq1` → overflow past 65535 rows.
- **Open/bounded:** `dsv4_misc.metal` top-6 bitonic is not index-stable on exact
  probability ties (C reference: lowest index wins); hash_mode routing exists
  in the kernel but is never driven by the Swift engine despite nHashLayer=3 in
  the shape table — verify against ds4.c before enabling.
- **Open/dormant:** `dsv4_hc.metal` hcExpand4/hcWeightedSum token strides
  (`nb_post1`, `nb_comb2`, `nb_w1`) are only correct for nTokens == 1 when the packed
  split buffer (96 B rows) is bound — every production call site passes 1.
  The comb orientation (nb_comb0=4/nb_comb1=16, i.e. transposed vs the
  kernel's dst/src naming) matches the Swift-side doc but should be verified
  against ds4.c once available.
- **Open/production long-context risk:** `dsv4_rope.metal` computes theta in
  f32 without 2π reduction and the library compiles with fast-math — phase
  error grows with pos (~2-3% on the fastest pairs at pos ≈ 300k). Gradual
  quality degradation on very long contexts, no NaNs. A fix (fmod reduction /
  precise::sincos) changes numerics vs the C reference — needs on-device parity
  evaluation first.
- **Open/dormant:** `GraphContext` binds activations with `offset: 0`
  (byteOffset honored only for matmul weights): passing an offset view (rowView,
  staging slices) to rmsNorm/add/swiglu would silently read the wrong data.
- **Open/bounded by callers:** `GraphCompressor` writes the emitted compressed
  row at `cache[comp.count]` without a maxComp bound check (callers cap pos at
  maxKeys today).
