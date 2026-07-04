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

## Known latent hazards (kernel audit 2026-07-04)

Full low-level audit of all 20 kernel files against the Swift dispatch
wrappers. Two live bugs were FIXED (iq2_xxs shared-table loader racing/OOB
with nsg=4 — hit every token on iq2_xxs models; fp8_kv_quantize missing the
in_nope guard for n_nope % 64 != 0) and one hard guard added (the router
kernel hardcodes 256 experts / 1.5 scale: StreamingDecoder now refuses other
shapes loudly). The following are DORMANT defects in kernels or parameter
combinations that no production dispatch currently reaches — fix them before
wiring any of these paths:

- `moe.metal` `kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16`: writes `sb` before
  the first barrier of each k-iteration (races the previous iteration's up
  MMA) and its final staging needs 16 KB of threadgroup memory (loop uses 6).
- `moe.metal` `kernel_mul_mv_{table,addr}_q4_K_sum6_f32`: invalid expert id
  in ANY slot does `return` (dst never written, other slots' contributions
  lost) instead of `continue` like the q2_K variant; the addr variant also
  misses the `addr == 0` check the q2_K one has.
- `dense.metal` matvec impls (and moe.metal mul_mv): the last partial
  threadgroup READS all nr0 weight rows unguarded — OOB reads if outDim is
  not a multiple of nr0(*nsg). All production dims are multiples.
- `dense.metal` `switch (args.nr0)` without default: nr0 outside {2,4} is a
  silent no-op (stale dst).
- `flash_attn.metal` vec kernel `ss4` zero-init writes 512 B per simdgroup
  into a 256 B region (overlap is zeros-before-barrier today — benign, but
  any layout change turns it into corruption). The non-vec kernels (not
  dispatched from Swift) use `ushort` for `iq1` → overflow past 65535 rows.
- `dsv4_misc.metal` top-6 bitonic is not index-stable on exact probability
  ties (C reference: lowest index wins); hash_mode routing exists in the
  kernel but is never driven by the Swift engine despite nHashLayer=3 in the
  shape table — verify against ds4.c before enabling.
- `dsv4_hc.metal` hcExpand4/hcWeightedSum token strides (`nb_post1`,
  `nb_comb2`, `nb_w1`) are only correct for nTokens == 1 when the packed
  split buffer (96 B rows) is bound — every production call site passes 1.
  The comb orientation (nb_comb0=4/nb_comb1=16, i.e. transposed vs the
  kernel's dst/src naming) matches the Swift-side doc but should be verified
  against ds4.c once available.
- `dsv4_rope.metal`: theta is computed in f32 without 2π reduction and the
  library compiles with fast-math — phase error grows with pos (~2-3% on the
  fastest pairs at pos ≈ 300k). Gradual quality degradation on very long
  contexts, no NaNs. A fix (fmod reduction / precise::sincos) changes
  numerics vs the C reference — needs on-device parity evaluation first.
- `GraphContext` binds activations with `offset: 0` (byteOffset honored only
  for matmul weights): passing an offset view (rowView, staging slices) to
  rmsNorm/add/swiglu would silently read the wrong data.
- `GraphCompressor` writes the emitted compressed row at `cache[comp.count]`
  without a maxComp bound check (callers cap pos at maxKeys today).
