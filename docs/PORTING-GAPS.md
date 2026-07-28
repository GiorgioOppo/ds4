**English** | [Italiano](PORTING-GAPS.it.md)

# Porting gaps vs. `antirez/ds4` (runtime / Metal)

This document is the turnkey plan for the runtime gaps between the Swift port
and the upstream C reference that **cannot be validated without Apple hardware**
(they add or change Metal kernels and the decode/prefill path, so logits-parity
must be checked on a Mac). The offline tooling gaps (GGUF writer, offline
requantizer) are already implemented in pure Swift — see
`Sources/DS4Core/Formats/GGUF/GGUFWriter.swift` and
`Sources/DS4Core/Formats/Quantization/GGUFRequantizer.swift`.

Out of scope by design (see [`UPSTREAM-SYNC.md`](UPSTREAM-SYNC.md)): CUDA/ROCm
backends, tensor-parallel + RDMA (`ds4_tp.c`), the upstream MTP speculative
path, `ds4-eval`, and the terminal agent.

Each gap below lists: the exact upstream reference, the Swift touch-points, a
step plan, and the validation gate. Upstream paths are relative to a clone of
`https://github.com/antirez/ds4.git`.

---

## Gap 1 — Routed-expert quant formats Q8_K and Q8_0

### Status today
`MoEQuant` (`Sources/DS4Metal/Model/Quantization/MoEQuant.swift`) accepts only
`q4_K (12)`, `q2_K (10)`, `iq2_xxs (16)` for routed experts, and the loader
**hard-rejects** anything else
(`Sources/DS4Metal/Backends/DeepSeekV4/Model/GGUFWeights.swift:63`
`validateRuntimeLayout`). A GGUF whose routed experts are Q8_K or Q8_0 fails to
load. `get_rows` exists only in `f32/f16/i32` (`metal/deepseek/get_rows.metal`).

> Priority note: no model in `ModelCatalog` currently ships Q8_K/Q8_0 routed
> experts, so this is low practical payoff today. Do it if/when such a GGUF is a
> target, or to remove the on-device `Q8_0→Q4_K` requant for dense weights.

### Upstream reference
- `ds4.c` `layer_routed_moe_batch` (~line 10967): `routed_q8_0` and `routed_q8_k`
  branches — per-format activation quantization + expert-grouped matvec.
- `metal/moe.metal`: `kernel_mul_mv_id_*` family; `metal/get_rows.metal`:
  `kernel_get_rows_q8_0_f32`, `kernel_get_rows_q4_K_f32`.
- Q8_K block layout: 256 elems / 292 bytes (already in
  `GGUF.typeTable[15]`); Q8_0: 32 elems / 34 bytes (`typeTable[8]`).

### Swift touch-points
1. `MoEQuant.swift`: add `case q8_K` (and optionally `q8_0`) with `kernel`,
   `blockBytes` (292 / 34), `nr0`, `threadgroupBytes`, and `from(ggufType:)`
   (15 / 8).
2. `metal/deepseek/moe.metal`: port `kernel_mul_mv_id_q8_K_f32` (and
   `_q8_0` if needed) from upstream `moe.metal`. Regenerate the embedded
   kernels: `make embed-kernels` (writes
   `Sources/DS4Metal/Runtime/Generated/KernelSources.swift`).
3. `GGUFWeights.swift:validateRuntimeLayout`: widen the accepted set to match
   `MoEQuant.from`.
4. Confirm `GGUFWeights.gatherExperts` sizing is already `blockBytes`-driven
   (it is) so no per-type hardcoding remains.

### Steps
1. Produce a small Q8_K-routed fixture GGUF from an existing one with the new
   offline requantizer: `DS4Demo requantize in.gguf q8k.gguf q4_k>q8_k@ffn`
   (extend `QuantEncode` if a needed source type isn't dequantizable yet).
2. Add the kernel; wire `MoEQuant`; relax the validator.
3. Add a byte-exact matvec fixture test (mirror `QuantEncodeTests` /
   `scripts/quant-fixtures/`).

### Validation gate
Logits-parity of the Q8_K fixture vs. the C reference within the existing
tolerance; no regression on `q4_K/q2_K/iq2_xxs` decode + batched prefill.

---

## Gap 2 — DeepSeek V4 Pro Q4 split-load (two-shard package)

### Status today
Pro Q2 single-file runs; the Pro Q4 two-shard package is **download-only** and
cannot be selected as a local model (`docs/ARCHITETTURE-SUPPORTATE.md`; the
GUI/`Browse` path rejects a lone shard). This is loader/model-management work,
largely NOT new Metal shaders.

> **Progress:**
> - `GGUFShardSet` (pure Swift, tested): unions N shards by name + per-layer
>   shard routing (`shard(forLayer:)`, `shard(owning:)`, `primary`).
> - `StreamingDecoder.fromGGUFShards` (DS4Metal, additive): builds the decoder
>   routing each layer to its owning shard, so **`GGUFWeights` is reused
>   unchanged** (the split is layer-disjoint, so a whole layer resolves to one
>   shard). This is the simple resident path (analog of `fromGGUF`).
>
> - `DS4Demo` CLI (additive): a comma-separated path list
>   (`DS4Demo shardA.gguf,shardB.gguf …`) opens the split and runs it through
>   `fromGGUFShards` — so the whole chain is **executable end-to-end** for
>   on-device validation without touching the GUI.
>
> What remains needs on-device validation: (a) the expert-streaming shard
> variants (mapped/cached-expert pools span shards); (b) GUI `InferenceService`
> wiring (its init is built around one `GGUFModel`; add a shard path that keeps
> the single-file path unchanged); (c) catalog/GUI recognizing the two files as
> one selectable model.
>
> Design note: a naive "read protocol" over `GGUFWeights` was rejected — weight
> loading uses per-tensor raw access (`mapBase`, `uncachedFD`, `path`) that is
> per-shard, so per-layer shard routing (above) is the correct, lower-risk seam;
> `GGUFWeights` never had to change.

### Upstream reference
- Upstream assembles multi-file GGUFs; see the shard/loader handling in `ds4.c`
  model open path and `download_model.sh` (the Pro Q4 shard targets).
- `docs/ARCHITETTURE-SUPPORTATE.md` already documents the intended boundary.

### Swift touch-points
- `Sources/DS4Core/Formats/GGUF/GGUFShardSet.swift` — **DONE**: unified tensor
  directory over N mmaps (`find`/`tensorData` route to the owning shard;
  metadata first-shard-wins).
- `Sources/DS4Metal/Backends/DeepSeekV4/Model/GGUFWeights.swift`: tensor byte
  access must resolve through the shard that owns each tensor — accept a
  `GGUFShardSet` (or a small protocol both `GGUFModel` and `GGUFShardSet`
  satisfy) instead of a single `GGUFModel`.
- `Sources/DS4Engine/ModelManagement/Catalog/ModelCatalog.swift`: mark the Pro
  Q4 package `runnable` once assembly exists; the GUI scan/`Browse` filter must
  recognize the shard set as one model.

### Steps
1. ~~Shard manifest / naming~~ — the package is two layer-range GGUFs
   (`…Layers00-30.gguf`, `…Layers-31-output.gguf`), tensor names disjoint.
2. ~~Multi-shard open with a unified tensor index~~ — done in `GGUFShardSet`.
3. Route weight access (`GGUFWeights`, expert gather) through the shard set —
   simplest via a read protocol implemented by both `GGUFModel` and
   `GGUFShardSet` (`findTensor`, `tensorData`, metadata accessors).
4. Flip catalog `runnable`; teach the GUI scanner the shard grouping (recognize
   the two Pro Q4 files as one selectable model).

### Validation gate
Pro Q4 loads and produces logits matching the C reference; the distributed Pro
path (already protocol-tested) gets its pending numerical validation
(see `UPSTREAM-SYNC.md` open item `f2d701a`).

---

## Gap 3 — Server: mixed prefill+decode batching (continuous batching)

### Status today
The server **serializes** requests behind `RequestGate`
(`Sources/DwarfStar/Features/Server/Concurrency/RequestGate.swift`) on a single
mutable engine — one generation at a time. Upstream serves multiple sessions by
fusing a prefill batch with in-flight decodes in one layer FFN encode.

> Highest-value gap, but the most invasive: it changes the engine's batch
> geometry and the server's scheduler. Do it last, behind an env flag, with the
> serialized path kept as fallback.

### Upstream reference
- `ds4.c` `metal_graph_encode_layer_ffn_batch` (~line 28412): takes
  `n_tokens` (prefill) **and** `decode_items`/`decode_count`; "only the routed
  expert dispatch is shared between the two arithmetic paths."
- `ds4_server.c`: `server_slot`, `batched_mode`, `active_generations`,
  `server_prefill_quantum`, `server_session_sync` — the slot scheduler.
- Env knobs: `DS4_CUDA_MIXED_PREFILL_DECODE`, `DS4_CUDA_PREFILL_PIPELINE_*`
  (CUDA-specific; the Metal analog is the shared FFN encode above).

### Swift touch-points
- `Sources/DS4Metal/Backends/DeepSeekV4/Decode/Prefill/StreamingDecoder+Prefill.swift`
  (`batchedExpertLayer`, `PrefillStage`): extend the batched FFN to accept a
  decode tail in rows `[n, n+decodeCount)` sharing the routed-expert gather —
  the direct analog of the C encode.
- `Sources/DS4Engine/Inference/Service/*`: a slot scheduler that admits new
  prefills onto a running decode batch (prefill quantum, per-session KV).
- `Sources/DwarfStar/Features/Server/*`: replace the single-flight `RequestGate`
  with the scheduler; keep serialized mode as the default/fallback.

### Steps
1. Land the engine primitive first: a batched layer that runs `n` prefill rows +
   `k` decode rows sharing one routed-expert union gather. Unit-test it against
   the sum of the separate paths (numerically identical).
2. Add the per-session KV + slot scheduler in the engine service.
3. Wire the server to the scheduler behind `DS4_SERVER_BATCH` (default off).

### Validation gate
Aggregate throughput rises with concurrent sessions; single-session output is
byte-identical to the serialized path; KV isolation between sessions verified.

---

## Gap 4 — Laguna S 2.1 decoder (and optional DFlash speculation)

### Status today
The complete Laguna frontend is in the port and unit-tested: architecture
registration/detection (`ModelArchitectureID.laguna`), exact geometry
validation (`LagunaConfiguration`), tokenizer (`LagunaTokenizer` with the
newline pre-split), native chat/tool protocol (`LagunaChatRenderer`,
`LagunaToolCodec`, reference sampling defaults), tensor schema for the two
published recipes plus the mixed Q2_K/Q3_K file (`LagunaTensorSchema`), the
runtime registration (`LagunaBackendDefinition`) and the download catalog
(`LagunaModelCatalog`, DFlash accessory). `LagunaRuntimeGate.enabled` is off:
a Laguna GGUF is recognized, validated and refused with a distinct error.

### Upstream reference (branch `laguna-s2.1`, head `448d569`)
- `ds4.c`: `DS4_SHAPE_LAGUNA_S21`, `weights_bind_laguna_layer`,
  `layer_routed_moe_batch` Laguna paths, `ds4_laguna_layer_is_swa` — GQA
  attention with per-layer 48/72 heads, gated attention (`attn_gate`),
  Q/K RMS-norm per head-dim, full attention with 64 YaRN RoPE dims on every
  fourth block, 512-token sliding window with 128 RoPE dims at base 10000 on
  the others, one dense leading block, 256 routed experts (top-10, gating
  func 2, scale 2.5) plus one shared expert.
- `metal/laguna.metal` (~1.2k lines) and the `ds4_metal.m` Laguna driver
  paths: the kernels to port into `metal/` and re-embed with
  `make embed-kernels`.
- DFlash (optional, separate GGUF): `metal/dflash.metal`, the
  `DS4_DFLASH_*` constants and the `--dflash` verification pipeline; greedy
  speculation that never changes accepted tokens, with automatic fallback
  when it is not paying.

### Already landed (validated without hardware)
- CPU oracles of the whole decode path
  (`Backends/Laguna/Reference/LagunaLayerReference.swift`): per-head
  norm/RoPE with YaRN, gated GQA attention over the F16 ring, 10-expert
  router, FFN blocks — with hand-computable unit tests.
- Kernel port: `metal/laguna/laguna.metal` (verbatim from upstream plus a
  local `block_q6_K`), wired into `MetalRuntime.kernelFiles`,
  `kernelSubdirectories` and `scripts/embed_kernels.sh`;
  `KernelSources.swift` regenerated.
- Validation wrappers (`Backends/Laguna/Kernels/LagunaKernels.swift`):
  `lagunaQKHeadRMSNormRope`, `lagunaStoreKV`, `lagunaAttentionDecode`, with
  GPU/CPU parity tests (skip without Metal) covering both block kinds, ring
  wrap and the split reduction.
- `LagunaWeightMap` (payload-free directory with the detected quantization
  layout retained).

### Also landed (compiles-on-Mac pending — written without a toolchain)
- First-cut resident engine
  (`Backends/Laguna/Engine/LagunaResidentModel.swift`): full residency of
  the official Q8_0-signal + Q4_K recipe, per-layer F16 ring caches
  (SWA=512 / full=context), per-token graph dispatching the shared GLM
  primitives (`kernel_glm52_rms_norm_f32`, `kernel_glm52_matvec_pair_sg`,
  `kernel_glm52_router_select` with 10 experts, `glm52_moe` matvecs) plus
  the Laguna kernels; host-side router readback per MoE layer;
  token-by-token prefill. Legacy and mixed Q2_K/Q3_K files are refused with
  distinct errors until their matvec paths are wired.
- `RuntimeBackendKind.laguna` + selector routing behind `LagunaRuntimeGate`,
  the DS4Demo dispatch branch (greedy loop, `DS4_LAGUNA_LAYERS` truncation)
  and a backend-neutral GUI refusal message in `InferenceService`.
- Engine tests: Q8_0 row-dequant parity with the shared encoder
  (device-free) and an opt-in real-weights smoke test
  (`DS4_LAGUNA_GGUF`).

### Remaining steps (need a Mac at every one)
1. **Compile.** The engine was written without a Swift toolchain: fix
   whatever `swift build` reports first.
2. **Run the parity tests on hardware** (`LagunaKernelsTests`, then the
   `DS4_LAGUNA_GGUF` smoke test with a truncated stack): they gate the rest.
3. **End-to-end logits parity** against the reference C engine on the real
   Q4_K_M GGUF (the branch's `ds4` with `--temp 0`), then tokenizer golden
   parity on the real vocabulary.
4. Performance pass: batched prefill and chained decode (drop the per-phase
   waits), then the Q2_K/Q3_K routed matvecs for the mixed file and the
   legacy-recipe paths (`kernel_laguna_q6_K_matmul_f32` is already ported).
5. Catalog: pin byte counts and SHA-256 digests for the three artifacts,
   then flip `LagunaRuntimeGate.enabled` (`ModelDownloaderTests` enforces
   "runnable ⇒ pinned digest").

### Validation gate
End-to-end logits parity against the reference C engine on the real
Q4_K_M GGUF (and the mixed Q2_K/Q3_K), tokenizer golden parity on the real
vocabulary, then flip `LagunaRuntimeGate.enabled`. DFlash afterwards, greedy
token-exactness against ordinary decoding with a fixed verifier width.

---

## Suggested order

1. **Gap 3 engine primitive** (shared prefill+decode FFN) — highest value; start
   with the engine-only, unit-testable core.
2. **Gap 2** (Pro Q4 split-load) — unlocks a shipped artifact; mostly Swift.
3. **Gap 4** (Laguna decoder) — the frontend, schema and catalog are already
   merged; the decoder is a self-contained resident engine.
4. **Gap 1** (Q8_K/Q8_0 routed) — do when a target GGUF needs it.

Every step here requires a Mac (Metal) for its validation gate; this repo's
Linux CI can build/test only the pure-Swift `DS4Core` surface.

## From upstream drift (see UPSTREAM-SYNC.md, reviewed `80ebbc3..0a7ad77`)

Two additional in-scope items surfaced by re-running the upstream comparison,
both needing an on-device code read:

- **`519c4d8` — expert-cache / sampling correctness.** Upstream makes computed
  logits independent of the SSD streaming expert-cache budget (+ a too-small-cache
  warning and a sampling fix). Verify the Swift `ExpertSlotCache`/streaming path
  holds the same "logits ⟂ budget" invariant.
- **`427e281` — Metal kernel resync.** A large upstream kernel-optimization pass;
  the port's kernels (ported from `metal/*.metal`) are now behind. Evaluate
  porting the changes and regenerate the embedded kernels.

Also: `36cd0ca` (native Metal session batching) is now the Metal reference for
Gap 3, and `005afed` (GLM 5.2 + quality fixtures) supports GLM parity
certification.
