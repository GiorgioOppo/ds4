# Performance

Where time is currently spent, what's known to be slow, and how to
profile. Open this when:

- "Why is my first token taking 30 seconds?"
- "I want to optimize a specific kernel."
- "What's the ceiling on this hardware?"

## At a glance

| Component | Status | Bottleneck | Speedup potential |
|---|---|---|---|
| Cold first forward | SSD-bound | Page-in of working set (~13 GB) | 0× (hardware-limited) |
| Warm decode steady-state | Compute-bound | Scalar tiled GEMM, no simdgroup_matrix | 5-10× via simdgroup matrix |
| Sparse attention | Compute-bound | One thread per (b,m,h), device-mem accumulator | 3-5× via Flash tiling |
| MoE dispatch | Mixed | Host-side `prepare` + N tiny dispatches | 2× via persistent kernel |
| KV cache writes | Negligible | Already blit copies (memory-mapped) | — |
| RoPE / RMSNorm / Sampler | Negligible | Already optimal-ish | — |
| Pipeline state cache | Setup overhead | `makePipeline` calls per init | 10-50 ms per forward call saved |
| Converter (parallel) | I/O-bound on SSD write | `DispatchQueue.concurrentPerform` saturates CPU | 0× (disk-limited after fix) |

## 1. Where time goes

A single decode token in V4-Flash on M3 Max touches:
- 43 layers × (MLA + MoE) = 86 sublayer forwards
- Each MLA: ~9 GEMMs (wq_a, wq_b, wkv, wo_a-einsum, wo_b, plus the
  attn itself)
- Each MoEFFN: 2 active experts × 3 GEMMs each (w1/w2/w3) + 1 shared
  expert × 3 = 9 GEMMs

So ~(9 + 9) × 43 ≈ 770 GEMM kernel launches per token, plus
non-GEMM ops (RMSNorm × 4 per layer, RoPE × 2 per layer, sparse_attn
× 1, HC.pre × 2, HC.post × 2, etc.).

With naive scalar tiled GEMM, each launch is dominated by FLOP count
× clock cycle, not by overhead. Once we move to `simdgroup_matrix`
the launches will be FLOP-saturated by the matrix instructions
themselves.

## 2. Bottlenecks today

### GEMM kernels are scalar-tiled

`Sources/DeepSeekKit/Kernels/gemm_bf16.metal` does a classic 16×16
threadgroup tile with a scalar inner-product accumulator. On M3 Max:

- Theoretical BF16 peak with `simdgroup_matrix`: ~10 TFLOPS
- Our scalar tile reaches: ~1-2 TFLOPS

That's a ~5-10× gap for any forward that's GEMM-dominated, which is
all of them.

The same applies to FP8 / FP4 GEMMs (`fp8_gemm.metal`, `fp4_gemm.metal`)
— scalar inner loop with shader-side dequant.

### Sparse attention is serial per (b, m, h)

`Sources/DeepSeekKit/Kernels/sparse_attn.metal` launches one thread per
output cell `(b, m, h)`, which then loops over `K` (= win + compressed
≈ 128 + 32) topk indices, reading 512-element KV rows from device
memory each time, accumulating into device-memory output.

For V4-Flash decode (B=1, M=1, H=64): only 64 threads run in parallel.
Way under-utilizes the GPU (M3 Max has ~5000 ALUs).

Replacement plan: FlashAttention-style tiling with `simdgroup_matrix`
for the Q·K and softmax·V inner GEMMs, threadgroup-shared KV pages,
warp-level online softmax. ~3-5× faster in steady state.

### MoE dispatch overhead

`Sources/DeepSeekKit/Layers/MoE.swift:170-200` does:
1. Run `Gate` on GPU.
2. CPU/GPU sync to read indices + weights back to host.
3. Host-side `MoEDispatch.prepare` builds permutation tables (alloc +
   loop in Swift).
4. For each active expert (top-2 × N_tokens ⇒ 2-N unique experts),
   dispatch 3 separate Linear forwards.
5. CPU-side blit of expert outputs into the right slot.
6. Scatter back into dense output.

The CPU/GPU sync + tiny per-expert dispatches are pure overhead at
batch=1 decode. A persistent kernel that runs all active experts in
one launch + does the scatter inline would save ~tens of ms per token.

### Pipeline state creation

Several Layers create their `MTLComputePipelineState` lazily on each
instance init (rather than caching by name globally). For deeply
nested per-token pipelines this adds CPU overhead on the first call
of each new instance.

Fix: a library-wide cache keyed by `(kernel_name, constantValues)`.
Save once at app start, look up on each `init`. Estimated 10-50 ms
saved on the first token.

**Status: implementato.** `Device.shared.makePipeline(_:)` ora cacha
internamente (`Sources/DeepSeekKit/Device.swift`); la variante
`makePipeline(_:constants:)` (`Sources/DeepSeekKit/PipelineTuning.swift`)
estende il caching ai kernel con function constants (ActQuant,
MoE.Gate, HCSinkhorn). Le pipeline sono ora condivise fra istanze
dello stesso layer con gli stessi parametri.

## 3. Hardware sizing

Expected single-token decode time for **V4-Flash, current naive
kernels**:

| Hardware | RAM | Working set fits? | Decode time/token |
|---|---|---|---|
| M2 Max 64 GB | 64 GB | with SSD paging | 5-10 s (cold), 1-3 s (warm) |
| M3 Max 128 GB | 128 GB | mostly | 1-3 s warm |
| M3 Ultra 192 GB | 192 GB | yes | 200-500 ms warm |
| M3 Ultra 512 GB | 512 GB | yes, with headroom | 200-500 ms warm |

With the optimization pass (simdgroup matrix GEMM + FlashAttention
tiling + persistent MoE dispatch + pipeline cache), expected:

| Hardware | Expected after opts |
|---|---|
| M3 Max 128 GB | 100-200 ms warm |
| M3 Ultra 192 GB | 30-80 ms warm |

V4-Pro is a different beast — 900 GB at FP4+FP8 won't fully fit on
any current Mac. Even M3 Ultra 512 GB will SSD-page constantly. Decode
times will be SSD-bound, ~seconds per token.

## 4. How to profile

### CPU side: Instruments → Time Profiler

```bash
xcrun xctrace record --template "Time Profiler" \
    --launch .build/release/deepseek -- \
    /Volumes/DATA/V4-Flash-bf16 "ciao" --mode chat --max-tokens 1
```

Then open the `.trace` file in Instruments. Useful for:

- Pipeline state creation hotspots
- Swift allocation churn
- CPU/GPU sync waits (look for `wait` in the call tree)

### GPU side: Xcode → GPU Frame Capture

```bash
# Launch from Xcode with "Capture GPU Frame" enabled, OR via the env var:
MTL_HUD_ENABLED=1 .build/release/deepseek ...
```

Better: use Xcode's GPU debugger directly. Set a breakpoint after the
prefill forward, capture one decode call. You'll see:

- Per-pipeline-state dispatch count
- Per-encoder timing
- Buffer dependencies (which kernels wait for which)
- Idle GPU bubbles

### Manual timing in code

For ad-hoc measurements:

```swift
import Metal

let cmd = Device.shared.queue.makeCommandBuffer()!
cmd.addCompletedHandler { commandBuffer in
    let gpuStart = commandBuffer.gpuStartTime
    let gpuEnd = commandBuffer.gpuEndTime
    print("GPU \(String(format: "%.3f", (gpuEnd - gpuStart) * 1000)) ms")
}
// ... build encoders, dispatch ...
cmd.commit()
cmd.waitUntilCompleted()
```

Wall-clock around a forward:

```swift
let t0 = ContinuousClock().now
let logits = model.forward(inputIds: [ids], startPos: 0)
print("prefill: \(t0.duration(to: .now).seconds * 1000) ms")
```

### Memory pressure tracking

```bash
# In another terminal while deepseek is running:
sudo memory_pressure -l warn
top -o mem
vm_stat 1
```

Apple Activity Monitor's Memory tab is the easiest read of
"compressed", "swap used", and "wired" memory in real time.

## 5. Optimization roadmap

Listed in rough order of impact-per-effort:

### simdgroup_matrix BF16 GEMM (high impact, medium effort)

Replace the 16×16 scalar tile in `gemm_bf16.metal` with
`simdgroup_matrix<bfloat>` instructions. M3+ has matrix multiply
accelerators directly in the SIMD group. ~5-10× faster matmul.

Spec:
- Input: A `[M, K]`, B `[N, K]^T` both bf16
- Output: C `[M, N]` f32
- Tile: 8×8 simdgroup_matrix, 4× unroll for cache friendliness

Estimated effort: 1-2 days for a working version, another 1-2 days
for tuning.

### FlashAttention tiling for sparse_attn (high impact, high effort)

Replace the per-thread serial loop with FlashAttention-style tiled
attention:

- Block K dim into tiles of 64
- Load Q tile once per block (threadgroup memory)
- Stream KV tiles, accumulate online softmax in threadgroup memory
- Spill `O` tile back to global memory at the end of each K block
- Use `simdgroup_matrix` for the Q·K and S·V matmuls

Estimated effort: 3-5 days.

### Persistent MoE dispatch (medium impact, medium effort)

One kernel that:
1. Reads indices + weights from device memory
2. Per-token, runs the top-K active experts inline (each expert is a
   `simdgroup_matrix`-fused SwiGLU)
3. Scatters weighted outputs back to the dense output

Eliminates host-side `prepare` + per-expert kernel launches. Trickier:
needs to manage per-expert weight pointers and handle the case where
the same expert is active for many tokens.

Estimated effort: 2-3 days.

### Pipeline state cache (low impact, low effort) — implementato

Singleton dict `[PipelineCacheKey: MTLComputePipelineState]` popolato
lazy in `Device.shared.makePipeline(_:)` /
`makePipeline(_:constants:)`. La chiave include il nome del kernel
e — per i kernel con function constants — un wrapper Hashable
(`PipelineConstants`) dei `(type, raw-bytes, index)` con cui si
ricostruisce `MTLFunctionConstantValues`.

Vedi `Sources/DeepSeekKit/Device.swift` +
`Sources/DeepSeekKit/PipelineTuning.swift`. ActQuant, MoE.Gate e
HCSinkhorn sono migrati al routing centralizzato.

### KV cache pool (low impact in current single-batch workflow)

Today every layer allocates its KV cache at construction. For a
multi-session server (one Mac serves many requests), pooling caches
between requests avoids the allocation overhead.

Not on the critical path for the current CLI.

### MLA multi-token forward with `startPos > 0` (high impact for tool-heavy turns)

Today `MLA.callAsFunction` enforces `precondition(S == 1)` when
`startPos > 0` — the decode path is single-token only. After a
tool call the chat builds an `<eos><tool_outputs>…<Assistant>`
delta of ~30-100 tokens and feeds them back through the model. The
current loop calls the forward N times (~12-15 s on a 50k-context
chat) instead of one multi-token forward (~1-2 s).

Refactor:
1. Drop the precondition (or hide it behind an opt-in
   `Transformer.allowMultiTokenWithOffset` flag for safety).
2. Add a branch in MLA's KV-blit for `S > 1, startPos > 0`: writes
   `S` consecutive rows starting at `slot = startPos % windowSize`,
   wrap-around when needed.
3. `Compressor.forwardDecode` with `S > 1, startPos > 0`: an inner
   loop of `S` steps is safe and cheap; a vectorised version is
   higher-risk.
4. Verify `SparseAttention.apply` produces the right causal mask
   for `S > 1, startPos > 0` (likely already correct, but this is
   the part to validate first).

Estimated effort: 1-2 days. Highest single perf win for
tool-heavy chats and for the delegation loop (since each
delegation's tool round-trips also pay this cost).

### KV snapshot/restore (already shipped)

`Transformer.snapshotKVCache` / `restoreKVCache` (B2) wrap each
sub-agent delegation in `ChatStore.runSubAgentToCompletion`. Cost:
one `memcpy` per layer's KV buffer per begin (~few hundred MB on
V4-Flash with `windowSize 4096`), released on end. Saves the cold
re-prefill the host would otherwise pay when the sub-agent run
trashes the GPU's KV cache.

No optimisation backlog here — it's not on a hot path
(delegations are a sub-second-amortised user action).

### KV cache persistence to disk (B3, not started)

Goal: a project-attached chat reopens after an app restart
without paying the prefill of the project context again.

Design space: file format (parallel to safetensors?), eviction
policy, model-fingerprint invalidation (a different model
loaded would render the cache garbage), shard size /
write-amplification trade-off.

Estimated effort: 1-2 weeks for a sound first cut.

## 6. Cold vs warm timing

The first forward is fundamentally slower than steady-state because
the OS hasn't paged anything in yet.

To measure cold-start:

```bash
# Purge OS page cache (requires sudo, may need a restart-of-process to flush):
sudo purge

# Then run, time the first token only:
time .build/release/deepseek <model_dir> "ciao" --mode chat --max-tokens 1
```

To measure warm (post-cache):

```bash
# Run once to populate the cache, then time the second invocation:
.build/release/deepseek <model_dir> "warmup" --mode chat --max-tokens 1 > /dev/null
time .build/release/deepseek <model_dir> "real" --mode chat --max-tokens 10
```

Cold time is SSD-bound (~7 GB/s sequential read on Apple internal SSD).
Warm time is compute-bound.

For benchmark consistency:
- Run the same prompt 3 times, take the minimum (warm).
- Note RAM headroom (`vm_stat`) — paging activity will tank
  measurements.

## 7. Converter performance

Recent optimization (commit `739fbf2`):

- Pre-computed lookup tables (`e4m3LUT[256]`, `e2m1LUT[16]`,
  `e8m0LUT[256]`) replace per-element bit twiddling in
  `Sources/converter/main.swift:240-260`.
- `DispatchQueue.concurrentPerform(iterations: outDim)` parallelizes
  the row loop across all CPU cores.
- Per-block scale lookup hoisted out of the inner loop (1× per 128 or
  32 elements).

Result: bottleneck moves from CPU dequant to SSD write throughput.

Throughput on M3 Max + external Thunderbolt SSD: ~1-2 GB/s sustained
write, total conversion ~5-15 minutes for V4-Flash (vs ~30-90 minutes
single-threaded before).

To verify your conversion is CPU-bound vs disk-bound:

```bash
# In a separate terminal while converter runs:
top -o cpu
# If CPU stays near 100% across all cores → CPU-bound (room to optimize further)
# If CPU bounces around 30% with cores idle → disk-bound (you're at SSD ceiling)
iostat -d 1
# Watch the kB/s output for your save-path's volume.
```

## 8. Sources

- `Sources/DeepSeekKit/Kernels/gemm_bf16.metal` — current GEMM (scalar)
- `Sources/DeepSeekKit/Kernels/sparse_attn.metal` — current serial sparse_attn
- `Sources/DeepSeekKit/Layers/MoE.swift:130-200` — MoE host orchestration
- `Sources/converter/main.swift:260-380` — parallel converter fusion
- [ROADMAP.md](ROADMAP.md) — broader feature roadmap

Apple references:
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) — simdgroup_matrix in §8.
- WWDC 2023 "Optimize Metal compute pipelines" — pipeline state and
  function constant best practices.
