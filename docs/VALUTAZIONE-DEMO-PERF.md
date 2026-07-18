# Demo performance evaluation (DS4Demo)

> **Historical benchmark log, not a reference for defaults.** The measurements
> in this document are snapshots from July 5 and 8, 2026 on a MacBook Pro M1
> Pro with 16 GB of RAM and a DeepSeek V4 Flash IQ2_XXS/Q2_K GGUF (43 layers,
> 256 experts, top-6). Configurations change between rows: slot cache 16/20/24,
> partial or extended Q4, bundle and MetalIO are always stated in the context
> of each test. The values do not describe the current GUI preset and are not
> universal results. For the current operating parameters use the
> [Configuration Reference](../README.md#configuration-reference); for a new
> measurement also record the GGUF hash, build version, memory pressure and
> warm/cold cache state.

This document records where time and memory went during that optimization
cycle and the decisions taken from the measurements recorded in the
repository (`Sources/DwarfStar/Features/Benchmark/README.md` and
`Sources/DS4Demo/README.md`).

## 1. The picture in one line

Decode is **I/O-bound on the SSD**: each token moves ~0.6–1.8 GB of experts
(gather) plus — with `DS4_DENSE_STREAM` and without Q4 — up to ~6.2 GB of
streamed dense weights. GPU compute and command buffer synchronization are
secondary costs. Prefill, on the other hand, is already amortized layer-major
and improves with prompt length (4.6 → ~8 tok/s from 64 to 3k tokens).

## 2. Decode: per-token budget

Path: `StreamingDecoder.forward()` → for each of the 43 layers
`runLayer()` (route+attn → selection readback → shared FFN async → gather →
routed FFN) → `outputHead()`. Reference:
`Sources/DS4Metal/Backends/DeepSeekV4/Decode/Execution/StreamingDecoder.swift`.

| Phase (profile) | What it moves/does | Typical cost | Bottleneck |
|---|---|---|---|
| `gather IO` | 6 experts × ~6.9 MB × 43 layers = **~1.78 GB/token** cold; ~0.6 GB/token with a warm slot cache | 200–600 ms | SSD parallel-random bandwidth (~ceiling measured by the DIAG) |
| dense stream (inside `route/attn`) | ~145 MB/layer × 43 ≈ **6.2 GB/token** without `DS4_DENSE_Q4`; ~1.6 GB/token with Q4+resident shared Q4 | overlapped one layer ahead; shows up when the SSD is contended with the gather | SSD sequential bandwidth |
| `route/attn` (compute) | dense projections + flash-attn + router | tens of ms with resident dense weights; **2.4 s/token measured** when the dense weights re-read from a degraded page cache (the pathology that `DENSE_STREAM`/`MLOCK` cure) | RAM/compressor |
| synchronization | ~3 command buffers per layer ≈ **~130 commit+wait/token** (measurable with the demo's DIAG probe in `Sources/DS4Demo/Command/main.swift`) | ~10–25 ms/token at ~100 µs/cb | CPU↔GPU round-trip latency |
| `output head` | Q8 matvec of ~560 MB | ~10–20 ms resident+`MLOCK`; **235–260 ms measured** when compressed/remapped | RAM vs memory compressor |
| `embed` | 1 row of ~8 KB (staging) | negligible | — |

Steady-state result observed in one of the historical configurations:
**~2.5 tok/s** at 4–8k context; ~1.3 tok/s with the window configured at 104k
(pressure of the pre-allocated KV on the 16 GB). This is not the throughput of
the current preset.

Two structural properties to keep in mind:

- **Router selection returns to the CPU at every layer** (readback after the
  route commit): the CPU needs it to issue the experts' `pread`s. As long as
  the gather is done by the CPU, the layer cannot be fully GPU-driven — the
  ~130 syncs/token limit is architectural, not a bug.
- **The only I/O–compute overlap in decode is the shared FFN** (async commit
  before the gather, in `Decode/Execution/StreamingDecoder.swift`). Layer i's
  gather cannot overlap layer i−1's compute because it depends on the route of
  that same layer i; the only alternative is speculative prediction from the
  usage prior (`DS4_PREFETCH_EXPERTS`, today opt-in and explicitly risky).

## 3. Prefill: per-chunk budget

Layer-major path (`prefill()` → `prefillRange()` → `batchedExpertLayer()`):
each layer's weights are loaded **once per chunk** (default 512 tokens) and
applied to all tokens in the chunk.

- Fixed cost per chunk: re-reading the dense weights of ALL layers (~6 GB with
  `DENSE_STREAM`) → **~12 MB/token at chunk 512**, halvable with
  `DS4_PREFILL_CHUNK=1024` (+~160 KB/token of transient activations).
- Gather: union of the experts per group (cap `DS4_PREFILL_UNION`, baseline
  value in these tests 192 — at 64 it read ~1.7 GB/token, measured). Group
  g+1's I/O runs in the background during group g's FFNs (`PrefillGather`).
- Sync: the batched phase A (32 routes per command buffer) and phase B at one
  command buffer per group have already removed the historical ~22k
  syncs/chunk.
- Measured: 4.6 tok/s at 64 tokens → ~8 tok/s at 3k tokens; the fixed cost
  amortizes with the prompt.

## 4. Memory: where the GBs go

| Item | Size | Notes |
|---|---|---|
| Expert slot cache | 6.9 MB/slot/layer wired → **S=16 ≈ 4.7 GB** across 43 layers | historical base configuration, not the current default; usage-driven redistribution under a fixed budget |
| Resident Q4 dense (`DS4_DENSE_Q4`) | ~1.4 GB (+ shared with `DS4_SHARED_Q4`) | removes ~4.6 GB/token from the SSD stream |
| Resident output head (with `DENSE_STREAM`) | ~560 MB | `DS4_MLOCK` needed to keep it out of the compressor |
| Dense staging ring | ~300 MB (2 slots) / +150 MB with `DS4_DENSE_AHEAD=2` | instead of ~6 GB resident |
| Total `DS4_MLOCK` | ~3.3 GB pinned at defaults | the macOS compressor re-reads unpinned buffers at ~2.4 GB/s |
| Transient prefill | union 192 × ~7 MB × 2 (pipeline) ≈ **~2.7 GB** + ~80 MB (`PREFILL_MM`) | lower `DS4_PREFILL_UNION` on tight machines |
| Raw KV | lazy (zero-fill-on-demand); `DS4_RAW_RING` makes it constant | ring in Metal shared memory, not on-disk KV; the footprint without the ring tracks the tokens actually generated |

In the 16-slot baseline the budget is contended: slot cache + resident Q4 +
mlock ≈ 7 GB wired before the KV even starts — this is why the 104k-context
bench drops to 1.3 tok/s.

In the raw-ring path, the wrap of the 128-row window is materialized with a
single F32→F16 GPU dispatch. Split-K counts both raw and compressed rows and
uses exactly `min(32, max(1, ceil(totalRows/32)))`: the 128→129 transition
therefore uses 4→5 workgroups, not the previous 4→8 rounding. To isolate this
policy in an A/B, use `DS4_ADAPTIVE_SPLITK=0` as a fixed-depth-32 control.

## 5. Historical reproduction runbook (on the Mac)

The demo is already instrumented. This command reproduces the 16-slot baseline
of the measurements below; it does not apply the current GUI preset:

```sh
# base + full diagnostics (SSD ceiling, command buffer probe, per-phase profile)
DS4_DIAG=1 DS4_EXPERT_CACHE_SLOTS=16 DS4_DENSE_STREAM=1 DS4_MLOCK=1 \
  swift run -c release DS4Demo model.gguf 48 "Tell me the history of Rome."

# A/Bs that decide the next moves (one knob at a time, same usage file):
#  1. DS4_DENSE_Q4=1 [+DS4_SHARED_Q4=1]  -> how much route/attn and the total drop
#  2. DS4_EXPERT_BUNDLE=1                -> gather bandwidth vs ceiling (verdict in the DIAG)
#  3. DS4_EXPERT_CACHE_SLOTS=8/12/16     -> hit rate vs RAM
#  4. DS4_PREFILL_MM=1 and DS4_PREFILL_CHUNK=1024 with a ~3k-token @file prompt
scripts/bench.sh model.gguf prompt.txt report.txt   # automatic matrix with a single report
```

Read: `gather IO` MB/token and effective bandwidth vs ceiling (verdict printed
by the DIAG), cache hit rate, `route/attn` before/after Q4, STEADY-STATE tok/s
(the first 4 tokens are excluded from the profile with `DS4_DIAG`).

## 6. Hypotheses evaluated in the historical cycle

This was the work list at the opening of the campaign, not the current
backlog. The experimental conclusions in sections 8 and 9 supersede the
initial estimates reported here.

1. **MTP speculative decoding** (historical estimate: 2–4×, high effort, not
   implemented in the current runtime).
   Decode pays GB/token *regardless* of how many tokens it verifies: with the
   MTP weights (the DIAG already checks whether they are in the GGUF,
   `mtpReport`) a draft of N tokens verified in one batched step amortizes the
   entire dense+expert stream over N tokens. The building blocks already
   exist: the prefill's batched phase A is effectively a multi-token
   verification step, and `batchedExpertLayer` knows how to deduplicate the
   expert union of multiple tokens. It is the only lever that attacks the
   fundamental constraint (bytes/token from the SSD) instead of shaving it.
   The DIAG can only detect the presence of the tensors: there is no MTP
   loader or execution path, so this was a design hypothesis.

2. **Defend the measured profiles in the CLI demo too** (yield: 2× vs a naive
   run, minimal effort). At the time, the GUI baseline used 16 slots,
   `DENSE_STREAM`, `MLOCK` and Q4; the demo starts with the engine's bare
   defaults, and a run without env vars reproduces the 2.4 s/token pathology.
   Aligning the demo's defaults (or printing a hint when `DS4_DIAG` detects
   the weak config) would make every evaluation repeatable without environment
   incantations.

3. **Evaluate `DS4_PREFILL_MM`**. The matrix-matrix path reads the weights
   once per tile instead of once per token, but changes the accumulation
   order. The subsequent A/B on M1 Pro was negative (section 8): it stays
   opt-in and must not be promoted based on the theoretical estimate alone.

4. **Further reduce the dense stream's bytes**. This hypothesis led to the
   `DS4_QKV_Q4` and `DS4_SHARED_Q4` experiments, which later entered the fast
   profile. The residual sizes and the quality trade-off must be recomputed on
   the current configuration, not extrapolated from this list's pre-QKV
   numbers.

5. **Fusing the decode command buffers** (yield: ~10–25 ms/token, medium-high
   effort). Of the ~3 cb/layer, the pair "layer i's routed FFN" + "layer
   i+1's route" is fusable (no CPU readback in between when the slot cache
   serves all 6 experts). It would remove up to ~43 round-trips/token in
   fully-hit layers. The DIAG probe says whether the game is worth the candle
   on the target machine: below 100 µs/cb probably not.

6. **Prior-driven speculative expert prefetch** (yield: uncertain, low effort
   — it already exists). `DS4_PREFETCH_EXPERTS=N` is off because it steals
   bandwidth from the real gather when the prior is cold; with the persisted
   usage imatrix (`<gguf>.usage.json`) and high routing concentration in the
   DIAG, a targeted A/B on the most concentrated layers is worth it.

Non-levers (already closed or not paying off): the embed is already row-staged
(~8 KB); the interleaved pool has already brought a miss down to 1 pread of
~7 MB; the sidecar bundle covers the "gather < 60% of ceiling" case; the
resident output head + mlock has already eliminated the compressor's 235 ms.

## 7. Real measurements (2026-07-05, M1 Pro 16 GB, Flash IQ2XXS)

Runbook from §5 executed on a 13-token prompt, 48 generated, steady state = 44
tokens. Measured SSD ceiling: **5.25–5.69 GB/s** (parallel random); empty
command buffer **~21 µs** → synchronization is worth ~3 ms/token (not a
lever). **The GGUF in use does NOT contain MTP weights**. Even with those
tensors, lever 1 (§6) would first have required integrating the MTP loader
and path: the current runtime diagnoses their presence but does not consume
them.

| Config | Steady-state decode | gather IO | route/attn | experts | head |
|---|---|---|---|---|---|
| base (slots 16, stream, mlock) | **2110 ms/tok (0.47 tok/s)** | 1685 ms (80%), 617 MB/tok @ **0.38 GB/s = 7% of ceiling** | 216 ms | 190 ms | 19 ms |
| + `DENSE_Q4` + `SHARED_Q4` | **880 ms/tok (1.14 tok/s)** | 662 ms (75%), 636 MB/tok @ **1.01 GB/s = 18% of ceiling** | 116 ms | 94 ms | 7 ms |
| + `DS4_EXPERT_PREAD` | **455 ms/tok (2.20 tok/s)** | 225 ms (49%), 635 MB/tok @ **2.97 GB/s = 53% of ceiling** | 126 ms | 96 ms | 8 ms |
| + `EXPERT_BUNDLE` | *not tested*: build skipped for disk space (~72 GB required, 46 free) — run ≈ identical to the previous one (443 ms, 3.10 GB/s = 57%) | | | | |

Expert cache: 63–65% hits (94 misses/token ≈ 650 MB read); top-16
concentration ~0.3–0.5 per layer, usage-driven allocation active.

Reading the measurements:

- **Resident Q4 is worth 2.4×** on its own (0.47 → 1.14 tok/s). Not just for
  the ~90+95 ms removed from route/attn+experts: the gather — at NEARLY
  IDENTICAL bytes (617 vs 636 MB/token) — went from 1685 to 662 ms/token. The
  dense stream was contending the disk with the gather; removing it nearly
  tripled the gather's effective bandwidth. Experimental confirmation of
  lever 4 (§6).
- **`EXPERT_PREAD` is worth another 1.9×** (1.14 → 2.20 tok/s): a miss is no
  longer a memcpy from the mmap through 16 KB page faults, but an F_NOCACHE
  pread of the entire slab. Gather bandwidth rose from 1.01 to 2.97 GB/s, and
  prefill from 1085 to 480 ms/token (the union's gather was the same
  pathology). The §6 projection (~2.3 tok/s) was hit.
- The route/attn split profile (separate run, without cache/Q4 — read the
  reports): `q` 27% and `out` 24% dominate, flash-attn 9% ⇒ these are exactly
  the tensors `DENSE_Q4` makes resident; on the Q4 path there is not much
  left to squeeze there.
- **Cumulative: 0.47 → 2.20 tok/s (4.7×) with knobs alone.** The gather
  remains 49% of the token at ~53-57% of the ceiling: NVMe queue ~6-9
  requests (2-3 misses × 3 slabs per layer) against the ~24 at which the disk
  delivers the ceiling.

## 8. Measurements 2026-07-08 (same M1 Pro 16 GB, bundle ACTIVE, prompt 17 tok, 48 generated)

The sidecar bundle came online (built by the app, reused by the demo via
`DS4_BUNDLE_DIR`) and the day's matrix closed four questions:

| Config (on top of the Q4+PREAD+bundle base) | Steady-state decode | gather IO | hit |
|---|---|---|---|
| slots 16 | 2.78 tok/s | 627 MB/tok, 158 ms | 64% |
| + `DS4_QKV_Q4` | **3.06 tok/s (+10%)** | 606 MB/tok, 141 ms | 65% |
| + slots 20 | 3.20 tok/s | 550 MB/tok, 128 ms | 68% |
| + slots 24 | **3.33 tok/s** | 479 MB/tok, 115 ms | **73%** (no collapse) |

- **`DS4_QKV_Q4` promoted** during this campaign and today included in the
  GUI's fast preset: resident Q4 q_a+kv
  are worth +10% on their own — fewer bytes in the stream AND a halved matvec.
  The `.q4dense` cache extends incrementally (86 tensors, ~30 s).
- **`DS4_PREAD_SPLIT=2`: in the noise** (2.84 vs 2.78) — the gather already
  runs at 89-94% of the measured ceiling; the queue is no longer the
  bottleneck.
- **pread alignment: NOT a lever.** The "MISALIGNED random" probe (DIAG)
  swings above and below the aligned one between runs: thermal variance of
  the disk, no systematic penalty on this SSD/OS.
- **`DS4_MOE_NSG`: the default 4 remains the best on M1 Pro** (A/B on the
  QKV+24 slot config: nsg=2 → 3.10, nsg=8 → 3.25, nsg=4 → 3.33 steady-state
  tok/s; in prefill `experts` regresses ~20% with 2/8). Consistent with the
  fact that in decode ASYNC_FFN hides most of the routed FFN
  from the critical path. The knob stays (auto-tune 2/8) for wider GPUs.
- `DS4_PROFILE_ROUTE` split on the Q4+QKV path (decode, synchronous):
  out 63 ms (output proj+HC+router), q 38, comp 28, attn 24, kv 16, plus
  100 ms of experts that ASYNC_FFN overlaps almost entirely. `q` and `out`
  remain large WITH the weights already resident in Q4 ⇒ it is not weight
  bandwidth: it is the CHAINS of micro-dispatches (projection→norm→rope, HC,
  router+readback), ~140 ms/token of small sequential operations.

### Evening update (same day)

- **`DS4_SHARED_Q4` now PAYS: +7%** (3.13 → 3.36 total tok/s, steady state
  3.45) — it was neutral in the 07-06 tuning, but now that trio+q_a+kv are
  resident the shared FFNs were the last big item in the stream.
  Lossy (the greedy continuation changes while remaining coherent): GUI toggle.
- **Self-speculative (DS4_SPEC_K): perfect parity, negative economics**
  (K=2: 78% acceptance but 2.38 vs 3.36). Structural reason: the token is
  now dominated by serial route/attn, which batched verification does not
  amortize. Parked as opt-in; details in SELF-SPECULATIVE.md.

- **MTLIO (Metal fast resource loading): no repeatable advantage in these
  tests.** The MTLIO/pread ratio in the same run was 1.11 then 0.81, within
  thermal variance. Not a universal conclusion: the current preset enables it
  with a circuit breaker and automatic fallback to `pread`, so it should be
  re-evaluated per machine and memory state.

- **Prefill A/B on 3481 tokens (SHARED+QKV+24 slot config)**:
  union 192 = 7.23 tok/s; **union 256 = 8.63 tok/s (+19%, winner)**;
  `DS4_PREFILL_MM=1`+chunk 1024 = 4.23 tok/s (`experts` 58→168
  ms/token): **PREFILL_MM closed** on M1 Pro. The quick benchmark in
  Settings covers 192/256 and applies the best.

## 9. Current status and open tasks (July 13, 2026)

The decisions consolidated by this campaign are: Q4 extended to QKV/shared in
the fast profile, union 256 on the reference machine, `DS4_PREFILL_MM` left
opt-in and the CLI self-speculative parked. The current GUI preset uses
22 slots, `DENSE_Q4` + `QKV_Q4` + `SHARED_Q4`, MetalIO with fallback, route
batch 32 and union 256; it remains a snapshot for M1 Pro 16 GB, not a value to
copy onto every Mac.

Still-useful tasks:

1. **Repeat a full matrix per machine class** (Pro/Max/Ultra), with the same
   GGUF, cache, prompt and warm-up, including memory pressure and swap.
   Slots, simdgroup count and MetalIO's convenience depend on the hardware.
2. **Profile route/attention on the GPU timeline** with Instruments before
   rewriting or fusing micro-chains. `DS4_PROFILE_ROUTE` adds syncs and its
   absolute times are not a sufficient baseline.
3. **Re-evaluate self-speculative only after a structural change**: genuinely
   multi-token route/attention verification, a much cheaper draft, or a future
   real integration of the MTP path. The mere presence of MTP weights in the
   GGUF is not enough; the current CLI implementation does not use them, and
   preserves greedy parity in the day's tests but reduces throughput.
4. **Keep `DS4_PREFILL_MM` as an A/B for new GPUs**, verifying both throughput
   and quality: on the campaign's M1 Pro it was clearly slower and changes the
   accumulation order.
