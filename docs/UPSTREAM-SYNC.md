**English** | [Italiano](UPSTREAM-SYNC.it.md)

# Upstream Sync (`antirez/ds4`)

DwarfStar's engine is a **pure-Swift rewrite** of the upstream C project
[`antirez/ds4`](https://github.com/antirez/ds4.git). This file records the last
upstream comparison so that the next review starts from a known baseline instead
of rediscovering the same commit history.

For a turnkey implementation plan of the still-open runtime gaps (routed Q8_K,
Pro Q4 split-load, mixed prefill+decode server batching), see
[`PORTING-GAPS.md`](PORTING-GAPS.md). The offline tooling gaps (GGUF writer,
offline requantizer) are already implemented in pure Swift.

This is a dated synchronization snapshot, not a statement about the current
upstream HEAD. The documentation restructuring on 2026-07-13 did not perform a
new network comparison; use the commands in the final section before advancing
the baseline.

## Current Baseline

| Field | Value |
|---|---|
| Upstream HEAD reviewed | `b030961` |
| Date | 2026-08-05 |
| Repository | `https://github.com/antirez/ds4.git` |
| Result | **Shared correctness fixes and the M1-capable streaming top-512 prefill selector ported.** |

## Update 2026-08-05 (reviewed upstream `main` through `b030961`)

Reviewed the latest 20 commits on `main` and ported the shared behavior rather
than the C ownership/runtime details:

- `222b2cb`: exact `kernel_topk_stream512` plus the Swift dispatch for wide
  DeepSeek indexed-prefill rows (`topK=512`, `nComp>1024`, `nTokens>=32`). The
  Metal 4/MPP dual-head variant remains deferred because it cannot run on the
  current M1 Pro validation machine.
- `7694112`: GLM live-prefix rewind. A repeated stateless prompt that prefixes
  the live prompt+answer state now rolls back to its penultimate token and
  reevaluates the last prompt token instead of rebuilding the prompt.
- `af80694` + `7fb2830`: speculative accepts are always restored and replayed
  through ordinary one-token decode, keeping recurrent compressor state greedy
  identical after both full and partial accepts.
- `fe2d3b0`: the full tool prompt no longer asks the model to reopen the think
  block, and tool-enabled DeepSeek streams hold tentative post-think text until
  a tool/final boundary or a second close marker classifies it.
- `a169cff`: client tool histories are validated in one forward pass, with
  duplicate/unknown call ids rejected before OpenAI, Responses or Anthropic
  inference.
- `a968c08`: GGUF writer/type-size/requantization shape products now use checked
  arithmetic. The external safetensors C converter still needs the upstream C
  patch until that converter itself is rewritten in Swift.

Already aligned: disk-KV restores retain successful checkpoints (`24fa85e`),
complete DSML calls are recognized regardless of an unclosed reasoning block
(`51a1c14`), and project edits never enabled fuzzy `[upto]` matching (`b030961`).
The routed MoE MPP prefill kernel from `532ec8b` is retained as a future
Metal-4 port; it has no executable path on pre-M5 Apple GPUs.

## Update 2026-07-28 (reviewed branch `laguna-s2.1`, head `448d569`)

Reviewed the upstream **`laguna-s2.1` feature branch** (17 commits over merge
base `efdadd4`, not yet on upstream `main`): native **Laguna S 2.1** support
(Poolside GQA+MoE model — chat, interleaved reasoning, tagged tool calls, two
published quant recipes plus a mixed Q2_K/Q3_K file) with Metal/CUDA/ROCm
decoders and optional **DFlash** speculative decoding.

Port decision: land the complete **frontend now, decoder as a gap**, following
the GLM staged pattern. Ported in this pass, each with deterministic unit
tests: architecture registration/detection, `LagunaConfiguration`
(`config_validate_laguna_model`), `LagunaTokenizer`
(`bpe_tokenize_text_laguna` newline pre-split + single-digit GLM4-shape
groups), `LagunaChatRenderer`/`LagunaToolCodec`
(`render_laguna_chat_prompt_text`, `laguna_chat_append_*`,
`append_laguna_tool_calls_text`), the reference sampling defaults
(`ds4_engine_sampling_defaults`), `LagunaTensorSchema`
(`weights_validate_laguna_layout`) and the download catalog
(`download_model.sh` targets `laguna-q4`, `laguna-q2-q3`, `laguna-dflash`).
Out of this pass: `metal/laguna.metal` + drivers and DFlash — tracked as
**Gap 4** in [`PORTING-GAPS.md`](PORTING-GAPS.md) because their validation
gate needs Apple hardware and the real GGUFs. `LagunaRuntimeGate.enabled`
stays off until that gate passes. CUDA/ROCm Laguna code remains out of scope.

**Parity re-review (same day, same head).** The whole ported Laguna frontend
was re-reviewed line by line against `448d569` with independent verification
of every finding; all confirmed divergences were fixed (server-exact think
framing and raw-tool-text replay, schema declaration-order arguments,
XML-entity decoding, the exact server parser `parseServer`, cross-family
scanner literals, the BOS/CLI prompt-encode path, think-mode stops, wired
sampling defaults, the oracle SWA clamp) or documented as deliberate
deviations. Full report:
[`architectures/laguna-s-2.1/C-PARITY-REVIEW.md`](architectures/laguna-s-2.1/C-PARITY-REVIEW.md).
The completeness sweep also surfaced four previously undeclared server-level
gaps (model aliases + `/v1/models`, malformed-tool-call recovery wiring,
Laguna session-cache suffixes, the `laguna-openrouter-100` QA fixtures), now
listed under Gap 4.

## Update 2026-07-27 (reviewed `80ebbc3..0a7ad77`)

Re-ran the comparison. Since the last baseline there are **15 upstream commits**
(to `0a7ad77`, 2026-07-23), **12 touching the shared scope**. Reviewed at the
commit-message + file-stat level; items marked *evaluate* still need a code read
on a Mac.

| Commit | Area | Port decision |
|---|---|---|
| `519c4d8` Fix PRO streaming and sampling correctness | engine/Metal/server | **NEW GAP — evaluate.** Adds an expert-cache service-thread guard and makes computed logits **independent of the SSD streaming expert-cache budget**, plus a too-small-cache warning and a sampling fix. Check the Swift expert cache (`ExpertSlotCache`, streaming) holds the same "logits ⟂ budget" invariant. |
| `427e281` Optimize Metal prefill and decode kernels | Metal | **NEW GAP — resync.** Large kernel pass (ds4_metal.m +2572; dense/rope/misc/cpy `.metal`). The port's kernels are ported from these files and are now behind — evaluate porting the optimizations/fixes. |
| `36cd0ca` Add native Metal session batching | server/engine | **Reference for Gap 3.** The Metal implementation of mixed prefill+decode + server slots — the port's continuous-batching gap now has a Metal reference (not only CUDA). |
| `a185c36` Stabilize CUDA and Metal session batching | Metal/CUDA | **Deferred** — pairs with Gap 3 (Metal batching not yet in the port); CUDA out of scope. |
| `0a7ad77` Fix batched server session recovery race | server | **Deferred** — depends on server session batching (Gap 3), which the port does not have (serialized `RequestGate`). |
| `005afed` Add GLM 5.2 inference and quality fixtures | engine/Metal | **Relevant to GLM parity.** Upstream GLM landing + quality fixtures; the port has GLM (experimental). Use the fixtures to certify GLM logits parity. |
| `e0824dd` Make release builds warning-free | C | **N/A** (C-only warnings). |
| `ef8d923` Add ROCm GLM 5.2 support | ROCm | **N/A** (out of scope). |
| `66cbce5` Unify distributed inference CLI and documentation | CLI/docs | **N/A** (distributed CLI/docs; the port has its own). |
| `36ee8c1` Add CUDA tensor parallelism and session batching | CUDA/TP | **N/A** (out of scope). |
| `63d9874` Add two-machine Metal tensor parallelism | TP | **N/A** (tensor-parallel + RDMA out of scope). |
| `fc9efd1` Add DSpark speculative decoding | speculative | **PORTED, DEVICE VALIDATION IN PROGRESS.** Swift has downloads, strict 81-tensor validation, target-hidden capture, private-ring seeding/maintenance, the three-stage Metal forward (non-causal attention + MoE), final HC/output/Markov/confidence heads and exact target verification/replay in the GUI and demo greedy loops. A same-checkpoint performance/acceptance A/B still remains. |

Net new actionable items: `519c4d8` (expert-cache/sampling correctness) and
`427e281` (Metal kernel resync) — both need on-device evaluation. Advance the
reviewed baseline to `0a7ad77` after those two are triaged on a Mac.

## What Is in Scope

DwarfStar shares only the inference-engine surface with upstream. These upstream
files are relevant:

- `ds4.c` — decoder, MoE/NSA, SSD streaming, expert cache. DwarfStar equivalent:
  `DS4Core` + `DS4Metal`.
- `ds4_metal.m` — Metal runtime and kernels. DwarfStar equivalent: `DS4Metal`.
- `ds4_server.c` — HTTP server behavior. DwarfStar equivalent:
  `Sources/DwarfStar/Features/Server`.

Out of scope for this port: CUDA/ROCm backends (`ds4_cuda.cu`, `ds4_rocm.cu`,
`rocm/`), the upstream MTP-backed speculative path, the terminal agent
(`ds4_agent.c`, raw-mode/TTY behavior), `ds4-eval`, and `ds4_cli.c`. DwarfStar's
separate, MTP-free self-speculative CLI experiment is documented in
[`SELF-SPECULATIVE.md`](SELF-SPECULATIVE.md) and is not evidence that the
upstream MTP surface was ported.

## Recent Commit Review (Through `80ebbc3`)

| Commit | Area | Port Decision |
|---|---|---|
| `d75e23d` Guard Metal tensor frees | Metal | **N/A.** Guards against double-free of bridged Objective-C handles in C. Swift `GPUTensor` is ARC-managed; the bug does not exist in this port. |
| `8384adf` Fix Metal SSD streaming cache reuse | Metal/streaming | **N/A.** Avoids evicting an expert slot while a command buffer is still in flight. DwarfStar's `GraphContext.commit()` waits for completion, so no gather/evict command buffer race exists. |
| `91bafb5` Recover tool calls inside unclosed `<think>` | server/generation | **N/A.** DwarfStar enters tool mode on the DSML token regardless of `inReasoning`; if parsing fails, the markup is shown as text instead of being discarded. |
| `fd2d173` Harden server JSON parsing | server | **N/A.** The C parser was handwritten; DwarfStar uses Foundation `JSONSerialization`. |
| `cafc134` Fix server const warning | server | **N/A.** C-only warning. |
| `1cfa5cc` Refactor streaming expert cache API | streaming | **N/A.** Multi-backend refactor with no behavioral change to port. |
| `7a77a28` Release cache margin on mlock failure / `cd57428` Cap oversized caches | streaming | **Partially covered.** DwarfStar now has best-effort `DS4_MLOCK` for hot buffers and a slot-limited expert cache, but it does not use the same C slab allocator. Keep this class of failure in mind when changing cache sizing. |
| `f2d701a` Fix distributed SSD streaming layer slices | distributed | **Deferred.** DwarfStar's distributed mode is implemented but still needs numerical validation. Review together with that validation work. |
| `81f35e7` + `b548d86` mixed-precision routed experts | streaming/quant | **Ported.** Per-layer routed expert quantization is supported. See below. |

Commits not listed here, such as ROCm/CUDA, the upstream MTP path, terminal
agent, and `ds4-eval` work, are currently outside the DwarfStar port boundary.

## Ported: Per-Layer Mixed-Precision Routed Experts

Some GGUFs use non-uniform routed expert quantization across layers, for example
an IQ2_XXS/Q2_K base with selected layers upcast to Q4_K via `--tensor-type`.
DwarfStar supports this without changing uniform-model behavior:

- `LayerWeights` stores per-layer `gateQuant`, `upQuant`, and `downQuant`.
- `GGUFWeights.layer` detects the real tensor types even when experts are not
  fully loaded (`loadExperts == false`).
- `decodeExperts` selects kernels from the layer-local quant fields instead of
  global `DSV4Dims` quant fields, covering both decode and batched prefill.
- `GGUFWeights.gatherExperts` already computes bytes per expert from tensor
  `blockBytes`, so the copy size is correct per layer.
- Expert slot-cache remains a single size-class cache. Out-of-class mixed layers
  bypass it and use direct gather, which is correct.
- `InferenceService` logs the count of out-of-class layers at startup.

This still needs on-device validation with a mixed GGUF fixture.

## Open Gap: Distributed Fix

`f2d701a` should be reviewed when distributed inference receives numerical
parity validation. The current priority is validating the distributed pipeline
as a whole before porting upstream slice-level fixes in isolation.

## How to Repeat the Comparison

```sh
git clone --depth 60 https://github.com/antirez/ds4.git /tmp/ds4-upstream
git -C /tmp/ds4-upstream log --oneline --since=2026-06-17
git -C /tmp/ds4-upstream log --oneline 80ebbc3..HEAD -- ds4.c ds4_metal.m ds4_server.c
```

For every new commit, ask:

- Does it touch an area DwarfStar shares with upstream: engine, Metal, or server?
- Is it a behavioral or correctness change, rather than a C-specific memory,
  warning, or backend-only change?
- Is it outside the excluded surfaces: CUDA/ROCm, upstream MTP execution, TTY
  agent, eval tooling?

If the answer is yes, evaluate a Swift port. Otherwise, record the commit as
N/A and advance the baseline after review.
