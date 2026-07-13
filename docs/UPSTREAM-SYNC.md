# Upstream Sync (`antirez/ds4`)

DwarfStar's engine is a **pure-Swift rewrite** of the upstream C project
[`antirez/ds4`](https://github.com/antirez/ds4.git). This file records the last
upstream comparison so that the next review starts from a known baseline instead
of rediscovering the same commit history.

This is a dated synchronization snapshot, not a statement about the current
upstream HEAD. The documentation restructuring on 2026-07-13 did not perform a
new network comparison; use the commands in the final section before advancing
the baseline.

## Current Baseline

| Field | Value |
|---|---|
| Upstream HEAD reviewed | `80ebbc3` |
| Date | 2026-06-17 |
| Repository | `https://github.com/antirez/ds4.git` |
| Result | **No urgent changes required** for the standard model path. |

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
