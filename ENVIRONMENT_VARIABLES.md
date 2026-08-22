# Performance and diagnostic environment variables

Command-line options are the supported interface for normal inference.  The
variables below are the curated, user-facing switches used to isolate optimized
paths, require test coverage, or collect diagnostics.  They are not a promise
that every internal `getenv()` knob is a stable API.

Most backend switches are cached on first use.  Start a new process when
changing them.  Unless a row says otherwise:

- use the documented value `1`; many rollback switches are presence-based, so
  setting them to `0` still enables the rollback and they must instead be unset;
- `DISABLE` or `NO` is the rollback switch and takes precedence;
- `REQUIRE` turns an eligible silent fallback into an error, so tests cannot
  pass without exercising the intended path;
- `STATS`, `PROFILE`, and `ORACLE` are diagnostic and may perturb timing;
- benchmark controls and candidates in separate processes.

## Metal

| Variable | Default behavior and purpose |
| --- | --- |
| `DS4_METAL_PREFILL_CHUNK=N` | Set the prefill cap when `--prefill-chunk` is absent; the CLI option takes precedence. This historical name is consumed by the shared graph planner rather than by a Metal kernel alone. |
| `DS4_METAL_NO_RESIDENCY=1` | Skip creation and residency requests for the model-view residency set. Diagnostic rollback for resident, non-streaming models. |
| `DS4_METAL_DISABLE_QUEUE_RESIDENCY_SET=1` | Still create, commit, and request the model residency set, but do not attach it to Metal command queues. This isolates queue-residency behavior without disabling the complete residency policy. |
| `DS4_METAL_STREAMING_EXPERT_NOCACHE=1` | Reopen the Metal SSD expert file with `F_NOCACHE` so streamed experts do not displace the dense working set from the page cache. Leave unset for cached `pread`. |
| `DS4_METAL_STREAMING_EXPERT_PREAD_SPLIT=N` | Split each expert read into 1–8 aligned requests. The automatic value is 1 below 64 configured cache experts and 4 at 64 or more. |
| `DS4_METAL_DISABLE_Q4_DENSE_PAIR=1` | Split the default Metal Q-A/KV Q4 pair back into two standalone projections. |
| `DS4_METAL_DISABLE_M1_IQ2_MID_ONLY=1` | Restore the canonical IQ2 address-table gate/up producer on the exact M1 SSD-streaming decode shape. The specialization is automatic by default. |
| `DS4_METAL_REQUIRE_M1_IQ2_MID_ONLY=1` | Fail closed when an otherwise eligible M1 IQ2 mid-only dispatch cannot use the specialization. |
| `DS4_METAL_ENABLE_M1_IQ2_MID_ONLY=1` | Compatibility alias for the former opt-in. It is harmless because the path is now automatic. |
| `DS4_METAL_DISABLE_IQ2_XXS_SSD_PREFILL_MM=1` | Restore sparse matvec for an eligible grouped IQ2 SSD-prefill chunk. |
| `DS4_METAL_REQUIRE_IQ2_XXS_SSD_PREFILL_MM=1` | Require grouped IQ2 SSD-prefill MM for eligible chunks and reject insufficient cache instead of silently falling back. |
| `DS4_METAL_ENABLE_STREAMING_PREFILL_EXPERT_READAHEAD=1` | Restore the historical `F_RDADVISE` plus parallel-`pread` sequence for cold-storage A/B tests. Normal grouped prefill skips the redundant hint. |

The detailed Metal A/B contracts and expected oracle counters live in
[`QA_BEFORE_RELEASES.md`](QA_BEFORE_RELEASES.md).

## CUDA Q4 and Q8 diagnostics

| Variable | Default behavior and purpose |
| --- | --- |
| `DS4_CUDA_DISABLE_Q4_DENSE_PAIR=1` | Split the Q-A/KV Q4 pair back into two standalone projections. |
| `DS4_CUDA_NO_Q4_GB10_FAST=1` | Umbrella rollback for the GB10-specific Q4 choices; it does not disable the older cross-CUDA dense pair. |
| `DS4_CUDA_ENABLE_Q4_GROUPED_ATTN_A_BATCH=1` | Enable grouped attention-A for two-to-eight-token GB10 verifier batches. |
| `DS4_CUDA_REQUIRE_Q4_GROUPED_ATTN_A_BATCH=1` | Fail closed if that grouped batch path is unavailable. |
| `DS4_CUDA_Q4_GROUPED_ATTN_A_ORACLE=1` | Compare grouped attention-A with the canonical result and retain the canonical output. Disable graph capture for this diagnostic. |
| `DS4_CUDA_ENABLE_Q4_K1024_PERSISTENT=1` | Enable the experimental persistent-CTA kernel for the exact `32768x1024` Q4 shape. |
| `DS4_CUDA_NO_Q4_K1024_PERSISTENT=1` | Roll back the persistent K1024 experiment. |
| `DS4_CUDA_REQUIRE_Q4_K1024_PERSISTENT=1` | Require the K1024 candidate before enqueue instead of silently using canonical MMVQ. |
| `DS4_CUDA_ENABLE_Q8_FOLD=1` | Enable the experimental one-shot Q8_1 producer-to-consumer fold. |
| `DS4_CUDA_NO_Q8_FOLD=1` | Dominant rollback for the Q8_1 fold. |
| `DS4_CUDA_Q8_FOLD_ORACLE=1` | Compare fresh canonical Q8_1 bytes and consumer outputs. Use with `DS4_CUDA_DECODE_GRAPHS=0`; require nonzero calls and zero mismatches/skips. |

## ROCm Q4

| Variable | Default behavior and purpose |
| --- | --- |
| `DS4_ROCM_DISABLE_Q4_PREFILL_TILE8=1` | Restore the legacy Q4 prefill kernel. TILE8 is automatic for validated chunks of 9 through 4096 tokens. |
| `DS4_ROCM_REQUIRE_Q4_PREFILL_TILE8=1` | Fail closed when an eligible Q4 prefill call cannot use TILE8. |
| `DS4_ROCM_ENABLE_Q4_PREFILL_TILE8=1` | Legacy no-op accepted by existing scripts; TILE8 no longer requires an enable variable. |
| `DS4_ROCM_Q4_PREFILL_TILE8_STATS=1` | Report dense, pair, attention-batch, and token counters at process exit. |
| `DS4_ROCM_ENABLE_Q4_DENSE_PAIR=1` | Share one Q8_K activation quantization between the two Q4 dense projections. This pair remains opt-in. |
| `DS4_ROCM_DISABLE_Q4_DENSE_PAIR=1` | Dominant rollback for the ROCm Q4 dense pair. |
| `DS4_ROCM_ENABLE_Q4_GROUPED_ATTN_A=1` | Enable the two-launch grouped attention-A decode path. This path remains opt-in pending model A/B results. |
| `DS4_ROCM_DISABLE_Q4_GROUPED_ATTN_A=1` | Dominant rollback for grouped attention-A decode. |
| `DS4_ROCM_REQUIRE_Q4_GROUPED_ATTN_A=1` | Fail closed if grouped attention-A decode is disabled or ineligible. |
| `DS4_ROCM_Q4_GROUPED_ATTN_A_STATS=1` | Report grouped calls, dispatches, groups, fallbacks, and failures. |

Run `make test-strix-rocm-q4-parity` and
`make test-strix-rocm-q4-prefill` on a `gfx1151` Strix Halo host before making
performance claims.  The synthetic oracle proves layout and numerical parity;
it does not by itself prove that a complete Q4 model fits safely in GTT.

## Historical `DS4_METAL_*` graph controls

The shared graph implementation predates the CUDA backend and retained a few
`DS4_METAL_*` names.  In a non-ROCm GPU build, these controls affect the shared
Metal/CUDA graph policy; ROCm explicitly ignores them:

- `DS4_METAL_DISABLE_HC_FUSION`
- `DS4_METAL_DISABLE_HC_NORM_FUSION`
- `DS4_METAL_DISABLE_KV_FUSION`
- `DS4_METAL_DISABLE_QKV_NORM_FUSION`
- `DS4_METAL_DISABLE_QKV_PAIR_PROJ`
- `DS4_METAL_DISABLE_COMPRESSOR_PAIR_PROJ`
- `DS4_METAL_DISABLE_ATTN_OUT_HC_FUSION`
- `DS4_METAL_DISABLE_SHARED_DOWN_HC_FUSION`

The prefix is historical rather than an indication that these particular
switches are always Metal-only.  New backend-specific controls should use the
backend they actually configure (`DS4_CUDA_*` or `DS4_ROCM_*`) instead of
extending this legacy naming.
