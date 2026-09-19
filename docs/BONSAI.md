# Ternary Bonsai 2 27B

This branch provides native text inference for the Prism
[Ternary-Bonsai-2-27B GGUF](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
on Metal, plus a scalar CPU reference implementation. It does not invoke or
link llama.cpp at runtime.

## Run

Build on macOS with `make -j4 ds4 ds4-server`, then run:

```sh
./ds4 --metal \
  -m gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --ctx 4096 --nothink --temp 0 \
  -n 50 -p "narrami la storia di roma"
```

The model is dense, with no routed experts. Do not pass `--ssd-streaming`:
the packed PQ2 file is about 6.71 GiB and is mapped directly as Metal weights.
The 4096-token session additionally uses about 512 MiB for full-attention KV,
150 MiB for GDN state/history, and scratch buffers including 16.4 MiB for batched
prefill. Metal checks
the aggregate session allocation against the device's recommended working set.

```sh
./ds4-server --metal \
  -m gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --ctx 4096 --host 127.0.0.1 --port 8000
```

The OpenAI-compatible model ID is `ternary-bonsai-2-27b`.
Text chat uses Qwen ChatML, with the checkpoint's thinking/non-thinking prefix.
The Qwen Flash Next reasoning-effort system instruction is not inserted for
this model.

## Implemented graph and formats

The GGUF architecture identifier is `qwen35`. This loader admits the known
64-layer, 5120-wide dense checkpoint: three Gated DeltaNet layers followed by
one full-attention layer, repeated sixteen times. Each attention operation
and dense SwiGLU FFN has its own RMS normalization and residual connection.
It has no Flash Next hyper-connections, sparse indexer, MoE, or n-gram table.

Prism's private group-128 types are decoded directly:

| GGUF type | Block | Encoding |
| --- | --- | --- |
| PQ2_0, 142 | 128 values / 34 bytes | FP16 scale and adjacent two-bit codes |
| PTQ1_0, 143 | 128 values / 28 bytes | Packed ternary codes and FP16 scale |

These are distinct from upstream TQ1/TQ2 and legacy Q2_0. The loader validates
the explicit signs, transform metadata, complete folded-weight list, tensor
shapes, and scalar tensor types. Missing or incompatible metadata is rejected.

PQ2 is already a two-bit weight representation, not two-bit floating-point
arithmetic: its four codes represent {-1, 0, 1, 2}, multiplied by a shared
FP16 scale. Its effective size including scales is 2.125 bits per weight.
Ternary weights need only {-1, 0, 1}, while the decoder also supports the fourth
PQ2 code. Relabeling this as FP2 would provide neither
smaller storage nor a new Metal arithmetic instruction. FP32 accumulators
and recurrent state are independent of the packed weight format.

For each declared folded projection, activations undergo the signed,
normalized 1024-wide Sylvester Hadamard transform. Embedding lookup applies
the inverse order (Hadamard, then signs). GDN's folded output projection also
permutes value heads from tiled to grouped order before the transform.
The BF16 alpha/beta projections consume the untransformed input.

Both backends keep activations, recurrent state, and KV in FP32. Metal uses
native packed GEMV kernels for decode and batched projections for prefill. The CPU
path uses double-precision dot-product accumulation as an independent numeric
reference and is intentionally slow.

The generic PQ2 Metal kernel decodes a whole 128-value block per iteration,
loading its scale once and extracting four weights per lane. It preserves the
original lane assignment, addition order and final SIMD reduction. Sibling projections
(Q/K/V, QKV/Z and FFN gate/up) reuse a signed Hadamard transform when both its
sign table and width match. This removes 144 redundant transforms per token
for the supplied checkpoint without reusing stale activations across layers.

A further pass applies the typed-load and SIMD-reduction techniques used by
`metal/dense.metal` and `metal/norm.metal`, while retaining Bonsai's arithmetic.
BF16 alpha/beta projections resolve their storage type before the K loop.
Hadamard computes within-SIMD and within-register butterfly stages without
threadgroup synchronization (four barriers instead of eleven). Normalization
uses two barriers instead of nine but reproduces the original 256-thread
reduction tree, rather than substituting a differently associated `simd_sum`.
No activation/state precision or quantization changes are made.

Large PQ2 decode projections use a specialized kernel that reuses sixteen
input coefficients across eight output rows per SIMD group. It centers each
packed byte before evaluating its four coefficients, avoiding a separate
zero-point correction and preserving exact zero for a zero-weight row. The
scale remains FP16 on disk and is applied in FP32. This changes summation
association relative to the generic GEMV; it does not requantize weights.
The row/stripe mapping is adapted from Prism's reference runtime, with its
license and commit attribution included in the shader source.

During decode, compatible PQ2 gate/up matrices use a fused four-row-pair
kernel with an immediate SiLU product. Each projection retains the standalone
kernel's lane/block order, FP32 sums and SIMD reduction. This removes the two
intermediate vectors and two dispatches per FFN. Matching dimensions and the
same Hadamard sign table are required; other layouts use the original sibling
path. Unrotated BF16 alpha/beta projections also share one kernel while keeping
independent accumulators and the original reduction order. These two fusions
do not change the multi-token prefill projection path.

For complete groups, separate template instantiations remove row-bound checks
inside these GEMV loops: sixteen rows per standalone threadgroup and eight
per fused gate/up threadgroup. The host selects them only when the row count
is divisible by the corresponding group size. The general kernels retain all
checks for partial groups; both versions share the same arithmetic body.

Metal prefill processes up to 32 tokens per chunk in layer order. The effective
chunk size is the smaller of `--prefill-chunk`, 32, and the remaining prompt.
GDN convolution and recurrence still run in token order; full attention writes
KV and applies the causal bound for each query. Projection matrices, Hadamard
transforms, normalizations, residuals and FFN activation work on batched rows.
Only the final prompt row needs a vocabulary projection.

Hadamard transforms dispatch all token rows together, keeping signs and grouped
head permutations local to each row. Full-attention Q/K normalization, RoPE,
KV publication and output gating are also batched. Publishing future KV rows
does not make them visible to earlier queries: each score/attention operation
retains its own causal position. At sixteen or more tokens, GDN convolution,
Q/K L2 normalization and recurrence use three batched dispatches per layer.
Convolution and recurrence walk tokens causally and retain the single-token
arithmetic order and FP32 state. Smaller chunks retain the original GDN path.

For PQ2 matrices with at least 4096 output rows and chunks of at least sixteen
tokens, prefill uses a 64-output by 32-token by 32-K matrix tile, following the
packed shared-memory layout used by the existing DeepSeek Metal kernels.
Each thread expands sixteen coefficients with one scale load. Matrix operands
and accumulators stay FP32; weights are expanded only in 12 KiB of threadgroup
scratch, not into a persistent full-precision matrix. Contiguous `float4`
activation loads feed the tile, and complete output tiles store directly to
device memory; partial tiles retain a bounds-checked scratch path. Matrix accumulation
changes association relative to GEMV. Smaller projections and chunks use a
four-token kernel that preserves each token's original reduction, or serial
GEMV for chunks below four tokens. CPU prefill remains sequential.

Each Metal chunk completes before its session position advances. Invalid
token batches are rejected before dispatch; a GPU failure invalidates the
checkpoint so a subsequent synchronization rebuilds recurrent state.

## Scope

- CLI and HTTP text generation, incremental prompt reuse, independent sessions,
  reset and prefix rebuilding are supported.
- Rewind invalidates the recurrent checkpoint; synchronize the retained prompt
  before evaluating more tokens. Changing a prompt prefix rebuilds its state.
- MTP, vision/mmproj, CUDA/ROCm, SSD expert streaming, distributed execution,
  directional steering and serialized session/KV caches are not implemented.
- PQ2 was tested with the full 27B checkpoint. PTQ packing and execution are
  tested with synthetic weights; a full PTQ checkpoint comparison is still needed.

## Validation

```sh
make test-bonsai             # codecs, Hadamard, graph and malformed inputs
make test-bonsai-metal       # native kernels and CPU/Metal graph parity
make tests/test_bonsai_model # optional real-model API/lifecycle runner
make bench-bonsai-pq2        # original vs optimized PQ2 kernels, no model needed
make bench-bonsai-bf16       # typed BF16 alpha/beta projections
make bench-bonsai-elementwise # exact norm/Hadamard checks and GPU timings
make bench-bonsai-mm         # four-token projections versus repeated GEMV
make bench-bonsai-mma        # FP32 tiled PQ2 prefill versus four-token kernel
./tests/test_bonsai_pairs --bench # exact gate/up and BF16 decode fusions
./tests/test_bonsai_fullrows --bench # complete-row specialization vs frozen kernels
```

The model-free tests cover all fp16 scales, PQ2/PTQ packing, explicit-matrix
Hadamard oracles, grouped output, GDN SiLU versus sigmoid, causal convolution,
state/reset, skipped output heads, and folded quantized embedding/vocabulary.
They also cover split and partial batches, prompt capacity and invalid-token
rejection, grouped folded projections, matrix-tile tails and canaries, all
256 PQ2 bytes against basis vectors, and FP64 error bounds for the new decode
kernel. `test-bonsai-metal` enables Metal API validation.
Paired decode tests compare the fused kernels with separate production
projections and SiLU, including every packed-byte basis, zero rows, output
canaries, PQ2 row tails and BF16 column tails. Batched GDN tests check whole
state/history and chunk continuation against fixed single-token references.
Complete-row tests compare both template paths with frozen shader references,
including the real projection/vocabulary shapes and partial row/K stripes.

`tests/bonsai_reference.cpp` is an optional external comparison driver for
[Prism's reference runtime](https://github.com/PrismML-Eng/llama.cpp/tree/9a9394a895b96003ca842a6041cb28ac49a108f7).
Its build instructions are in the source. It writes full vocabulary logits
using the same row format as `tests/test_bonsai_model.c`; compare identical
token IDs and use FP32 KV and flash attention off for arithmetic comparisons.
The external reference is not a build dependency of ds4.

On M1 Max, the seven-token raw prompt `narrami la storia di roma` and three
subsequent decode steps produced the same greedy tokens as Prism. Across
four full-vocabulary logit rows, maximum absolute error was 5.67e-5 and
maximum RMSE was 1.15e-5. These checks establish numerical agreement for the
tested inputs, not a general quality or long-context benchmark.

The formatted 19-token chat prompt and sixteen decode steps also matched
Prism's greedy token at all 17 checked positions. For this longer comparison,
maximum logit error was 0.003181 and maximum RMSE was 0.000346. The scalar CPU
and Metal first-token comparison matched argmax, with maximum absolute error
7.63e-6 over all 248320 logits. These comparisons used the earlier generic
kernel baseline.

Current same-prefix synchronization remains bit-identical. Incremental append
and fresh batched replay can use different FP32 reductions, so lifecycle tests
require finite logits, matching argmax, maximum absolute difference <=1e-4
and RMSE <=1e-5. Reset, changed prefix, append and rewind/replay passed; the
largest measured lifecycle difference was 1.98e-5, with RMSE 3.89e-6.

## Current Metal performance

### First batched pass

On Apple M1 Max (32 GiB), three alternating pairs compared the previous
sequential prefill / block-PQ2 GEMV implementation with batched FP32 prefill and
centered PQ2 decode. Both used the same local PQ2 file, context 4096, configured
prefill chunk 128, and 32 greedy decode steps. One model process ran at a time,
with warm file-cache weights. Median measured rates:

| Prompt and phase | Previous, token/s | First batched pass, token/s | Gain |
| --- | ---: | ---: | ---: |
| Rome chat, 19-token prefill | 7.628 | 17.137 | +124.7% |
| Decode after Rome chat | 7.707 | 13.327 | +72.9% |
| Synthetic repeated IDs, 128-token prefill | 8.559 | 26.286 | +207.1% |
| Decode after 128-token prompt | 8.242 | 13.934 | +69.1% |

The 128-token input repeats the Rome chat token IDs; it is a throughput fixture,
not a quality dataset. The real-model driver includes frontier logit copying
and saving in its prefill timer, while decode timing covers evaluation only.
Initialization and shader compilation are outside both phase timers.

The final combined kernels were checked on Rome, multiplication, Python-code
generation, and the repeated-ID fixture. All 228 checked greedy predictions
matched the previous implementation, with identical input-token trajectories.
Across all vocabulary entries, maximum absolute logit difference was 7.94e-5
and maximum per-row RMSE was 8.88e-6. These FP32 kernels are not bit-identical
to the previous reduction order; the tested agreement does not establish
quality parity on arbitrary or long-context inputs.

### Batched scheduling and exact decode fusions

A second pass batches Hadamard, attention preparation and GDN front/scan,
uses contiguous MMA input loads and direct stores for complete tiles, and
fuses decode gate/up/SiLU and BF16 alpha/beta. The final version also removes
row-bound checks for complete standalone and fused GEMV groups. Three new alternating pairs
compare it with the first batched pass on the same M1 Max, PQ2 file and
4096-token context, now with 64 greedy decode steps per run:

| Prompt and phase | Before this pass, token/s | After this pass, token/s | Gain |
| --- | ---: | ---: | ---: |
| Rome chat, 19-token prefill | 17.633 | 22.441 | +27.3% |
| Decode after Rome chat | 13.403 | 14.817 | +10.5% |
| Repeated IDs, 128-token prefill | 27.605 | 38.498 | +39.5% |
| Decode after 128-token prompt | 14.319 | 14.648 | +2.3% |

All full-vocabulary logits and token trajectories were bit-identical across
the twelve runs. Separate multiplication and Python-code runs also matched
the first batched pass bit for bit over 65 and 97 vocabulary rows respectively.
These changes preserve the arithmetic used by that pass; they do not undo
the earlier change in reduction order relative to the original scalar kernel.
Metal API and shader validation passed, including batched state and tile tails.

The fused gate/up kernel measured about 1.18x throughput in two independent
eight-round comparisons. The subsequent full-row specialization improved that
fused kernel by another 1.127x in an isolated comparison, QKV by 1.096x,
Q by 1.133x and the vocabulary projection by 1.074x. Precomputing activation
coefficients and a uniform K loop did not justify additional paths or buffers.
These microbenchmark gains are not additive end-to-end percentages.
BF16 pairing varied from 1.02x to 1.10x, so its
contribution is smaller. Integer-prefix decoding, larger row unrolling and
GDN SRAM staging were slower and remain excluded. These results establish a
modest decode improvement, not a twofold end-to-end generation speedup.

A lightweight 64-token decode profile before the full-row specialization measured
71.00 ms/token overall, 68.72 ms on the GPU, 1.36 ms encoding commands,
0.009 ms submitting and 0.186 ms copying/checking logits. GPU duration is
contained in the CPU wait interval, so those two durations must not be added.
The profiling run's vocabulary logits were bit-identical to the normal build.
Eliminating command encoding alone therefore has only about a 2% upper bound.

Further isolated rewrites were rejected: ternary select/bit-sign decoding,
lossless aligned or tiled weight repacking, SRAM/shuffle lookup tables, and
matrix multiplication at batch one all lost against the current GEMV. Explicit
FMA calls produced identical outputs and essentially identical timings; pipeline
thread-limit hints also failed to improve performance. The MLX-style mask and
input-prescaling algebra reduced FP64 error but did not accelerate gate/down.
No extra full-model weight copy, precision reduction or lookup-table path from
these experiments is enabled.

For context, single runs of Prism's reference runtime on this same M1 Max and
PQ2 file measured 31.39 prefill / 15.15 decode token/s on the 19-token prompt
with FP32 KV and flash attention disabled, and 82.86 / 16.01 on the 128-token
fixture with its default FP16 KV / automatic flash attention. These are not
controlled median A/B results, and the latter uses different cache/attention
settings. They show that there is still substantial prefill work to do.
The [vendor's 46.8 token/s result](https://prismml.com/news/bonsai-2-27b)
is for M5 Max, not this M1 Max.

### FP4 experiment

A model-free experiment repacked PQ2 coefficients losslessly into E2M1
nibbles, retaining the original FP16 scale per 128 weights. This is a custom
66-byte block layout, not an NVFP4/MXFP4 file conversion. It uses 94.1% more
weight bytes than PQ2. All PQ2 code combinations and all finite FP16 scales
were checked for exact coefficient preservation.

On M1 Max, eight rotating measurement rounds compared the same NR8/NSG2
decode geometry. Even the fastest FP4 decoder, specialized for the four codes
that PQ2 can produce, was slower:

| Projection | PQ2, microseconds | Best FP4, microseconds | FP4 time / PQ2 time |
| --- | ---: | ---: | ---: |
| FFN gate/up, 17408 x 5120 | 154.7 | 253.1 | 1.64x |
| FFN down, 5120 x 17408 | 169.0 | 285.2 | 1.69x |
| Vocabulary, 248320 x 5120 | 2077.2 | 3406.3 | 1.64x |

Changing only the prefill tile's weight loader also regressed: at 32 tokens,
gate/up took 2298.4 microseconds with PQ2 versus 2463.8 with specialized FP4
(+7.2% time), and down took 3244.5 versus 3422.5 (+5.5%). The general E2M1
decoder was slower still. Full prefill outputs were bit-identical between
the FP4 and PQ2 loaders. Metal API and shader validation passed, including
partial output rows and 19/33-token tails.

These are isolated kernel measurements, not full-model FP4 generation rates.
FP4 storage was expanded in software for FP32 computation; this did not test
native FP4 arithmetic. The experiment is not enabled in the runtime and did
not create or modify a GGUF file.

## Earlier baseline

On Apple M1 Max (32 GiB), three alternating pairs of runs compared the initial
native scalar-decode Metal kernel with the block-decode kernel and shared
Hadamard transforms. The same local PQ2 GGUF, 19-token non-thinking Rome chat
prompt, 16 greedy decode steps and 4096-token context were used. Only one model
instance ran at a time; weights were warm in the file cache. Median rates:

| Phase | Initial native implementation | Optimized | Speedup |
| --- | ---: | ---: | ---: |
| Prefill | 1.700 token/s | 7.181 token/s | 4.22x |
| Decode | 1.652 token/s | 7.234 token/s | 4.38x |

All 17 full-vocabulary logit rows were **bit-identical** across all six runs.
These historical measurements used sequential prefill and are not a comparison
against Prism's speed. Current batched prefill is described above.
The HTTP server smoke test returned the expected model ID and generated the
same non-thinking response, at approximately 7.2 decode token/s.

`bench-bonsai-pq2` measures eight alternating A/B rounds on representative
projection shapes, reports GPU timestamps as CSV, and requires bit-identical
outputs relative to the initial scalar decoding kernel. The full-model driver
prints separate prefill and decode timings; pass `--logits FILE` to retain all
vocabulary values for a correctness comparison.
