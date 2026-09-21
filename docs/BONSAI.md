# Ternary Bonsai 2 27B

This branch provides native text inference and static-image input for the Prism
[Ternary-Bonsai-2-27B GGUF](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
on Metal, plus a scalar CPU text reference implementation. Both PQ2_0 and
PTQ1_0 language checkpoints are supported. It does not invoke or
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
150 MiB for GDN state/history, and scratch buffers including 65.55 MiB for
batched prefill. Metal checks the aggregate session allocation against the
device's recommended working set.
These figures cover the language session; image encoding additionally uses
the projector weights and vision workspace.

To use PTQ1_0, substitute `gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf` for the
model path. Both language files also work with `--cpu` for text inference.
PTQ1_0 has dedicated block decoding and fused Metal gate/up kernels. Its
prefill kernels reuse activations across output rows and coefficients across
up to eight tokens, preserving the original scalar FP32 sum order. PQ2_0 uses a separate
matrix-tiled prefill path; PTQ1_0 does not repack the model into PQ2 or expand
weights persistently.

```sh
./ds4-server --metal \
  -m gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --ctx 4096 --host 127.0.0.1 --port 8000
```

The OpenAI-compatible model ID is `ternary-bonsai-2-27b`.
Text chat uses Qwen ChatML, with the checkpoint's thinking/non-thinking prefix.
The Qwen Flash Next reasoning-effort system instruction is not inserted for
this model.

## Images on Metal

Pass either matching projector with `--vision`, alongside either PQ2_0 or
PTQ1_0 as the language model:

| File | Role |
| --- | --- |
| `Ternary-Bonsai-2-27B-mmproj-BF16.gguf` | BF16 vision tower and projection weights |
| `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` | Q8_0 vision weights with F16 FFN-down matrices |

These projector files are accessories to the language checkpoint, not
standalone models. Image inference requires Metal; CPU support covers text.
For the interactive CLI:

```sh
./ds4 --metal \
  -m gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --vision gguf/Ternary-Bonsai-2-27B-mmproj-BF16.gguf \
  --ctx 4096 --nothink --temp 0
```

At the prompt, use `/read tests/vision-fixtures/qwen38/maple.png`, then ask
follow-up questions in the same conversation. PNG and JPEG static images are
supported. The shared Qwen3-VL encoder resizes images to a grid of 16-pixel
patches, merges 2x2 patches, and produces 5120-component embedding rows.
It defaults to 64–1024 image tokens; `DS4_QWEN4_IMAGE_MAX_TOKENS` controls the
upper bound. BF16 weights widen directly to FP32 in the existing vision
matrix kernel, including the 4304-wide FFN tail; F32, F16 and Q8_0 paths remain
available.

For HTTP, start:

```sh
./ds4-server --metal \
  -m gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --vision gguf/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf \
  --ctx 4096 --host 127.0.0.1 --port 8000
```

OpenAI chat accepts `image_url` content parts containing PNG/JPEG data URIs.
For example, from another terminal:

```sh
python3 - <<'PY' | curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' --data-binary @-
import base64, json
from pathlib import Path
data = base64.b64encode(Path("tests/vision-fixtures/qwen38/maple.png").read_bytes()).decode()
print(json.dumps({
    "model": "ternary-bonsai-2-27b", "temperature": 0, "max_tokens": 64,
    "think": False,
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Read the text in this image."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}}
    ]}]
}))
PY
```

Remote image URLs and server-side file paths are rejected. See
[server image input](SERVER.md#images) for the request limits.

Image rows already occupy the language model's original embedding space.
They bypass the folded token-embedding lookup and its inverse Hadamard;
subsequent folded projections still apply their declared transforms.
Full-attention layers use interleaved MRoPE sections `[11, 11, 10]` with
temporal, row and column coordinates. The rotary counter advances by the
image grid's maximum dimension, while causal KV positions continue to count
every token. Text continuation retains this offset; reset and rewind clear it
before replay. GDN continues to process all text and image rows in order.

Live image-prefix reuse checks token spans, source-image fingerprints, grid
width/height and layout. Changing the grid with the same fingerprint therefore
rebuilds the state. Invalid prompts, inconsistent grids and nonfinite embedding
rows are rejected before prefill changes the session.

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

PTQ1_0 also decodes one 128-weight block per iteration. The byte indices and
powers of three for each lane are computed outside the block loop; each scale
is loaded once for four coefficients. The integer byte product wraps modulo
256 before extracting each trit, including the final eight coefficients.
The output still uses the original lane assignment, multiplication/addition
order and final SIMD reduction. Activations and accumulators stay FP32.

For PTQ matrices with at least 4096 output rows and batches of at least four
tokens, each SIMD group computes two rows by four or eight tokens. Batches
of eight or more tokens use eight-token tiles; complete tiles skip bounds
checks in the K loop, and partial tiles retain bounded loads and stores.
Compatible gate/up
projections instead share input loads and publish the SiLU product directly:
two row pairs per SIMD group for decode, one pair by four or eight tokens for prefill.
These fusions require matching types, shapes and Hadamard sign tables.
Smaller projections, short batches and incompatible siblings retain the
separate block-decoding path. The weights remain in their 28-byte packed
blocks and the session's allocation sizes are unchanged.

Fused PTQ decode gate/up uses an exact 512-byte constant table containing
the five trit codes for each possible stored byte. Lane-specific shifts
select the required codes, retaining wrap-before-extraction semantics for
all 256 byte values. This lookup is restricted to decode gate/up; other
projections retain arithmetic extraction. The standalone PTQ decode loop
is unrolled four blocks at a time, without reordering its FP32 additions.

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

For compatible PQ2 prefill gate/up matrices and chunks of at least sixteen
tokens, a separate tiled fusion now computes both projections and SiLU in
one dispatch. A physical 64-row tile holds 32 matching gate/up rows. It keeps
the existing FP32 matrix accumulation order and eight accumulators per SIMD
group, then reuses weight scratch for the activation. Gate/up intermediate
vectors are not written or read on this path. Declared threadgroup scratch
remains 12 KiB; short chunks and incompatible shapes or sign tables retain
the separate projection path.

For complete groups, separate template instantiations remove row-bound checks
inside these GEMV loops: sixteen rows per standalone threadgroup and eight
per fused gate/up threadgroup. The host selects them only when the row count
is divisible by the corresponding group size. The general kernels retain all
checks for partial groups; both versions share the same arithmetic body.

Metal prefill processes up to 128 tokens per chunk in layer order, bounded by
`--prefill-chunk` and the remaining prompt. Chunks larger than 32 are rounded
down to a multiple of 32; the final short remainder is handled separately.
This coalesces complete legacy blocks while preserving the scalar/four-token
projection path of their original tail. Explicit chunk sizes of 32 or less
keep their previous schedule. The fixed per-session batch activation buffers
use 65.55 MiB at capacity 128, versus 16.39 MiB at capacity 32.
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
arithmetic order and FP32 state. Smaller chunks retain per-token dispatches.

The checkpoint's 128-dimensional GDN state has specialized prefill and decode
kernels. Each value lane loads its state column into a fixed-size private
array, executes the two recurrence passes in the original ascending key
order, and writes state after the complete scan. Prefill keeps the state
across all tokens in the chunk. FP32 arithmetic, convolution, normalization
and causal token order are retained. Other state dimensions use the generic
kernel.

For unrotated BF16 alpha/beta projections, chunks of at least four tokens
share a paired four-token kernel. It reuses input loads across the two
projections, retaining each output's original lane/K order and SIMD sum.
Other types, rotated weights and short chunks keep the separate path.

Full-attention prefill groups up to eight queries into three dispatches:
scores, softmax and the weighted value sum. Each query retains its original
causal bound, FP32 dot-product order, 256-thread softmax tree and ascending
value accumulation. This is complete causal attention, without sparse masks.
The reusable score workspace is capped at 32 MiB by reducing the query batch
at long contexts; if one query already exceeds that cap, it retains the
original single-query allocation. At context 4096 the checkpoint uses 3 MiB
of score workspace rather than 384 KiB. Decode retains its single-query path.

For PQ2 matrices with at least 4096 output rows and chunks of at least sixteen
tokens, prefill uses a 64-output by 32-token by 32-K matrix tile, following the
packed shared-memory layout used by the existing DeepSeek Metal kernels.
Each thread loads its scale and four packed words once per 128-coefficient
quantization block. Four 32-K passes reuse these registers, with aligned
16-bit loads accommodating the 34-byte PQ2 blocks. Both the standalone and
paired gate/up tiles retain the original sequence of eight-wide matrix
accumulations. Matrix operands and accumulators stay FP32; weights are
expanded only in 12 KiB of threadgroup scratch, not into a persistent
full-precision matrix. Contiguous `float4` activation loads feed the tile.
Complete standalone projection tiles store directly to device memory;
partial tiles retain a bounds-checked scratch path. Paired gate/up tiles
stage their outputs in scratch to apply SiLU. Matrix accumulation
changes association relative to GEMV. Smaller projections and chunks use a
four-token kernel that preserves each token's original reduction, or serial
GEMV for chunks below four tokens. CPU prefill remains sequential.

Each Metal chunk completes before its session position advances. Invalid
token batches are rejected before dispatch; a GPU failure invalidates the
checkpoint so a subsequent synchronization rebuilds recurrent state.

## Scope

- CLI and HTTP text generation, incremental prompt reuse, independent sessions,
  reset and prefix rebuilding are supported.
- Static PNG/JPEG images are supported on Metal with either language format
  and either matching BF16 or Q8_0 projector, through CLI `/read` and HTTP
  `image_url` data URIs. Video and audio input are outside this implementation.
- Rewind invalidates the recurrent checkpoint; synchronize the retained prompt
  before evaluating more tokens. Changing a prompt prefix rebuilds its state.
- MTP, CUDA/ROCm, SSD expert streaming, distributed execution,
  directional steering and serialized session/KV caches are not implemented.
- PQ2_0 and PTQ1_0 are native language formats on CPU and Metal, with separate
  optimized Metal projection paths. Their performance measurements below
  refer to the named checkpoint and are not interchangeable.

## Validation

```sh
make test-bonsai             # codecs, Hadamard, graph and malformed inputs
make test-bonsai-metal       # native kernels and CPU/Metal graph parity
make tests/test_bonsai_model # optional real-model API/lifecycle runner
make tests/test_bonsai_rows tests/test_qwen4_vision_bf16
make tests/test_bonsai_vision_model # real-model image/session runner
make bench-bonsai-pq2        # original vs optimized PQ2 kernels, no model needed
make bench-bonsai-bf16       # typed BF16 alpha/beta projections
make bench-bonsai-elementwise # exact norm/Hadamard checks and GPU timings
make bench-bonsai-mm         # four-token projections versus repeated GEMV
make bench-bonsai-mma        # FP32 tiled PQ2 prefill versus four-token kernel
make bench-bonsai-ptq        # frozen generic PTQ versus block and batched kernels
./tests/test_bonsai_ptq --bench-fused # PTQ gate/up/SiLU GPU timings
./tests/test_bonsai_pairs --bench # exact gate/up and BF16 decode fusions
./tests/test_bonsai_fullrows --bench # complete-row specialization vs frozen kernels
./tests/test_bonsai_gdn_prefill --bench # generic vs register-state recurrence
./tests/test_bonsai_prefill_pair --bench # tiled prefill gate/up/SiLU fusion
./tests/test_bonsai_bf16_pair_batch --bench # paired GDN alpha/beta prefill
./tests/test_bonsai_pq2_prefill_load --bench # PQ2 block-load reuse
./tests/test_bonsai_attention_batch --bench # serial vs batched causal attention
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 ./tests/test_bonsai_rows
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 ./tests/test_qwen4_vision_bf16
./tests/test_bonsai_vision_model \
  gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  gguf/Ternary-Bonsai-2-27B-mmproj-BF16.gguf \
  tests/vision-fixtures/qwen38/maple.png --lifecycle
```

For the format/vision addition on 2026-09-20, the model-free embedding-row and
MRoPE tests passed, as did BF16 dense-matrix and synthetic 27-layer vision
encoder tests with Metal API and shader validation. The BF16 tests compare
against exactly expanded F32 weights and cover split/unsplit accumulation,
tails, offsets, BF16 range, FFN width 4304 and output width 5120.
All four PQ2_0/PTQ1_0 × BF16/Q8_0 combinations read `MAPLE 8153` from the
maple fixture. Image-cache reuse, invalid-input atomicity, grid changes,
reset and rewind/replay checks passed. The CLI also read the image with its
default 300 image tokens, spanning multiple 128-token chunks. An HTTP
`image_url` request returned `MAPLE 8153`; a subsequent question returned
`8153` while the server reused both the image embedding and live KV state.
This is one image fixture, not a general vision-quality benchmark.

The shared vision encoder was compared with Prism's CPU implementation on
the same 54 image tokens. With identical normalized input pixels, the BF16
embedding cosine was 0.99999424 (RMSE 0.002682); Q8_0 was 0.99991836
(RMSE 0.010244). The reference uses lower-precision intermediate dot-product
inputs and a different merger GELU approximation; bitwise agreement is not
expected. Its default image preprocessing also adds padding, whereas ds4's
existing Qwen preprocessing resizes directly. Using each runtime's default
preprocessing gave cosines 0.946733 and 0.947465 respectively. The higher
same-input figures validate the encoder comparison, not equivalence of the
two complete image pipelines. `tests/bonsai_vision_reference.cpp` makes both
comparisons reproducible without linking Prism into ds4.

Text-only PQ2_0 logits remained bit-identical to the pre-vision implementation
for a 128-token prompt plus 32 decode steps; PTQ1_0 matched for a 19-token
prompt plus eight steps. Both complete Bonsai suites and the shared Qwen
kernel suite passed after the changes.

The optional vision lifecycle runner also checks malformed prompts before
span access, finite embeddings, cache identity and grid changes with the same
fingerprint against cold replay. Substitute the language/projector paths to
exercise another supported combination. Run real-model tests one process at
a time.

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

### PTQ1_0 block decoding and exact fusions (2026-09-21)

Baseline: `bfb488f`, before the dedicated PTQ block loaders and fusions.
Both binaries used `Ternary-Bonsai-2-27B-PTQ1_0.gguf` on Apple M1 Max
(32 GiB), context 4096, chunk 128, and 32 greedy decode steps. The short
fixture is the 19-token non-thinking chat prompt `narrami la storia di roma`;
the 128-token fixture repeats and truncates those same prompt IDs. Three
alternating baseline/candidate pairs ran for each fixture,
with one model process at a time. Median rates:

| Prompt tokens | Phase | Baseline, token/s | Optimized, token/s | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 19 | prefill | 3.489 | 14.053 | 4.03x |
| 19 | decode | 1.627 | 7.570 | 4.65x |
| 128 | prefill | 3.759 | 15.172 | 4.04x |
| 128 | decode | 1.602 | 7.019 | 4.38x |

All captured full-vocabulary logits and greedy token IDs matched the
baseline bit for bit in all six pairs. Prefill timing includes copying and
saving its frontier logits; decode timing covers session evaluation only.
These are local results for PTQ1_0, not forecasts for other hardware or a
comparison with a different quantized checkpoint.

The gain comes from hoisting PTQ byte/trit metadata out of the block loop,
sharing coefficient/input loads, and fusing gate/up with SiLU. The packed
weights, FP32 activations, allocation sizes and per-result reduction order
are retained. Multirow standalone decode variants were measured and omitted
because they did not consistently improve on the block decoder.

`make test-bonsai-metal` includes the independent PTQ test, with scalar
MV/MM and SwiGLU references frozen from `bfb488f`. It covers all 256 stored
byte values at every trit position, scale extremes, zero rows/tokens,
partial and empty dimensions, real FFN shapes, read-only inputs and guarded
outputs. The complete CPU/Metal suite and the PTQ test with Metal API and
shader validation passed. To reproduce the real-model short fixture:

```sh
./tests/test_bonsai_model \
  --model gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf --backend metal \
  --text 'narrami la storia di roma' --chat \
  --ctx 4096 --chunk 128 --decode 32 --logits /tmp/ptq-logits.f32
```

### PTQ1_0 eight-token tiles and decode lookup (2026-09-21)

A second PTQ pass reuses each decoded coefficient across eight prefill tokens
and specializes complete tiles to remove bounds checks from the block loop.
Four-token tiles remain available for smaller batches. Fused decode gate/up
uses an exact 512-byte trit table; standalone PTQ decode unrolls four blocks
without changing the FP32 accumulation order. No extra model-weight copy or
change to the GGUF is required.

The baseline for this comparison is the preceding PTQ block/fusion pass,
not `bfb488f`. Three new alternating pairs per fixture used the same M1 Max,
PTQ1_0 file, context 4096, chunk 128 and 32 greedy decode steps. Median rates:

| Prompt tokens | Phase | Before this pass, token/s | After this pass, token/s | Gain |
| ---: | --- | ---: | ---: | ---: |
| 19 | prefill | 13.157 | 14.998 | +14.0% |
| 19 | decode | 7.192 | 7.428 | +3.3% |
| 128 | prefill | 15.674 | 21.074 | +34.5% |
| 128 | decode | 7.315 | 7.408 | +1.3% |

A separate three-pair ablation kept the eight-token prefill in both binaries
and compared decode with and without the lookup/unrolling changes. With the
128-token fixture and 64 decode steps, median decode rose from 7.305 to
7.793 token/s (+6.7%). Individual paired gains varied from +0.2% to +10.5%;
this is a smaller and less stable benefit than the prefill improvement.
It is not an additional percentage to multiply by the table's decode gains.

All full-vocabulary logits and greedy token trajectories were bit-identical
within every pair in both experiments. The independent PTQ test also checks
the new eight-token full and partial tiles against the frozen scalar reference,
including all 256 byte encodings in both sides of fused gate/up, real FFN
shapes and guarded tails. The CPU/Metal suite and PTQ API/shader validation
passed. These results cover the tested fixtures, not arbitrary long contexts.

### PQ2 load reuse and larger exact chunks (2026-09-20)

This second pass combines PQ2 block-load reuse, paired BF16 alpha/beta
projections and a 128-token prefill cap. Its baseline is the preceding exact
prefill/recurrent-state pass below, already including its fused gate/up,
register-state GDN and batched attention. These are incremental measurements,
not a comparison against the original remote commit.

On Apple M1 Max (32 GiB), both versions used the same PQ2 model, context 4096,
configured chunk 128 and 32 greedy decode steps. The baseline's effective cap
was 32; the candidate's was 128. Both were warmed, then three alternating
pairs ran per prompt with one model process at a time. Median rates:

| Prompt tokens | Phase | Previous pass, token/s | This pass, token/s | Change |
| ---: | --- | ---: | ---: | ---: |
| 19 | prefill | 24.737 | 27.024 | +9.2% |
| 19 | decode | 16.364 | 16.459 | +0.6% |
| 128 | prefill | 46.317 | 58.895 | +27.2% |
| 128 | decode | 16.006 | 16.256 | +1.6% |
| 512 | prefill | 44.943 | 56.456 | +25.6% |
| 512 | decode | 14.488 | 14.483 | -0.0% |

All 18 runs matched bit for bit across the prefill frontier and 32 subsequent
full-vocabulary logit rows, including greedy token trajectories. The 19-token
input is the formatted Rome chat prompt; the larger inputs repeat those IDs
and measure throughput rather than answer quality. Prefill includes frontier
copy/save overhead; decode timing covers evaluation only. The decode kernels
were unchanged in this pass; the small decode differences do not establish a
new decode speedup. These timings do not predict other devices or prompts.

Additional full-model comparisons passed bit for bit for prompt lengths 33,
47, 48, 64, 65, 79, 111, 112, 129 and 2049 followed by four decode steps; for
129-token prompts with configured chunk sizes 8, 16, 48 and 64; and for
arithmetic/Python chat prompts followed by sixteen decode steps. Public
session lifecycle tests passed their existing tolerances. The CPU/Metal
Bonsai suites passed, including 43 GDN fixtures through 129 tokens, 132 BF16
pair fixtures, mixed graph chunks through 128 tokens, and frozen PQ2 tile
oracles with nonzero output tails. New BF16 and PQ2 paths also passed Metal
API/shader validation. These are scoped correctness checks, not proof of
quality for every prompt.

The chunk-only experiment, before the two kernel changes, measured medians
of 46.345, 53.166 and 54.639 token/s at effective caps 32, 64 and 128 for the
128-token fixture, with identical logits. It used three rotating measured
rounds after a warm-up. The full-model table above measures all three changes
together; their gains must not be added.

The wider M32/N64 and K64 matrix tile experiments were slower on this device
and were discarded. The retained tile still has eight accumulators per SIMD
group and 12 KiB of threadgroup scratch. No persistent expanded weight matrix
or lower-precision activation/state was introduced.

### Exact prefill and recurrent-state pass (2026-09-20)

Three alternating baseline/candidate pairs per prompt compared this pass with
commit `3add166` on Apple M1 Max (32 GiB), using the same PQ2 GGUF and context
4096. The configured prefill chunk was 128 (effective Bonsai cap 32), followed
by 32 greedy decode steps. Both binaries were warmed first; only one model
process ran at a time. Median rates:

| Prompt tokens | Phase | Baseline, token/s | Optimized, token/s | Gain |
| ---: | --- | ---: | ---: | ---: |
| 19 | prefill | 21.770 | 24.486 | +12.5% |
| 19 | decode | 14.906 | 15.882 | +6.5% |
| 128 | prefill | 37.071 | 45.477 | +22.7% |
| 128 | decode | 14.811 | 15.537 | +4.9% |
| 512 | prefill | 34.295 | 44.079 | +28.5% |
| 512 | decode | 12.992 | 14.037 | +8.0% |

All candidate runs matched the corresponding baseline bit for bit, covering
33 full-vocabulary logit rows per run and their greedy token trajectories
in the 18-run comparison. The 19-token input is the non-thinking
Rome chat prompt; 128 and 512 tokens repeat its IDs as throughput fixtures,
not quality datasets. Prefill timing includes copying/saving the frontier
logits; decode timing covers evaluation only. These are local measurements,
not guarantees for other devices or prompts.

Additional baseline/candidate checks passed bit for bit for 513- and
2049-token repeated-ID prompts followed by four decode steps, arithmetic and
Python chat prompts followed by sixteen steps, and 128-token prompts with
chunk sizes 8 and 16. Public session reset, prefix changes, invalidation and
rewind/replay also passed the existing lifecycle tolerances. Those lifecycle
checks compare different ingestion schedules, which need not be bit-identical.

The GDN, paired prefill projections and batched attention also have independent
kernel checks against the prior arithmetic, with Metal API/shader validation,
canaries and chunk/causality checks. The graph test exercises score-workspace
reuse with a synthetic context that reduces the attention group to six queries.

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
