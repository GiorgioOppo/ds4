<p align="center">
  <img src="logo.svg" alt="DwarfStar logo" width="220">
</p>

**DwarfStar** is a small native inference engine optimized first for
**DeepSeek V4 Flash** (including the experimental vision model).
It also supports **GLM 5.2 and 5.3**, **GLM 5.3 Flash**, and,
on very high-memory machines, **DeepSeek V4 PRO**. It is self-contained and
deliberately narrow, not a general GGUF runner. Model loading, prompt rendering,
tool calls, KV state, the HTTP server, and the coding agent are built and tested together.
The repository also includes tools and data for GGUF, imatrix, quality, and speed.

Supported backends:

* **Metal**, the primary target, on Macs with 96 GB or more. Smaller machines
  can use SSD streaming.
* **NVIDIA CUDA**, including multi-GPU systems and DGX Spark.
* **ROCm** on Strix Halo systems such as the Framework Desktop.

This project would not exist without **llama.cpp and GGML**, make sure to read
the acknowledgements section, a big thank you to Georgi Gerganov and all the
other contributors.

Model support is intentionally opportunistic. The project follows the best open
weights for useful local machine sizes, especially 128 GB laptops and 512 GB
workstations. A model may be removed when a better replacement arrives.

The project has first class support for SSD streaming of weights, so it is
possible to run models bigger than RAM while often still getting decent
performances, and even running very large models (like the full GLM 5.3 or DeepSeek v4 PRO)
on systems with just 128GB of RAM at a slower speed, but fast enough for
QA-style chats.

# So, what can I do with this software?

* You can run a very capable models in your consumer hardware, a MacBook, a DGX Spark, or a Strix Halo for example. Even if you have not enough RAM, with SSD streaming, you can run it at a decent speed.
* Using the CUDA multi-GPU support and with ds4-server micro batching of decoding and generation, you can turn a server with old-ish CUDA cards (Ada Lovelace architecture), no longer supported for new models by vLLM, into a multi-user LLM server for your company. We tested this setup with 8xL40S NVIDIA cards and multiple sessions with very good results. 120 t/s aggreated generation, 2000 t/s prefill.
* Using two MacBook M5 Max / M3 Ultra RDMA, you can run 4 bit DeepSeek Flash, GLM 5.2, or GLM 5.3 Flash with tensor parallelism.
* You can also use pipeline paralellism to glue together multiple systems to sum their RAM and run larger models.

## Motivations

* Capable open-weight models now fit on high-end personal machines.
* DeepSeek V4 Flash and PRO, GLM 5.2, tolerate aggressive routed-expert quantization.
* Compressed KV caches and fast local SSDs make long contexts practical.
* The idea of an inference system specialized for a few models.

# AI full disclosure

* This software is developed with **strong assistance from GPT 5.5, 5.6, Claude Fable** and with humans leading the ideas, testing, and debugging. We say this openly because it shaped how the project was built. If you are not happy with AI-developed code, this software is not for you. The acknowledgement below is equally important: this would not exist without `llama.cpp` and GGML, largely written by hand.

## Acknowledgements to llama.cpp and GGML

`ds4.c` does not link against GGML, but it **exists thanks to the path opened by the
llama.cpp project and the kernels, quantization formats, GGUF ecosystem, and hard-won
engineering knowledge developed there**.
We are thankful and indebted to [`llama.cpp`](https://github.com/ggml-org/llama.cpp)
and its contributors. Their implementation, kernels, tests, and design choices were
an essential reference while building this DeepSeek V4 specific inference path.
Some source-level pieces are retained or adapted here under the MIT license: GGUF
quant layouts and tables, CPU quant/dot logic, and certain kernels. For this
reason, and because we are genuinely grateful, we keep the GGML authors copyright
notice in our `LICENSE` file.

## Status

The software is currently very fast changing. Consider it beta quality.
Before each release, a big QA run is executed, however instabilities
are definitely possible.

# How to use this project?

I (Salvatore) believe that the way projects should be shipped and used changed because of AI. The main differences today are:

1. With AI, users can modify the software in significant ways with low efforts, costs, and even lacking deep domain knowledge about the task they want to accomplish. For instance, a DwarfStar user with a specific hardware setup can ask a coding agent to improve the inference speed of this software for the specific hardware setup, asking the model to reach the maximum prefill and generation speed without impacting correctness, and also asking to do a deep QA pass.
2. Similiarly, because of "1", software may be shipped in a different way than before. It must be more a working template for the biggest use cases, without trying to cover every possible setup. If DwarfStar showcases a few good implementations of tensor parallel execution, the code will work as a rail for implementing the same feature in specific conditions, for a new model, and so forth.

So, while this project attempts to be usable for the featured models and the most common hardware setups, I ask you, if you have access to coding agents, to consider using coding agents as an interface to discover the project, make modifications, create personalized setups. This way you can likely do more than what we ship, and certain things that are not documented or implemented, and that you require, are potentially very easy to achieve.

## More Documentation

If you are looking for very specific things, we have other
sub-README files. Otherwise for normal usage keep reading the
next sections.

- [CONTRIBUTING.md](CONTRIBUTING.md): correctness and speed regression testing
  guide for contributors. **Read this before sending a pull request**.
- [QA_BEFORE_RELEASES.md](QA_BEFORE_RELEASES.md): the complete release test
  matrix, including the remote Metal, CUDA, and ROCm machines.
- [ENVIRONMENT_VARIABLES.md](ENVIRONMENT_VARIABLES.md): complete runtime,
  test, and tooling environment-variable inventory, with a curated quick
  reference for supported rollback, fail-closed, and diagnostic switches.
- [gguf-tools/README.md](gguf-tools/README.md): offline GGUF generation,
  imatrix collection, quantization tooling, and quality checks.
- [gguf-tools/imatrix/README.md](gguf-tools/imatrix/README.md): how the
  routed-MoE imatrix is collected and used.
- [gguf-tools/imatrix/dataset/README.md](gguf-tools/imatrix/dataset/README.md):
  how the calibration prompt corpus is generated.
- [gguf-tools/quality-testing/README.md](gguf-tools/quality-testing/README.md):
  how local GGUFs are scored against official DeepSeek V4 Flash/PRO continuations.
- [dir-steering/README.md](dir-steering/README.md): directional steering data,
  vector generation, and usage.
- [speed-bench/README.md](speed-bench/README.md): benchmark commands, charts,
  and CSV generation.
- [tests/test-vectors/README.md](tests/test-vectors/README.md): official
  continuation vectors used for regression checks.

## Model Weights

This implementation only works with the DeepSeek V4 and GLM GGUFs listed
below. It is not a general GGUF loader, and arbitrary GGUF files will not have
the tensor layout, quantization mix, metadata, or optional MTP state expected by
the engine. The 2 bit quantizations provided here are verified to be actually
high quality: they behave well, work under coding agents, call tools in a reliable way.

The 2 bit quants use a very asymmetrical quantization: only the routed MoE
experts are quantized, up/gate at `IQ2_XXS`, down at `Q2_K`. They are the
majority of all the model space: the other components (shared experts,
projections, routing) are left untouched to guarantee quality.

Download one main model. **Prefer the imatrix versions.**

```sh
./download_model.sh ds4f-q2      # 96/128 GB RAM machines
./download_model.sh ds4f-q2-q4   # q2 with the last 6 expert layers at q4
./download_model.sh ds4f-q4      # >= 256 GB RAM machines
./download_model.sh ds4f-mxfp4   # native MXFP4 experts, about 156 GB
./download_model.sh pro-q2-imatrix  # 512 GB RAM machines, PRO 0813 q2 imatrix
```

The MXFP4 GGUF preserves DeepSeek's released MXFP4 routed-expert weights rather
than requantizing them. It runs on Metal and CUDA; Blackwell CUDA devices use
native FP4 matrix instructions and FP4 activations for batched expert work.
Decode and other CUDA devices use Q8 activations.

For the full PRO Q4 distributed run, download one half on each machine:

```sh
./download_model.sh pro-q4-layers00-30      # first half of PRO Q4 split
./download_model.sh pro-q4-layers31-output  # second half of PRO Q4 split
```

The script stores files under `./gguf/` and updates `./ds4flash.gguf` to point
at the selected main model. DeepSeek files come from
`antirez/deepseek-v4-gguf`; GLM targets use the repository named in the script's
help. Smaller files resume with `curl -C -`, while large files use the official
Hugging Face downloader.
The `pro-q4-layers00-30`, `pro-q4-layers31-output`, and `pro-q4-split` targets
download distributed PRO Q4 pieces and do not update `./ds4flash.gguf`.
Authentication is optional for public downloads, but `--token TOKEN`,
`HF_TOKEN`, or the local Hugging Face token cache are used when present.

If you want to regenerate GGUF files or collect a new imatrix, see
[gguf-tools/README.md](gguf-tools/README.md). Those tools are meant for offline
model-building work and can take a long time on the full DeepSeek weights.
Flash and PRO GGUF generation are supported by the local tools. PRO conversion
uses a compatible published PRO GGUF as its metadata, tensor-layout, and output
type template.

GLM 5.2 support is limited to the GGUF files tested by this branch:

```sh
./download_model.sh glm-unsloth-q4  # Unsloth UD-Q4_K_XL, 11 shards
./download_model.sh glm-antirez-iq2xxs  # antirez routed IQ2_XXS single-file GGUF
./download_model.sh glm-antirez-q2  # antirez routed Q2_K single-file GGUF
./download_model.sh glm-antirez-q4  # antirez routed Q4_K single-file GGUF
```

GLM 5.3 Flash has its own graph, artifacts, and run instructions in the
[GLM 5.3 Flash](#glm-53-flash) section below.

The full GLM 5.3 Q2 model is about 197 GiB. It can run resident on a 256 GB
machine, or with SSD streaming on a smaller system:

```sh
./download_model.sh glm53-full-q2
./ds4 -m gguf/GLM-5.3-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf --ssd-streaming
```

The supported GLM 5.2 layout keeps dense/model-control tensors in the existing
Q8/F32 paths and supports routed expert gate/up tensors in `Q2_K`, `Q4_K`, or
`Q5_K`; routed expert down tensors are supported in `Q2_K`, `Q4_K`, `Q5_K`, or
`Q6_K`. Other GLM GGUF quant layouts should be treated as unsupported until they
are added deliberately and scored against the official 100-case fixture.

These GLM 5.2 formats do not all support the same execution modes. The Q4 files
work for normal Metal and CUDA inference. Two-Mac tensor parallelism for GLM
5.2 currently requires an ownership-aware IQ2_XXS or Q2_K routed layout. GLM
5.3 has its own ownership-aware Q4 path.

GLM's MTP block is part of the main GGUF; it does not use the separate Flash
MTP file. Ordinary decode remains the default. `--mtp` enables experimental
greedy speculation. `--mtp-timing` also enables it and prints acceptance
and timing counters:

```sh
./ds4 -m gguf/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf \
  --mtp-timing --temp 0
```

GLM 5.2 uses the Metal, CUDA, or ROCm graph backend. GLM 5.3 is validated on
Metal, with Q2 also validated on CUDA. Directional steering, `--power` below
100, an explicit `--prefill-chunk`, and an external `--mtp-model` file are not
supported for GLM yet.

Then build:

```sh
make                  # macOS Metal
make cuda-spark       # Linux CUDA, DGX Spark / GB10
make cuda-generic     # Linux CUDA, other local CUDA GPUs
make strix-halo       # Linux ROCm, AMD Strix Halo
make cpu              # CPU-only diagnostics build
```

For ROCm packages, GTT configuration and the reproducible ROCm 10.0 container build, see [DS4 on Strix Halo](STRIXHALO.md).

`./ds4flash.gguf` is the default model path used by both binaries. Pass `-m` to
select another supported GGUF from `./gguf/`. Run `./ds4 --help` and
`./ds4-server --help` for the full flag list.

## DeepSeek V4 Flash Vision Experimental

Vision-Exp is a separate DeepSeek checkpoint, not the 0731 text model. It uses
a matching language GGUF and a 0.9 GiB vision encoder. The Q2 model is the
recommended version for 96 and 128 GB systems:

```sh
./download_model.sh ds4f-vision-q2
./ds4 --vision gguf/DeepSeek-V4-Flash-Vision-Encoder.gguf
```

Use `/read image.png` in the interactive CLI. The same `--vision` option gives
`ds4-agent` its `view_image` tool and enables image input through `ds4-server`.
PNG and JPEG are supported. The Q2 path is validated on Metal, single-GPU CUDA
including DGX Spark, and ROCm. Larger `ds4f-vision-q2-q4` and
`ds4f-vision-mxfp4` downloads are also available.

Vision-Exp has its own DSpark checkpoint. Do not use the 0731 support model:

```sh
./download_model.sh ds4f-vision-dspark
./ds4 --vision gguf/DeepSeek-V4-Flash-Vision-Encoder.gguf --dspark \
  --mtp-model gguf/DeepSeek-V4-Flash-Vision-Exp-DSpark-support.gguf
```

## GLM 5.3 Flash

GLM 5.3 Flash uses a separate graph for its recurrent KDA layers, sparse DSA
layers, hyper-connections, and built-in MTP block. The release GGUFs were made
from the official FP8 checkpoint:

```sh
./download_model.sh glm53-q2  # about 90 GiB
./download_model.sh glm53-q4  # about 178 GiB
./download_model.sh glm53-fp8 # about 305 GiB; packaged weights only
```

The Q2 file uses imatrix-guided IQ2_XXS gate/up experts and Q2_K down experts.
It runs resident on a 128 GB M3 Max or M5 Max, and on one DGX Spark. The Q4
file is the higher-quality control. Run it across two 128 GB Macs, or use SSD
streaming on one Mac. The FP8 file preserves the released text weights without
requantization; DwarfStar cannot execute that artifact yet.

On one 128 GB Mac:

```sh
./ds4 -m gguf/GLM-5.3-Flash-Q2.gguf --ctx 32768
```

Q2 is close to the practical memory limit on a 128 GB Mac. Stop other
memory-heavy workloads before loading it resident.

The model's MTP block is already inside the same GGUF. `--mtp` enables it;
`--mtp-timing` also prints acceptance and timing counters. At non-zero
temperature the default mode directly keeps target-matching greedy drafts.
Add `--mtp-exact-sampling` when the output must preserve the ordinary target
sampling distribution.

```sh
./ds4 -m gguf/GLM-5.3-Flash-Q2.gguf --mtp --temp 0 --ctx 32768
./ds4-agent -m gguf/GLM-5.3-Flash-Q2.gguf --mtp --ctx 50000
```

For a small multi-user server on one M5 Max, four 4096-token sessions fit the
tested Q2 layout. Native decode batching is used below token 4096; longer
sessions automatically use the ordered fallback.

```sh
./ds4-server -m gguf/GLM-5.3-Flash-Q2.gguf \
  --ctx 4096 --batched-session 4
```

To run Q4 on one Mac without making it resident:

```sh
./ds4 -m gguf/GLM-5.3-Flash-Q4_K.gguf --ssd-streaming --ctx 4096
```

For resident Q2 or Q4 across two 128 GB Macs, use the 50/50 setup documented
under [Tensor Parallelism over RDMA](#tensor-parallelism-over-rdma). Put the
same GGUF at the same path on both machines, start the worker first, and select
`--transport rdma` on both sides. Q4 is the main reason to use this setup.

On one DGX Spark, use Q2 without CUDA tensor parallelism:

```sh
make cuda-spark
./ds4 --cuda -m gguf/GLM-5.3-Flash-Q2.gguf --ctx 16384
```

Q4 does not fit one Spark, and Spark-to-Spark RDMA tensor parallelism is not
implemented.

### Vision

GLM 5.3 Flash vision uses a separate 1.1 GB encoder. The text GGUF stays the
same, and vision is enabled only when the sidecar is passed explicitly:

```sh
./download_model.sh glm53-vision
./ds4 -m gguf/GLM-5.3-Flash-Q2.gguf \
  --vision gguf/GLM-5.3-Flash-Vision-Encoder.gguf
```

The published encoder SHA-256 is
`ae23e14c6979e889051b2e4a39351abcdafb161e18e606fae4d8c40095a4bf3a`.

In the interactive CLI, `/read photo.jpg` or `/read image.png` submits the
image as a user turn. `ds4-agent` exposes the same support as its `view_image`
tool when started with `--vision`. JPEG and PNG decoding is built in; no image
library is required.

`ds4-server` accepts ordered image blocks in OpenAI Chat, Responses, and
Anthropic requests. HTTP images must be inline: use a PNG/JPEG data URI for
OpenAI or base64 image source for Anthropic. File paths and remote URLs are
rejected. A request may contain up to 16 images and the HTTP body is limited to
64 MiB.

Vision runs on Metal, single-GPU CUDA, and ROCm. On a DGX Spark, use the CUDA
command above and add `--vision FILE`. On the 128 GB Strix Halo reference host,
Q2 needs the same SSD-streaming options as text inference:

```sh
./ds4 --rocm --ssd-streaming --ssd-streaming-cache-experts 32GB \
  -m gguf/GLM-5.3-Flash-Q2.gguf \
  --vision gguf/GLM-5.3-Flash-Vision-Encoder.gguf
```

In two-Mac tensor parallel mode, pass the same `--vision` file on both the
coordinator and worker; the coordinator encodes the image and sends the
projected visual tokens to the worker.

## DSpark Speculative Decoding

DSpark is an auxiliary draft model released by DeepSeek for DeepSeek V4 Flash.
It reads hidden states from the main model and proposes up to five future
tokens. DwarfStar checks those proposals with the main Flash model and commits
only the accepted prefix. The main model remains authoritative; a rejected or
low-confidence suffix falls back to ordinary target decoding.

The possible gain is faster generation: when several proposed tokens are
accepted, one target verification pass advances the stream by several tokens.
It does not accelerate prefill, and the draft and verification work is not
free. Predictable continuations, especially code, tend to benefit most;
low-yield prompts can be no faster or even slower. DSpark is therefore still
experimental and explicitly opt-in.

Accepted proposals keep the state produced by the batched target verifier
instead of running the same tokens through one-token decode again. Both paths
execute the same inference graph, but floating-point operations are grouped in
a different order. A long greedy DSpark run may therefore diverge from a run
without DSpark after an otherwise valid accepted block. This is not a reduced
precision or approximate-model mode; use ordinary decoding, `--quality`, or
`--dspark-strict` when byte-for-byte reproducibility with one-token decode is
required.

The DSpark checkpoint for Flash 0731 is packaged here as a separate support
GGUF of about 5.6 GiB. It is not a standalone model. Download it once:

```sh
./download_model.sh ds4f-dspark
```

The support file can be used with the 0731 Flash `ds4f-q2`, `ds4f-q2-q4`, and
`ds4f-q4` models listed above. It is checkpoint-specific
and must not be paired with an older Flash model. For now **DeepSeek V4 PRO**
is not supported. On Metal, CUDA, and ROCm, the main model may be resident or
use `--ssd-streaming`; the support model is kept separately mapped or
device-cached and adds its own weights and runtime state to the memory
requirement. DSpark replaces the legacy one-stage MTP support model for that
run rather than stacking with it.

Run it with the normal sampling defaults:

```sh
./ds4 -m ds4flash.gguf \
  --mtp-model gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  --dspark
```

`--mtp-model` supplies the support GGUF, while `--dspark` selects the DSpark runtime.
Measured on a MacBook Pro M5 Max with the Q2 Flash model fully resident
(README prose prompt, 128-token context): plain decode 46.5 t/s, DSpark
36-44 t/s depending on the confidence threshold (30-55% of draft tokens
accepted; every verified draft row re-streams its routed experts, about a
third of a token's cost). Expect gains only on highly predictable text.
The default confidence threshold is `0.6` on Metal and `0.7` on CUDA and ROCm.
It prunes suffixes that are unlikely to repay their verification cost.
`--dspark-confidence 0` forces fixed five-token blocks and is intended for
diagnostics.

At a non-zero temperature, ordinary `--dspark` uses opportunistic sampling.
Tokens evaluated normally are sampled with the requested temperature, top-p,
top-k, and min-p. DFlash then proposes a temperature-zero suffix. Every draft
token that matches the target's temperature-zero continuation is committed
directly, even though the requested temperature is non-zero. Sampling resumes
at the first mismatch. This is deliberately more deterministic than ordinary
temperature sampling. On an M5 Max it retained enough of the greedy DSpark gain
to improve a predictable code continuation by about 8% at temperature 1. A
single M3 Max run was slightly slower, the same test was nearly neutral on DGX
Spark, and it was slower on Strix Halo, where verification is more expensive.

Use `--mtp-exact-sampling` when the output must follow the ordinary target
distribution. With the DFlash support model, exact mode disables direct
temperature-zero matching: it accepts each greedy proposal with its target
probability and, on rejection, samples from the remaining target distribution.
It uses a stricter `0.8` confidence threshold by default. Add `--temp 0` for
fully greedy decoding. `--quality` and `--dspark-strict` keep target-only
decoding, which is useful for reproducibility checks.
The same DSpark flags work with `ds4-agent` and with non-batched
`ds4-server` requests. Session-batched serving currently uses ordinary target
decoding.

On a single accelerator, the main model can instead stream its routed experts
from SSD while the DSpark support model remains mapped or device-cached
separately. On Metal the support mapping is file-backed and pageable; CUDA and
ROCm prepare a separate device cache. Select the backend with `--metal`,
`--cuda`, or `--rocm`:

```sh
./ds4 -m ds4flash.gguf \
  --mtp-model gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  --dspark --metal --ssd-streaming \
  --ssd-streaming-cache-experts 16 --temp 0
```

Use `--cuda` in a CUDA build. On ROCm, use `--rocm` and a verification-safe
cache, for example `--ssd-streaming-cache-experts 32`.

For memory-constrained Metal systems, use a small graph workspace as well as a
small expert cache. A practical 16 GiB starting point is:

```sh
./ds4 -m ds4flash.gguf \
  --mtp-model gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  --dspark --metal --ssd-streaming \
  --ssd-streaming-cache-experts 16 \
  --ctx 4096 --prefill-chunk 128 --temp 0
```

When DSpark+SSD runs on a Mac with at most 24 GiB and neither
`--prefill-chunk` nor `DS4_METAL_PREFILL_CHUNK` is set, the runtime selects 128
automatically. Set `DS4_DSPARK_LOW_MEMORY_PREFILL_CHUNK=0` to retain the normal
workspace policy, or set it to another row count.

The Metal SSD verifier already supports the checkpoint's full five-draft
speculative block. With top-6 routing it needs at least 30 effective target
expert-cache slots; `--ssd-streaming-cache-experts 32` is the practical
five-draft setting. Smaller caches automatically limit the verifier to the
number of complete top-k rows that fit (a 16-expert cache normally selects two
rows). The Metal proposer follows that effective verifier/cache cap, avoiding
work on a suffix that cannot be consumed. Set
`DS4_METAL_DSPARK_PROPOSER_BLOCK_MAX=0` to restore the checkpoint's native
five-row proposer for an A/B control, or set a positive value to cap it
explicitly. Override the verifier policy independently with
`DS4_DSPARK_SSD_VERIFY_BLOCK_MAX=N`.

Metal can experimentally mirror the final target-hidden prefill row from the
HC weighted-sum kernel itself, avoiding a separate 16 KiB blit and
compute/blit encoder transition on each captured target layer. Enable it with
`DS4_METAL_ENABLE_DSPARK_CAPTURE_FUSED_LAST=1`; the historical
weighted-sum-plus-blit sequence remains the default because short M1 Pro SSD
A/B runs were bit-identical but did not show a repeatable throughput win.
`DS4_METAL_DISABLE_DSPARK_CAPTURE_FUSED_LAST=1` is the dominant kill switch.

An experimental single-device Metal verifier can use the current target
logits for the first draft and evaluate only the remaining `N-1` target rows.
Enable it with `DS4_METAL_DSPARK_ACCEPTANCE_ONLY_VERIFY=1`; it remains opt-in
because the smaller batch did not improve throughput on the measured M1 Pro
SSD path. Two-draft blocks retain the legacy verifier
because its one-row SSD routed-FFN path does not yet use the tiny-batch expert
table. A five-draft block starting exactly on a ratio-4 compressor boundary
also retains the legacy path so acceptance arithmetic does not switch to the
aligned compressor kernel. After verification rolls back,
the exact replay also skips the output head and logits readback for accepted
prefix tokens whose logits would be discarded; set
`DS4_METAL_DSPARK_HEADLESS_REPLAY=0` to restore the legacy replay. With
`DS4_DSPARK_STATS=1`, `metal_accept_only`, `metal_verify_rows_saved`, and
`metal_replay_headless` show how often these paths were exercised.

On very small unified-memory Macs, an additional diagnostic can keep only the
stage-0 `main_norm` and `main_proj` support tensors resident. Set
`DS4_METAL_DSPARK_PIN_MAIN_PROJ=1`; the 0731 support file locks about 51 MiB,
not the full 5.6 GiB GGUF. A failed lock is non-fatal and leaves the existing
pageable path active. Keep this opt-in until a same-machine A/B shows lower
`prop_setup`/generation time without reducing target-only throughput.

An experimental two-draft Metal SSD verifier can commit a full accept without
the normal rollback/replay pass:

```sh
DS4_METAL_DSPARK_EXACT2=1 DS4_DSPARK_STATS=1 \
./ds4 -m ds4flash.gguf \
  --mtp-model gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  --dspark --metal --ssd-streaming \
  --ssd-streaming-cache-experts 16 --ctx 4096 --prefill-chunk 128 \
  --temp 0
```

It uses the canonical one-row decode kernels in layer order, restores and
replays token zero on a partial accept, and defaults both proposal and verify
width to two. Keep it opt-in until a long same-machine run is byte-identical,
has `exact2_attempt>0` and `exact2_fallback=0`, and improves throughput. The
generic Metal verifier can already evaluate five drafts together with a
32-expert cache, but its batch state is not numerically interchangeable with
ordinary decode and therefore still requires rollback plus exact replay.

`DS4_METAL_DSPARK_EXACTN_UNION=1` enables a separate experimental Metal SSD
verifier for two through five draft tokens. It executes canonical one-row
target decode in layer order, loads the union of the rows' routed experts once
per layer, and commits its verifier state directly after a full accept. On a
partial match it restores the frontier once, skips the boundary oracle,
exact-two path, and legacy token-by-token verifier, then exactly replays only
the already verified prefix. With `DS4_DSPARK_STATS=1`,
`exactn_union_partial_replay` and `exactn_union_verify_skip` should advance
together; `exactn_union_partial_replay_ms` isolates the required commit replay.
Set `DS4_DSPARK_FIXTURE_REQUIRE_METAL_EXACTN_PARTIAL=1` to require at least one
such partial match, equal replay/skip counts, no exact-union error fallback,
and byte-identical fixture output. The model-backed
`test-metal-exactn-oracle` is byte-identical to sequential decode for N=2..5,
including every N=5 partial prefix, EOS in the first or a middle row, serialized
KV/compressor state, logits, and a four-token continuation. Five drafts plus
the target token already available at the start of the cycle cover the
six-token speculative-cycle limit. Exact-union remains opt-in: correctness
does not imply a throughput improvement on a particular memory configuration.

For an independent Q8 output-head A/B inside exact-union, set
`DS4_METAL_DSPARK_EXACTN_BATCH_HEAD=1`. HC collapse and normalization remain on
the canonical one-row kernels, while one bit-exact decode-row dispatch projects
all two through five verifier rows to vocabulary logits. Non-Q8 output weights
are ineligible and a dispatch failure falls back to the ordinary per-row heads.
`DS4_METAL_DISABLE_DSPARK_EXACTN_BATCH_HEAD=1` is the unconditional kill switch
and wins if both variables are set. The
`metal_exactn_batch_head_attempt`, `metal_exactn_batch_head_use`, and
`metal_exactn_batch_head_fallback` counters identify the selected path; set
`DS4_DSPARK_FIXTURE_REQUIRE_METAL_EXACTN_BATCH_HEAD=1` to require a nonzero,
fallback-free use with byte-identical output.

The generic model-backed exact-N oracle keeps this Q8-only experiment disabled,
so it remains valid for target models with another output quantization. To add
model-backed batch-head coverage, use an OutQ8 target explicitly:

```sh
DS4_TEST_METAL_EXACTN_BATCH_HEAD=1 \
DS4_TEST_MODEL=/path/to/target-OutQ8.gguf \
make test-metal-exactn-oracle
```

The Metal proposer also has an independent experiment for confidence/Markov
synchronization overhead. Set `DS4_METAL_DSPARK_DEVICE_PROPOSER=1` when the
final confidence projection and both Markov matrices are Q8_0. On an eligible
single-device, tier-zero run it keeps the previous token, confidence decisions,
and Markov argmax chain on Metal, reuses the first confidence already computed
by the proposer, and returns one result for the complete draft block. Unlike
the CUDA experiment, the Metal path is eligible with SSD streaming; tensor
placement and proposal-quality mode remain excluded. It stops at the first
rejected confidence row and preserves the smaller-token argmax tie break.
Unsupported layouts, an incomplete result, or a CPU sigmoid-policy mismatch
fall back to the existing per-row implementation.

`DS4_METAL_DSPARK_NO_DEVICE_PROPOSER=1` is the unconditional kill switch;
`DS4_DSPARK_NO_GPU_MARKOV=1` and `DS4_DSPARK_NO_MARKOV=1` also keep the path
disabled. This remains opt-in because Q8 confidence accumulation moves from the
host CPU to Metal and therefore needs a same-machine greedy oracle and A/B.
With `DS4_DSPARK_STATS=1`, require
`metal_device_proposer_attempt == metal_device_proposer_use > 0`,
`metal_device_proposer_fallback=0`, and
`metal_device_proposer_policy_mismatch=0`. The acceptance fixture enforces
those conditions and byte-identical output with
`DS4_DSPARK_FIXTURE_REQUIRE_METAL_DEVICE_PROPOSER=1`.

Exact-union normally waits for every layer's routed-tail command buffer before
releasing its private expert-address scope. For an isolated A/B,
`DS4_METAL_DSPARK_EXACT_ROWS_ASYNC_TAILS=1` commits that tail without the CPU
wait, retains all scope resources until command-buffer completion, and lets the
next layer's router boundary provide the required ordering. The switch has no
effect outside exact-union and is also opt-in; unset it for the synchronous
control. Validate serialized state and greedy output as well as verifier time,
because removing a host wait is not by itself evidence of an end-to-end gain.

The AProjQ4 Metal decode path has several exact dispatch fusions relevant to
this verifier:

- HC RMSNorm plus the narrow F16 HC mixer is the M1-M4 default, including SSD
  split phases such as `TO_ROUTER`. Use
  `DS4_METAL_DISABLE_PRE_M5_HC_NORM_MIX_FUSE=1` for the reference control;
  `DS4_METAL_ENABLE_HC_NORM_MIX_FUSE=1` is the explicit non-default gate on
  other Apple generations.
- The Q4 Q-A/KV projections can share a dispatch with eligible F16 compressor
  projection/store work. It remains opt-in in both exact-union and ordinary
  `FULL` decode via `DS4_METAL_ENABLE_Q4_QKV_COMPRESSOR_FUSE=1`; the first M1
  Pro SSD A/B reduced dispatch count but did not improve verifier time. In
  either scope,
  `DS4_METAL_DISABLE_Q4_QKV_COMPRESSOR_FUSE=1` selects the existing fallback.
- The eligible Q4 attention-output B projection can perform the following HC
  expansion in the same dispatch. Use
  `DS4_METAL_DISABLE_Q4_ATTN_OUT_HC_FUSE=1` as its isolated A/B control.
- For an AProjQ4 multi-row attention-output batch with either attention B in
  Q8 or Q4_K, the opt-in
  `DS4_METAL_ENABLE_Q4_ATTN_OUT_TINY_BATCH=1` evaluates two through five rows
  in two dispatches while retaining the canonical one-row reduction order for
  every row. This covers the generic suffix verifier; the exact-union tape is
  intentionally still row-by-row and does not select this helper. Other
  output formats, unsupported shapes, a disabled Q4 classic matvec, or a
  dispatch setup failure return to the existing row-wise path.
  `DS4_METAL_DISABLE_Q4_ATTN_OUT_TINY_BATCH=1` is the unconditional kill
  switch and wins when both variables are set. For a fail-closed model-backed
  generic-verifier test, `DS4_METAL_REQUIRE_Q4_ATTN_OUT_TINY_BATCH=1` implies
  the enable gate for N=2..5 and turns an ineligible shape, the kill switch, or
  a dispatch failure into a hard error instead of a silent row-wise fallback.

The AProjQ8 Q-A/KV plus compressor compound remains the M1-M5 default for
eligible resident `FULL` decode. For an SSD-streaming A/B, including the
exact-union `TO_ROUTER` collection prefix, set
`DS4_METAL_ENABLE_Q8_QKV_COMPRESSOR_FUSE=1`. Ratio-4 layers combine the Q8
Q-A/KV pair with both attention and indexer F16 compressor pairs; ratio-128
layers combine it with the attention pair. The kernel preserves the canonical
NSG=4 Q8 and NR0=2 F16 reduction trees. A diagnostic Q8 NSG override or the
experimental NR0=4 compressor schedule therefore selects the separate
dispatches. `DS4_METAL_REQUIRE_Q8_QKV_COMPRESSOR_FUSE=1` turns such a fallback
into a visible error for model-backed tests. The existing ratio-specific
pre-M5/M5 QKV compound disable variables remain authoritative. Keep the SSD
extension opt-in until warm and cold A/B runs show a gain: it removes a launch
per row and layer but reads the same model bytes, and a compound grid can
change the order in which distant GGUF pages are faulted.

Additional PR #755 ports keep their established kernels as shape/resource
fallbacks:

- On Apple M1 through M5, an eligible one-row HC producer combines the F16
  RMSNorm/mixer, HC split and Sinkhorn-weighted sum, and destination RMSNorm in
  one compound dispatch for both attention and FFN producers. The global
  rollback is `DS4_METAL_DISABLE_HC_PRODUCER_PRE_NORM_FUSE=1`; the narrower
  controls are `DS4_METAL_DISABLE_PRE_M5_HC_PRODUCER_PRE_NORM_FUSE=1` and
  `DS4_METAL_DISABLE_M5_HC_PRODUCER_PRE_NORM_FUSE=1`. The existing
  `DS4_METAL_DISABLE_PRE_M5_DECODE_PORTS=1` umbrella also disables it before
  M5. `DS4_METAL_ENABLE_HC_PRODUCER_PRE_NORM_FUSE=1` permits a focused trial
  on another eligible Metal device.
- For an eligible ratio-4 layer on Apple M1 through M5, the standalone F16 compressor path can
  project the attention and indexer KV/gate pairs and append both recurrent
  states in one quad dispatch. It is the default in ordinary `FULL` decode and
  the exact-union collection prefix when the larger Q4 compound dispatch did
  not already store those states. Use
  `DS4_METAL_DISABLE_COMPRESSOR_QUAD_STORE=1` for the reference path;
  `DS4_METAL_DISABLE_PRE_M5_COMPRESSOR_QUAD_STORE=1` is an additional
  compatibility rollback. `DS4_METAL_ENABLE_COMPRESSOR_QUAD_STORE=1` permits
  a focused trial on another Metal device and widens the phase scope for
  diagnostics.
- The exact ratio-4, one-compressed-row pool specialization is the M1-M5
  default for supported 128- and 512-element head shapes. Disable it globally
  with `DS4_METAL_DISABLE_COMPRESSOR_EXACT_POOL_RATIO4=1`, or use the
  pre-M5/M5 controls
  `DS4_METAL_DISABLE_PRE_M5_COMPRESSOR_EXACT_POOL_RATIO4=1` and
  `DS4_METAL_DISABLE_M5_COMPRESSOR_EXACT_POOL_RATIO4=1`. For a diagnostic run,
  `DS4_METAL_REQUIRE_COMPRESSOR_EXACT_POOL_RATIO4=1` turns an unavailable
  exact dispatch into a visible failure instead of silently selecting the
  legacy reduction sequence.

Metal FlashAttention pipeline selection also keeps a generation-aware
one-entry host memo for hot specializations. This changes pipeline lookup, not
kernel arithmetic. Disable the pad/block memo with
`DS4_METAL_DISABLE_PRE_M5_FLASH_ATTN_PAD_BLK_MEMO=1` and the batched/vector
memo with `DS4_METAL_DISABLE_PRE_M5_FLASH_ATTN_BATCHED_MEMO=1` when isolating
host-side dispatch overhead.

The former 512-column streaming Metal top-k specialization has been removed;
its ordering was not deterministic for every input. The regular deterministic
top-k implementation is now used instead and has no runtime re-enable switch.
Use a previous binary only as a performance control, and require identical
selected ids on tie-heavy inputs before comparing timing.

Apple M1 defaults to a specialized SSD-streaming decode path for the exact
IQ2_XXS/Q2_K routed-MoE shape with 256 experts, top-6 routing, and a
4096-to-2048 gate/up projection. It replaces the IQ2 address-table pair-SwiGLU
producer, including complementary resident/missing cache masks.
It preserves the canonical dot-product,
reduction, clamp, activation, and route-weight order but writes `mid` directly
instead of materializing the otherwise unused gate/up rows. Every other
device, shape, streaming mode, unsupported mask/accumulate mode, or unavailable
pipeline keeps the canonical producer. Set
`DS4_METAL_DISABLE_M1_IQ2_MID_ONLY=1` to restore the canonical producer.
For fail-closed model coverage, `DS4_METAL_REQUIRE_M1_IQ2_MID_ONLY=1` rejects
an ineligible supported address-table dispatch; the kill switch still takes
precedence. The former `DS4_METAL_ENABLE_M1_IQ2_MID_ONLY=1` opt-in is accepted
as a harmless compatibility setting because the path is now automatic.
`make test-metal-iq2-midonly` compares all 12,288 top-6
output words bitwise at full shape for both unmasked and complementary masked
address tables, verifies that the candidates leave gate/up sentinels untouched,
and checks output guards. The routed-MoE stage profiler reports
`iq2_stream_addr_mid_only_4096x2048` or
`iq2_stream_addr_mask_mid_only_4096x2048` when the model path is actually
covered.

These gates change dispatch and intermediate-memory traffic, not model
arithmetic. Compare byte-identical output, exact-union counters, stage timings,
and generation rate on the same machine; do not infer a speedup from a lower
dispatch count alone.

CPU greedy decoding and the verifier's excluding-argmax scan use an unrolled
eight-lane implementation by default, including scalar tail handling and
first-index tie semantics. Set `DS4_CPU_DISABLE_UNROLLED_ARGMAX=1` to restore
the scalar scan for an isolated A/B. `tests/test_sampling` compares both paths,
including cross-lane ties, excluded ids, and non-multiple-of-eight vocabulary
sizes.

Exact file views for the two token embedding rows and repeatedly used Q8
support tensors are automatic; the compatibility kill switches are
`DS4_METAL_DISABLE_TOKEN_EMBED_EXACT_VIEW=1` and
`DS4_METAL_DISABLE_SUPPORT_Q8_DECODE_EXACT_VIEWS=1`.

DSpark attempts a proposal on every eligible cycle. Proposal cadence is not
adaptively throttled, so the reported acceptance rate covers the full runtime
sample. Quality and strict DSpark modes remain target-only.

Tune the expert-cache count for the available accelerator memory. ROCm needs
enough slots for a whole verification block (30 for the 0731 model; use at
least 32), and currently supports the IQ2_XXS/Q2_K or all-Q2_K routed-expert
layouts. CUDA uses a transient selected-expert cache for each target block.
The DSpark support weights are included in the startup memory budget even when
the Metal file-backed mapping remains pageable. This combination is
single-device only; CPU, distributed or
multi-GPU placement, tensor parallelism, and legacy MTP support models remain
incompatible with DSpark plus SSD streaming.

Resident single-GPU CUDA skips verifier captures that rollback/replay cannot
consume, batches frontier snapshot/restore copies behind one device fence,
computes the output head only for the final replayed token, pads the five-row
Q8 proposer head to the tensor-core shape, and fuses proposer Q RMSNorm with
RoPE. CUDA and ROCm also avoid the Metal-only mid-token submission split: on
those backends the same flush is a device-wide synchronization and only drains
the launch pipeline. The two kernel-selection kill switches for before/after
measurements are
`DS4_CUDA_DSPARK_NO_PADDED_HEAD=1` and
`DS4_CUDA_DSPARK_NO_Q_NORM_ROPE_FUSION=1`.

CUDA fuses HC split, weighted sum, and RMSNorm across multiple batch rows,
including the DSpark proposer and verifier; use
`DS4_CUDA_DISABLE_HC_SPLIT_NORM_FUSED=1` for an A/B fallback to the separate
kernels. AProjQ4 CUDA decode can also share the activation quantization for
the Q-A/KV dense pair; `DS4_CUDA_DISABLE_Q4_DENSE_PAIR=1` selects the two
standalone projections. The canonical Q4 path submits to the decode stream so
these projections and the attention-output tail can participate in CUDA
decode graphs.

Single-token resident CUDA can also experiment with folding the canonical
Q8_1 activation emitted by the 4096-wide HC split plus RMSNorm stage into its
next MMVQ consumer. This remains opt-in pending GB10/DGX validation: set
`DS4_CUDA_ENABLE_Q8_FOLD=1`; `DS4_CUDA_NO_Q8_FOLD=1` is the dominant kill
switch. The sidecar is one-shot and keyed by model map, physical device,
stream, source pointer, and session epoch. Capture, scratch growth, a model-map
or device transition, and every lookup mismatch reject or invalidate it and
fall back to the established quantizer. For a diagnostic run, disable decode
graphs and add `DS4_CUDA_Q8_FOLD_ORACLE=1`; the oracle compares canonical Q8_1
bytes and the reached aligned-Q8 or IQ2 MoE consumer output, then always keeps
the freshly quantized reference. Require nonzero fold hits, `byte_calls`, and
`output_calls`, with zero mismatches and skips before considering promotion.
The experiment supports ds4's serialized, single-inference-host-thread CUDA
runtime only; embeddings that submit concurrently to the same CUDA stream
must leave it disabled.

On a single DGX Spark/GB10, the AProjQ4 path also mirrors the safe parts of
the aligned-Q8 decode work while retaining canonical Q4_K MMVQ/Q8_1
arithmetic:

- dense and paired Q4 projections reuse the persistent 1-MiB Q8_1 scratch;
  `DS4_CUDA_NO_Q4_DENSE_SCRATCH=1` restores pool allocation;
- attention-output A evaluates all output groups through one channel-grouped
  MMVQ dispatch per token, preserving the one-row reduction tree of every
  group. For DSpark verification widths 2--8,
  `DS4_CUDA_ENABLE_Q4_GROUPED_ATTN_A_BATCH=1` flattens `(token, group)` into
  MMVQ channels and replaces the per-token loop with one grouped MMVQ
  dispatch while keeping `ncols_dst=1`;
  `DS4_CUDA_NO_Q4_GROUPED_ATTN_A_BATCH=1` restores the per-token grouped loop
  and `DS4_CUDA_NO_Q4_GROUPED_ATTN_A=1` restores the per-group loop;
- for prefill widths above eight, the default eligible GB10 path quantizes the
  strided `[token][group][K]` input in one launch and writes each group directly
  into `[token][group][rank]`. It removes the eight F32 pack/unpack copies while
  keeping one established stream-K MMQ reduction per group. On the production
  `groups=8`, `K=4096`, `rank=1024` shape, a fixed-layout eight-warp Q8_1
  producer is also the default: each warp emits two canonical 128-value DS4
  records, reducing quantizer CTA count by four while preserving every output
  byte. `DS4_CUDA_NO_Q4_GROUPED_ATTN_A_Q81=1` restores only the generic
  strided Q8_1 producer, and
  `DS4_CUDA_REQUIRE_Q4_GROUPED_ATTN_A_Q81=1` fails closed if the specialized
  producer is not selected. Set
  `DS4_CUDA_NO_Q4_GROUPED_ATTN_A_PREFILL=1` for the dominant local rollback;
  exact `DS4_CUDA_ENABLE_Q4_GROUPED_ATTN_A_PREFILL=0` is also a compatibility
  opt-out. Add `DS4_CUDA_REQUIRE_Q4_GROUPED_ATTN_A_PREFILL=1` for fail-closed
  tests. The separate single-grid/grid.z submission remains opt-in. The
  unrelated 16-warp MMQ experiment also remains opt-in and is not used by
  this producer;
- attention-output B keeps its canonical MMVQ result and the ordinary HC
  epilogue inside the graph-compatible fused call.  The row-packed epilogue
  remains oracle-only until a GB10 device run proves it bit-exact;
- the exact Q-b shape `32768x1024` has an experimental persistent-CTA kernel
  behind `DS4_CUDA_ENABLE_Q4_K1024_PERSISTENT=1`, with
  `DS4_CUDA_NO_Q4_K1024_PERSISTENT=1` taking precedence. Tests can add
  `DS4_CUDA_REQUIRE_Q4_K1024_PERSISTENT=1` to fail instead of silently using
  canonical MMVQ when the persistent dispatch is unavailable; this admission
  now fails before quantization, output clearing, or any kernel enqueue. Set
  `DS4_CUDA_Q4_K1024_PERSISTENT_STATS=1` for host-dispatch candidate/use/
  fallback counters. For a bitwise model-backed check, run with
  `DS4_CUDA_DECODE_GRAPHS=0 DS4_CUDA_Q4_K1024_PERSISTENT_ORACLE=1`; the oracle
  forces the candidate, compares it with canonical MMVQ, and always retains
  canonical output.

`DS4_CUDA_NO_Q4_GB10_FAST=1` is the umbrella rollback for these new GB10
choices; it does not disable the older cross-CUDA Q-A/KV pair itself. For a
fail-closed grouped attention comparison, set
`DS4_CUDA_ENABLE_Q4_GROUPED_ATTN_A_BATCH=1`,
`DS4_CUDA_REQUIRE_Q4_GROUPED_ATTN_A_BATCH=1`, and
`DS4_CUDA_Q4_GROUPED_ATTN_A_ORACLE=1`. The oracle computes the established per-group
MMVQ reference (or the established per-token grouped loop for a multi-token
candidate), reports aggregate calls/mismatches/skips plus separate
batch_candidates/batch_calls/batch_mismatches/batch_skips, and retains
canonical output. A valid multi-token test has nonzero candidates/calls and
zero batch mismatches/skips. The
oracle disables decode-graph capture. For a multi-token candidate, if it
encounters another active capture or cannot allocate comparison scratch, it
directly enqueues the canonical reference instead of consuming an unchecked
candidate. The scratch, grouped, and persistent paths are also covered by
`make test-mmq-parity-cuda CUDA_ARCH=sm_121`.

CUDA Q4_K MMQ performs its non-finite output guard in the final write-back
(or after the final stream-K fixup) instead of launching a separate
full-output sanitizer. Finite results and the per-group reduction order are
unchanged; the resident CUDA prefill harness checks them bit-for-bit against
the former pack/MMQ/unpack path.

Two additional CUDA fusions remain experimental until a device oracle passes
on the target GPU. `DS4_CUDA_ENABLE_HC_NORM_MIX_FUSE=1` combines HC RMSNorm
with the narrow F16 mixer when the selected standalone kernels have the same
reduction order; with the normal one-token cuBLAS path, also set
`DS4_CUDA_NO_F16_CUBLAS_ONE=1` to exercise it. The controls are
`DS4_CUDA_DISABLE_HC_NORM_MIX_FUSE=1` and
`DS4_CUDA_DISABLE_Q4_ATTN_OUT_HC_FUSE=1`. The Q4 attention-output B plus HC
path is automatic when MMQ is disabled, where the existing one-dispatch Q8_K
implementation is bit-compatible with its fallback. With the normal
MMVQ/Q8_1 decode path, the GB10 graph-compatible call preserves both the
canonical MMVQ projection and the ordinary HC expansion. The specialized
row-packed epilogue is evaluated only by the oracle below and is never
consumed by normal decoding. The older, truly single-dispatch Q8_K experiment
is isolated behind
`DS4_CUDA_Q4_ATTN_OUT_HC_Q8K_EXPERIMENT=1` and may differ numerically from
MMVQ/Q8_1.

For a fail-closed hardware comparison, set
`DS4_CUDA_Q4_ATTN_OUT_HC_ORACLE=1`. It retains the canonical MMVQ/Q8_1 output,
compares both the row-packed epilogue and the Q8_K compound bit-for-bit, and
prints `epilogue_mismatches`, `q8k_mismatches`, and `skips` at exit. The
oracle avoids readback while CUDA graph capture is active; run a separate
non-captured diagnostic and require `calls>0`, `skips=0`, and
`epilogue_mismatches=0` before promoting the row-packed path back into normal
decoding. A zero-call summary is therefore an explicit failed coverage gate,
not a silent pass.

An experimental resident-CUDA path can run the existing aligned
IQ2_XXS/Q2_K vector MoE kernels for two-to-five-draft routed batches,
preserving the established fused-SoA path as an automatic fallback:

```sh
DS4_CUDA_DSPARK_TINY_ALIGNED_VEC=1 DS4_DSPARK_STATS=1 \
./ds4 --cuda -m ds4flash.gguf \
  --mtp-model gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  --dspark --temp 0 -p 'Write a Python quicksort function with comments.'
```

Keep the aligned tiny-batch path opt-in until the same-machine acceptance,
decode-consistency, and throughput comparisons pass on CUDA hardware.

An experimental exact two-token resident-CUDA verifier is available for a
DGX Spark A/B test. It uses the ordinary decode kernels, commits a two-token
full accept without rollback/replay, and replays only the first token on a
partial accept. By default the switch runs both the proposer and verifier at
width two, instead of evaluating the checkpoint's native five-row proposal
when only two drafts can be consumed:

```sh
DS4_CUDA_DSPARK_EXACT2=1 DS4_DSPARK_STATS=1 \
./ds4 --cuda -m ds4flash.gguf \
  --mtp-model gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  --dspark --temp 0 -p 'Write a Python quicksort function with comments.'
```

The support model uses non-causal attention across the proposal block, so a
two-row proposal is not guaranteed to be a prefix-identical version of its
native five-row proposal. The target verifier still protects the emitted
greedy continuation. For isolated A/B tests,
`DS4_CUDA_DSPARK_PROPOSER_BLOCK_MAX=0` preserves the native proposer and an
explicit value such as `2` caps it independently of exact-2.

CUDA also has an opt-in tiled online-softmax kernel for this non-causal support
attention. It shares each raw KV row across a group of attention heads and is
selected only for the DSpark raw-ring/head geometry it supports; every other
shape keeps the reference kernel. Enable it with
`DS4_CUDA_ENABLE_DSPARK_NONCAUSAL_ONLINE=1`. The emergency control
`DS4_CUDA_DISABLE_DSPARK_NONCAUSAL_ONLINE=1` wins when both variables are set.
The online reduction order can change draft floating-point results even though
the target verifier still protects greedy output. Use
`DS4_DSPARK_VERIFY_NONCAUSAL=1` to print the first three comparisons against a
host double-precision reference, and require the final target continuation to
remain byte-identical in the performance A/B.

Keep this path opt-in until the CUDA acceptance fixture is byte-identical and
the same-machine statistics show lower `propose`, `verify` plus `replay` time.
The stats line reports `prop_capped`, `prop_scheduled_rows`, `exact2_attempt`,
`exact2_full`, `exact2_partial`, and `exact2_fallback`; a valid run must
exercise exact-2 and leave its fallback counter at zero.

A separate resident exact-N CUDA experiment extends the same canonical
one-token tape to two through five draft rows. It leaves hidden rows and all
target weights on one GPU, submits the per-row ordinary decode kernels in one
stream, and reads back only the `N-1` acceptance ids plus final logits. A full
match therefore commits its already-exact KV/compressor state without replay;
a partial match restores the pre-cycle frontier once and replays the prefix
already proven by exact-N, without running the legacy verifier a second time.
Only a backend error retains the legacy verifier/replay fallback. Enable it
independently with:

```sh
DS4_CUDA_DSPARK_EXACTN=1 DS4_DSPARK_STATS=1 \
./ds4 --cuda -m ds4flash.gguf \
  --mtp-model gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  --dspark --temp 0 -p 'Write a Python quicksort function with comments.'
```

The default verifier cap under this gate is five (or the available prefill
workspace when smaller). The proposer uses at most one fewer workspace row,
because its support stage also carries the current target row; the existing
explicit proposer and verifier cap variables still take precedence.
`DS4_CUDA_DISABLE_DSPARK_EXACTN=1` is the
kill switch and restores the previous path without changing the enable
variable. Track `cuda_exactn_attempt`, `cuda_exactn_full`,
`cuda_exactn_fallback`, its partial/error split, and `cuda_exactn_rows`. Keep
this experiment disabled by default until a real CUDA oracle and a long greedy
A/B show byte-identical output, no fallback errors, and a throughput win.

The layer tape can separately reuse its position-independent CUDA decode
islands with `DS4_CUDA_DSPARK_EXACTN_GRAPHS=1`. Graph keys use the stable
device address (including each batch-row offset), not the short-lived tensor
view wrapper. Four cache entries per layer/island remain reserved for ordinary
decode and five more are isolated for exact-N rows, so a width-five verifier
cannot evict the normal decode keys. The first encounter warms lazy allocators,
the second captures/instantiates, and only later encounters are pure replay;
benchmark at least 128--256 generated tokens rather than judging a short
capture-heavy run. `DS4_CUDA_DISABLE_DSPARK_EXACTN_GRAPHS=1` is the dedicated
kill switch, while `DS4_CUDA_DECODE_GRAPHS=0` still disables all decode graphs.
The stats fields `cuda_exactn_graph_attempt`, `..._use`, `..._warm`,
`..._capture`, `..._replay`, `..._no_slot`, and `..._failure` expose warmup,
reuse, capacity misses, and retired captures. This first rollout is for
serialized, single-session DGX testing; the graph cache and cuBLAS capture
state remain process-global. The fixture can require a clean post-warmup
replay (including zero no-slot/failure events) with
`DS4_DSPARK_FIXTURE_REQUIRE_CUDA_EXACTN_GRAPHS=1`.

For a separate output-head A/B, set
`DS4_CUDA_DSPARK_EXACTN_BATCH_HEAD=1`. The experiment keeps HC collapse and
normalization on the canonical one-row kernels, then runs the Q8 vocabulary
projection for all exact-N rows through the bit-exact decode-row kernel. It is
automatically ineligible for non-Q8 output weights and falls back to the
ordinary per-row heads on a dispatch failure. The emergency kill switch is
`DS4_CUDA_DISABLE_DSPARK_EXACTN_BATCH_HEAD=1` and wins when both variables are
present.

The CUDA proposer tail has a second, independent experiment for the fixed
confidence/Markov synchronization overhead:

```sh
DS4_CUDA_DSPARK_DEVICE_PROPOSER=1
```

When the final confidence projection and both Markov matrices are Q8_0, this
keeps the previous token, confidence decisions, and all Markov argmax steps on
the decode stream and reads one 64-byte result for the whole draft block.  It
stops at the first rejected confidence row, preserves the Markov smaller-token
tie break, and rechecks the returned confidence prefix with the established
CPU sigmoid policy.  Unsupported layouts, an incomplete result, or a policy
mismatch fall back to the per-row implementation.  The unconditional kill
switch is `DS4_CUDA_DSPARK_NO_DEVICE_PROPOSER=1`; the older
`DS4_DSPARK_NO_GPU_MARKOV` switch also keeps this path disabled.
The initial gate is intentionally limited to resident, single-GPU, non-quality
CUDA and reuses the already-computed first confidence value.

This remains opt-in because the Q8 confidence accumulation moves from the host
CPU to CUDA and must pass the DGX proposal/acceptance oracle before promotion.
With `DS4_DSPARK_STATS=1`, require
`cuda_device_proposer_attempt == cuda_device_proposer_use > 0`,
`cuda_device_proposer_fallback=0`, and
`cuda_device_proposer_policy_mismatch=0`.  The acceptance fixture can enforce
those conditions with
`DS4_DSPARK_FIXTURE_REQUIRE_CUDA_DEVICE_PROPOSER=1`.

The stats line separates `cuda_exactn_ms` into setup, layer, head, and read
components, and reports restore, legacy-error-fallback verification, and
partial replay time independently. `cuda_exactn_partial_replay` and
`cuda_exactn_verify_skip` should advance together on valid partial matches;
the batch-head attempt/use/fallback counters make its dispatch unambiguous.

Set `DS4_DSPARK_FIXTURE_REQUIRE_CUDA_EXACTN=1` on the candidate acceptance
fixture to require at least one `cuda_exactn_attempt` and zero
`cuda_exactn_error_fallback`. The aggregate `cuda_exactn_fallback` is reported
but is not required to be zero: it also includes valid partial draft matches,
which deliberately restore the frontier and use exact replay.

The acceptance fixture can exercise the same SSD path on both the target-only
baseline and the DSpark run. It also requires real proposals and accepted
draft tokens, so an unavailable verifier cannot pass as a silent no-op:

```sh
DS4_DSPARK_FIXTURE_BACKEND=cuda \
DS4_DSPARK_FIXTURE_SSD_STREAMING=1 \
DS4_DSPARK_FIXTURE_SSD_STREAMING_CACHE_EXPERTS=32 \
make dspark-acceptance
```

For ROCm, use `make rocm-dspark-acceptance` with the same model, support, and
SSD fixture environment variables. The ROCm-specific target preserves the HIP
object set and linker; the generic target selects CUDA objects on non-Apple
hosts. `make rocm-dspark-verify-depth` provides the corresponding verifier
invariant test.

## Speed

The current q2 results use `ds4-bench` with the standard *Promessi sposi*
input, 2048-token context steps, and 128 greedy generation tokens at every
frontier. Each prefill number is for the next 2048-token chunk. The complete
sweeps are in [m5_max.csv](speed-bench/m5_max.csv) and
[gb10.csv](speed-bench/gb10.csv). The GB10 optimization methodology and
validation are documented in
[ds4_gb10_q2_cuda_port_results.md](speed-bench/ds4_gb10_q2_cuda_port_results.md).

| Machine | Backend | Context | Prefill | Generation |
| --- | --- | ---: | ---: | ---: |
| MacBook Pro M5 Max, 128 GB | Metal | 2048 | 790.18 t/s | 39.35 t/s |
| MacBook Pro M5 Max, 128 GB | Metal | 16384 | 572.53 t/s | 36.14 t/s |
| MacBook Pro M5 Max, 128 GB | Metal | 32768 | 557.04 t/s | 34.36 t/s |
| MacBook Pro M5 Max, 128 GB | Metal | 65536 | 398.50 t/s | 27.64 t/s |
| DGX Spark GB10, 128 GB | CUDA | 2048 | 832.86 t/s | 20.58 t/s |
| DGX Spark GB10, 128 GB | CUDA | 16384 | 883.81 t/s | 16.80 t/s |
| DGX Spark GB10, 128 GB | CUDA | 32768 | 865.40 t/s | 15.99 t/s |
| DGX Spark GB10, 128 GB | CUDA | 65536 | 833.44 t/s | 15.27 t/s |

Older measurements for machines and model variants not rerun in this pass are
kept for reference. They used the earlier CLI prompt procedure and are not
directly comparable with the table above.

| Machine | Quant | Prompt | Prefill | Generation |
| --- | ---: | ---: | ---: | ---: |
| MacBook Pro M3 Max, 128 GB | q2 | short | 58.52 t/s | 26.68 t/s |
| MacBook Pro M3 Max, 128 GB | q2 | 11709 tokens | 250.11 t/s | 21.47 t/s |
| Mac Studio M3 Ultra, 512 GB | q2 | short | 84.43 t/s | 36.86 t/s |
| Mac Studio M3 Ultra, 512 GB | q2 | 11709 tokens | 468.03 t/s | 27.39 t/s |
| Mac Studio M3 Ultra, 512 GB | q4 | short | 78.95 t/s | 35.50 t/s |
| Mac Studio M3 Ultra, 512 GB | q4 | 12018 tokens | 448.82 t/s | 26.62 t/s |
| Mac Studio M3 Ultra, 512 GB | PRO q2 | 32768 tokens | 138.82 t/s | 9.56 t/s |

![M5 Max t/s](speed-bench/m5_max_ts.svg)
![PRO model M3 Ultra t/s](speed-bench/pro_model_m3_ultra_ts.svg)

## Running models larger than RAM

The normal Metal path tries to make the model resident in GPU-addressable
memory. This is the fastest path and should remain your default when the model
fits. DwarfStar also has an **SSD streaming** capacity mode on Metal and for
GLM 5.2 on ROCm. In this mode the non-routed model weights stay resident, while
routed MoE experts are kept in an in-memory cache and loaded from the GGUF file
on cache misses.

Streaming is not as fast as fitting the full model in RAM. It still needs memory
for non-routed weights, KV cache, graph scratch, activations, and the routed
expert cache. It is useful because routed experts dominate model size and modern
Mac SSDs are fast enough to make cache misses tolerable. Long prefills can still
be fast; generation is more sensitive to cache misses because every new token
routes through experts again.

Start with the automatic cache budget:

```sh
./ds4 -m ./ds4flash.gguf --ssd-streaming
```

To reserve more memory for context, set the routed expert cache explicitly:

```sh
./ds4 -m ./ds4flash.gguf --ssd-streaming --ssd-streaming-cache-experts 32GB
```

The `32GB` value is a routed-expert memory budget, not a generic byte cache.
DwarfStar first reserves headroom for the two full routed layers used by
overlapped streaming prefill, then converts the remaining bytes to the number of
dynamic cached experts that fit for the current GGUF. This is a target, not a
promise to allocate that much: DwarfStar reduces it when the model map, graph,
context, and backend working-set limit leave less room. A plain number such as
`--ssd-streaming-cache-experts 4000` requests 4000 dynamic expert slots without
the two-layer reserve, but it can be reduced by the same final memory check.
Non-routed weights, KV cache, graph scratch, and activations need additional
memory.

Metal Q8_0 dense decode matvecs and paired projections also use a single
reduction barrier on eligible non-TP/non-quality 4-simdgroup paths,
in both resident and SSD modes. Only the output-owning simdgroup performs the
final reduction, with the same sum order and launch geometry. Set
`DS4_METAL_DISABLE_Q8_MV_SINGLE_BARRIER=1` to restore their previous kernels.
This does not change prefill or the separate shared-expert SwiGLU policy.
`make bench-metal-q8-mv` runs bitwise checks followed by a resident synthetic
kernel-only A/B; full-model and SSD throughput still need separate tests.
Eight-simdgroup paths retain the legacy kernel after mixed local A/B results;
see [measurement notes](speed-bench/metal_q8_mv_single_barrier.md).

Metal SSD decode enables two kernel specializations by default on eligible
non-quality, non-TP paths. Q4_K selected experts (six cache slots or
address tables) load activations once for gate/up and compute SwiGLU from
registers; the separate Q8_0 shared-expert gate/up kernel uses one threadgroup
barrier instead of two, with the same reduction order and launch geometry.
The Q8 shared-expert change can apply to both AProjQ4 and AProjQ8 models; the
Q4 change requires Q4_K routed-expert weights. Neither changes SSD reads,
cache policy, prefill, CUDA, or ROCm.

For tester A/B runs, set either rollback independently (any defined value,
including `0`, disables it):

```sh
DS4_METAL_DISABLE_SSD_Q4_PAIR_SHARED_X=1 \
DS4_METAL_DISABLE_SSD_Q8_SINGLE_BARRIER=1 \
./ds4 -m ./ds4flash.gguf --ssd-streaming
```

Unset both variables for the candidate run. Keep model, cache size, prompt,
context, and SSD/cache warmup identical between arms; compare each rollback
separately to attribute changes. `make test-metal-ssd-decode-kernels` runs a
GGUF-free bitwise GPU oracle with guarded buffers. It does not measure SSD or
end-to-end performance; throughput gains still require model-backed A/B tests.

CUDA's raw Q8_0 activation quantizers also use a default-on warp reduction
outside quality mode, for both resident and SSD decode/prefill. The ordinary
and group-slice producers drop six block barriers; the dual Q8_K/Q8_0 producer
drops seven in its Q8_0 phase without changing the Q8_K signed-max reduction.
Launch geometry, stream, rounding, and padding are preserved. This does not
change the separate MMQ Q8_1 path. Use
`DS4_CUDA_DISABLE_Q8_QUANT_WARP_REDUCE=1` in a fresh process to restore the
previous kernels. Test correctness with `make test-cuda-q8-quantize` on a CUDA
host, then measure end-to-end throughput with identical model/cache settings.

Metal SSD+DSpark also has an experimental, support-aware pre-cap for A/B tests.
Set `DS4_METAL_DSPARK_SAFE_EXPERT_COUNT=1` to convert a numeric count to bytes
and cap it, when measurable, after accounting for the target's non-routed
weights, the context/KV estimate, and a 2 GiB active reserve for the mmap-backed
support model. It does not yet price the complete batch-prefill workspace or a
separate routed-prefill transient reserve. Startup reports requested/effective
slots and the support reserve. If this policy cannot measure safe room, it
retains the explicit count with a warning; the normal final memory check remains
authoritative. The experiment does not affect `NGB` budgets or CUDA/ROCm.
Prefer an `NGB` budget for normal use. The automatic cache budget takes
80% of the backend's recommended working set, subtracts non-routed weights, then
applies the same routed-prefill headroom before sizing the dynamic cache. Leave
the hot expert preload enabled for normal use; use `--ssd-streaming-cold` and
`--ssd-streaming-preload-experts N` only for measurements.

For Metal IQ2_XXS/Q2_K models, eligible SSD prefill chunks automatically use
grouped address matmuls when the dynamic cache can retain the complete expert
domain. This includes normal 128-token chunks; the automatic range is 32–760
tokens on the 256-expert Flash model and requires, for example,
`--ssd-streaming-cache-experts 256`. Once that material condition is met,
selection is fail-closed by default; no environment prefix is required. Set
`DS4_METAL_ENABLE_IQ2_XXS_SSD_PREFILL_MM=0` or
`DS4_METAL_DISABLE_IQ2_XXS_SSD_PREFILL_MM=1` for the legacy sparse-matvec
rollback. `DS4_METAL_REQUIRE_IQ2_XXS_SSD_PREFILL_MM=0` keeps automatic
selection but permits a fallback, while explicit `REQUIRE=1` also rejects an
insufficient cache. Combining `REQUIRE=1` with `DISABLE=1` fails on an eligible
grouped-MM candidate, while short tail chunks retain their normal fallback. The
IQ2 live cache index remains automatic for its production shape, selected-load
early commit remains off unless explicitly enabled, and grouped-MM statistics
plus streaming timing summaries remain opt-in diagnostics. The grouped prefill
loader also skips `F_RDADVISE` for chunks of at least 32 tokens because it
immediately reads the same expert ranges with parallel `pread`; short chunks
retain the hint. Set
`DS4_METAL_ENABLE_STREAMING_PREFILL_EXPERT_READAHEAD=1` to restore the old
hint-plus-read sequence for cold-storage A/B tests.

### Practical SSD streaming examples

On 64GB MacBooks, start with the 2-bit Flash GGUF and a moderate expert cache:

```sh
./download_model.sh ds4f-q2

./ds4 \
  -m ./ds4flash.gguf \
  --ssd-streaming \
  --ssd-streaming-cache-experts 32GB \
  --ctx 32768 \
  --nothink
```

On 128GB MacBooks, PRO q2 streaming is experimental but usable for inspection
and occasional work when you accept slow generation. Start with `--nothink`:

```sh
./download_model.sh pro-q2-imatrix

./ds4 \
  -m gguf/DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-imatrix-0813.gguf \
  --ssd-streaming \
  --ctx 32768 \
  --nothink
```

On an M5 Max with 128GB of RAM, a short PRO q2 streaming decode benchmark found
the automatic budget best: it selected about `59GB` of routed expert cache.
Manual `64GB` to `75GB` caches were close on that machine. Prefer the automatic
budget; if setting the cache manually on this class of machine, start around
`48GB` to `64GB`, then increase only while the machine remains responsive and
the startup log shows the requested dynamic cache. Once the machine is stable,
re-enable thinking with a conservative generation limit:

```sh
./ds4 \
  -m gguf/DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-imatrix-0813.gguf \
  --ssd-streaming \
  --ctx 32768 \
  --think \
  --tokens 1500
```

GLM 5.2 uses the same option. Its streaming path keeps the largest full-layer
prefix that fits resident, then uses the remaining budget for a dynamic expert
cache. Start with the automatic budget:

```sh
./ds4 \
  -m gguf/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf \
  --ssd-streaming \
  --ctx 32768
```

The startup report shows the effective cache and whether GLM decode uses one
global model map or the lower-memory per-layer fallback. Start conservative,
then increase the cache if the machine has headroom.

On a 128GB Strix Halo, use the routed Q2_K model and a 4096-token context as the
starting point. The automatic cache budget leaves room for the GLM graph and KV
state:

```sh
./download_model.sh glm-antirez-q2
make strix-halo
./ds4 --rocm -m gguf/GLM-5.2-UD-Q2_K_RoutedQ2K.gguf \
  --ssd-streaming --ctx 4096
```

## Distributed inference with pipeline parallelism

Pipeline parallelism lets DwarfStar **run a model that is too large for one machine** by
splitting transformer layers across multiple machines. The main example is the
full 4-bit Flash quant across two 128 GB MacBooks: each process maps only its
own layer slice, activations are sent over TCP, and the coordinator keeps normal
CLI/API behavior.

Pipeline parallelism can also **speed up prefill** by
using multiple GPUs at the same time to process different micro-batches at
different layers, like in an assembly line. Only prefill can be accelerated this
way. Generation is purely autoregressive: each token must finish across the
route before the next token can start. The model work is the same as a single
process, plus coordination latency, so distributed generation is slower.

To build an initial mental model, here are the high level concepts:

1. You put the GGUF on every machine, but each one loads just a subset. `--layers` controls which tensors are mapped, so a worker with `--layers 20:output` does not load the earlier layers.
2. Layer ranges are inclusive: `10:20` means layers 10, 11, ..., 20. `N:output` means layer `N` through the final layer plus the output head.
3. You assign one of the machines the role of `coordinator`, the others the roles of `workers`. Workers will connect to the coordinator and will tell they are there and which layers they are able to process.
4. Each worker keeps its slice of the KV cache.
5. Communication is worker-to-worker, there is no need to use the coordinator as relay, so if your coordinator is `A`, and you make a request, activations will flow in `A -> B -> C -> back to A`.

The resident ROCm MXFP4 routed-expert path supports the same pipeline mode. A
tested two-host Strix Halo split uses `--layers 0:21` on the coordinator and
`--layers 22:output` on the worker. This is a capacity configuration for a
model that does not fit on one 128 GB system; it does not add ROCm SSD
streaming support for Flash.

### How it works and how to configure it

The prefill path is pipelined (this is why it can go faster than in a single machine).
For large prompts the coordinator can run its
slice on chunk N+1 while the worker is running its slice on chunk N. The
distributed rows below were measured with two M5 Max 128 GB MacBooks connected
by Thunderbolt 5, using the Q4 Flash GGUF and the default 4096-token
distributed prefill chunk. The single-process column is a reference run with
the Q2 GGUF on a single machine, so it actually is a bit faster since
the routed MoEs are smaller.

| Prompt | Single-process reference | Two MacBooks | Speedup |
| ---: | ---: | ---: | ---: |
| 9421 tokens | 421.70 t/s | 582.22 t/s | 1.38x |
| 28684 tokens | 405.30 t/s | 674.16 t/s | 1.66x |
| 63819 tokens | 353.62 t/s | 654.79 t/s | 1.85x |

Generation is different. **It is strictly autoregressive**: token N+1 cannot start
until token N has produced logits and sampling has selected the next token. That
means distributed generation cannot use the long prefill pipeline. It pays at
least one cross-machine activation hop per generated token, so generation is
slower than a single local process. On the same two-Mac Thunderbolt setup, a
12k-context control run with the 91 GB Flash quant went from 30.59 t/s
single-process to 24.67 t/s distributed, a 19.4% loss. Distributed inference is
therefore mainly for fitting larger models and speeding up long prefills, not
for making decode faster.

### Full DeepSeek V4 PRO Q4 on two Mac Studios

The full-size PRO Q4 GGUF can be run across two 512 GB Mac Studio M3 Ultra
machines by giving the coordinator layers `0:30` and the worker
`31:output`. Use the split GGUF files so each side maps only the tensors it
needs:

```sh
# Coordinator machine.
./download_model.sh pro-q4-layers00-30

# Worker machine.
./download_model.sh pro-q4-layers31-output
```

The two files are:

```text
gguf/DeepSeek-V4-Pro-Q4K-Layers00-30.gguf
gguf/DeepSeek-V4-Pro-Q4K-Layers-31-output.gguf
```

This is a capacity use case: each process maps only its own half of the model,
while the worker owns the output head and returns logits.

The current PRO Q4 Metal path uses queue-resident exact expert tables for the
large routed experts. This avoids the broad multi-GiB routed-tensor bindings
that made early distributed PRO Q4 attempts either run very slowly or hit Metal
memory accounting limits. In a short greedy smoke test over the direct
`192.168.0.182` / `192.168.0.183` link, the model generated coherent text and
measured 11.47 t/s generation after startup. Per-token telemetry was balanced:
local layers were around 39-43 ms, remote layers around 44-49 ms, for total
token times around 84-92 ms. Expect a slow startup while each side maps and
makes its half of the model resident. Long-context PRO Q4 prefill and decode
performance still needs separate benchmarking.

The measurements above use a Thunderbolt 5 cable. The implementation is plain
TCP and also works over slower links, including WiFi, but fast Ethernet or
Thunderbolt networking is strongly recommended. Slow links mostly hurt
generation latency and short prefills; large prefills can still benefit when
the layer split is balanced. In the normal performance path, the last worker
owns the output head and returns logits directly.

Minimal two-host configuration:

```sh
# Machine A: coordinator, owns tokenization, sampling, the prompt, and layers 0..30.
./ds4 \
  -m gguf/DeepSeek-V4-Pro-Q4K-Layers00-30.gguf \
  --role coordinator \
  --layers 0:30 \
  --listen 169.254.43.68 1234

# Machine B: worker, connects to A and owns layers 31..output.
./ds4 \
  -m gguf/DeepSeek-V4-Pro-Q4K-Layers-31-output.gguf \
  --role worker \
  --layers 31:output \
  --coordinator 169.254.43.68 1234
```

Normally the final worker should own the output head too, for example
`--layers 20:output`. This avoids returning a full final hidden-state batch
after prefill and lets the final worker produce the logits directly. On very
slow or metered links, `--layers 20:42` is also supported: the coordinator will
load the output head and compute logits locally, trading extra coordinator work
for smaller per-token replies.

### Network Link Comparison

The table below shows the same two M5 Max hosts, the same 91 GB Flash quant,
coordinator `--layers 0:19`, worker `--layers 20:output`, an 8192-token prompt
from `speed-bench/promessi_sposi.txt`, and 128 generated tokens. WiFi and
Internet numbers vary with local conditions, but the shape is the important
part: high latency hurts generation directly, while lower bandwidth also pulls
down long-prefill speed.

| Link | Addresses | Ping avg | Prefill | Generation |
| --- | --- | ---: | ---: | ---: |
| Thunderbolt 5 | `169.254.43.68` -> `169.254.12.245` | 0.45 ms | 582.99 t/s | 25.09 t/s |
| WiFi | `192.168.1.57` -> `192.168.1.95` | 77.20 ms | 250.70 t/s | 10.70 t/s |
| Internet / VPN | `10.77.0.4` -> `10.77.0.3` | 152.10 ms | 114.88 t/s | 3.63 t/s |

The Internet/VPN case is not meant to be a good interactive experience. It is
still useful for collective testing: multiple people can temporarily combine
machines to run a larger model that would not fit on any single host, accepting
slow decode in exchange for being able to inspect the model at all.

Use the coordinator exactly like normal `./ds4`: interactive chat, `/read`,
and ordinary generation go through the same high-level session API. The same
distributed options are also wired into `ds4-agent`, `ds4-eval`, and
`ds4-bench`. For benchmarks, workers should already be running; `ds4-bench`
waits until a complete route is available.

Useful tuning and diagnostics:

```sh
./ds4-bench \
  -m gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 32768 \
  --ctx-max 65536 \
  --step-incr 32768 \
  --gen-tokens 0 \
  --role coordinator \
  --layers 0:19 \
  --listen 169.254.43.68 1234 \
  --debug
```

`--debug` on the coordinator prints route formation and per-hop telemetry:
layer range, token span, local evaluation time, downstream wait time, socket
send time, and input/output byte counts. This is the current profiling tool for
deciding whether a split is balanced. `--dist-prefill-window N` controls how
many prefill chunks may be in flight end-to-end; the default is conservative
and bounded. `--dist-prefill-chunk N` exists for experiments, but the default
4096-token chunk is the canonical setting and should be used unless you are
explicitly validating a different chunk size.

By default DwarfStar sends hidden-state activations as 32-bit floats. To reduce
traffic, pass `--dist-activation-bits 16` or `--dist-activation-bits 8` on the
coordinator. This changes only the transport format between machines, not the
model weights or KV cache. 16-bit transport halves activation traffic and is the
first option to try on Ethernet or WiFi. 8-bit transport is more aggressive and
should be treated as an approximate/experimental mode unless you have validated
the output for your use case. However experimentally reduction activation
size didn't provide a significant improvement, so this option may be removed
in the future.

**If a worker disconnects, the coordinator removes that worker from the active
route**. The request already in flight can fail, and later calls report an
incomplete route until a compatible worker reconnects and sends a new
registration. For live sessions, the coordinator keeps the token history and can
rebuild worker KV state by replaying the prefix when the route is available
again. Workers also validate a rolling 64-bit token-prefix hash on every work
item, so a restarted worker at position 0 cannot silently accept work for
position N; it reports the mismatch and the coordinator replays the current
transcript. Ctrl+C in the CLI and agent is cooperative: DwarfStar waits for the
current distributed token or prefill chunk to drain before returning control,
which avoids coordinator-caused KV splits. Saved agent/server sessions use the
same KV file format as single-machine sessions: during save the coordinator
fetches worker-owned layer tensors and serializes one normal payload; during
load it splits that payload over the currently registered route.

### Distributed protocol overview

At the protocol level there are two kinds of connections. Workers keep a
control TCP connection open to the coordinator and send a `HELLO` with their
model ID, model family, quant profile, layer slice, context capacity, and data
port. The coordinator uses these registrations to build a route that covers all
layers. Work then moves over low-latency TCP data connections: the coordinator
computes the first slice, sends a `WORK` frame with session ID, token positions,
rolling token-prefix hashes before and after the span, route information, and
hidden-state payload, and each worker computes its slice. Middle workers can
forward directly to the next worker. The final worker returns logits to the
coordinator, or ACKs for non-final prefill chunks so the prefill pipeline can
stay full. `RESULT` frames echo the request ID and the post-span hash. A worker
status error is handled differently from a socket failure: KV/hash mismatch can
be recovered by replaying the token history on the same route, while transport
failure drops the route and waits for a replacement worker. For persistent KV,
the coordinator opens worker data connections and sends snapshot save/load
messages for each worker-owned layer range; the disk payload remains a single
agent/server cache file. The protocol has no
encryption or authentication, and is not release-stable yet; coordinator and
workers should be built from the same commit and used on trusted machines and
trusted networks.

## Tensor Parallelism over RDMA

Tensor parallelism runs a single decode across two Macs connected with a
Thunderbolt 5 cable, splitting the heavy per-layer work between the two
GPUs and exchanging 16-24KB partial sums at synchronization gates inside the
graph (RDMA over Thunderbolt when available, a dedicated TCP socket
otherwise). Unlike the pipelined distributed mode above, both
machines work on the *same token at the same time*, so it reduces
per-token latency instead of just fitting a bigger model.

Each machine keeps one contiguous half of the routed experts resident. Dense,
attention, shared-expert, embedding, and output weights remain replicated.
This lets a model whose routed experts do not fit on one machine run fully
resident across the pair; routed kernels never touch the peer's expert half.

### Running GLM 5.2 or GLM 5.3 across two 128 GB MacBooks

One-time setup per boot, on **both** machines:

```sh
# Let the GPU wire ~117 GB (default cap is ~75% of RAM; the resident
# expert shard needs ~97.5 GiB plus KV/scratch).
sudo sysctl iogpu.wired_limit_mb=120000

# RDMA over Thunderbolt needs an IPv4 address directly on the cabled
# member interface (the bridge IP does not count). Use the interface
# that is 'active' in ifconfig, e.g. en1 on one side and en6 on the
# other. Skip this if you are fine with the TCP fallback.
sudo ifconfig en1 inet 10.99.0.2/30 alias     # machine A
sudo ifconfig en6 inet 10.99.0.1/30 alias     # machine B
```

Check the verbs device before loading the model:

```sh
rdma_ctl status
ibv_devinfo -v
```

The device must be active and expose the IPv4-mapped GID for the address above,
for example `::ffff:10.99.0.2`. A working IP ping does not prove that RDMA is
active.

Both machines need the same tree, commit, and GGUF path. Tensor parallelism is
always a 50/50 split with one worker, so do not pass `--layers`. Start the worker
first; it retries while the coordinator loads. The worker must dial the address
on the Thunderbolt member interface, not the bridge address:

```sh
MODEL=gguf/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
# For GLM 5.3 Q4, use MODEL=gguf/GLM-5.3-Flash-Q4_K.gguf instead.

# Machine B: worker.
./ds4 -m "$MODEL" --tensor-parallel --role worker \
  --coordinator 10.99.0.2 9911 --transport rdma

# Machine A: coordinator.
./ds4 -m "$MODEL" --tensor-parallel --role coordinator \
  --listen 10.99.0.2 9911 --transport rdma -c 8192 \
  -p "Tell me something about the sea."
```

The active verbs device and IPv4-mapped GID are selected automatically. If that
is ambiguous, add `--rdma-device rdma_en6 --rdma-gid-index 1` on the worker and
the matching `rdma_en1` flags on the coordinator. Use `--transport tcp` on both
sides to force TCP. Run workers with `ds4`; the coordinator may be `ds4`,
`ds4-agent`, or `ds4-server`.

Startup takes about 9 seconds per machine: each rank pre-faults its
~100 GiB shard from SSD and pins it through a Metal residency set.
DeepSeek V4 Flash works the same way with its own GGUF on both machines.
DeepSeek gate vectors are 16 KB and ride as one RDMA message. GLM 5.2's
6144-wide 24 KB vectors are split into two ordered RDMA messages. GLM 5.3 uses
its own KDA/DSA gate schedule, exchanged and checked during TP startup.

Measured on two M5 Max 128 GB MacBooks (GLM 5.2, IQ2_XXS, 188 GiB):

| | two Macs, tensor parallel | one Mac, SSD streaming |
|---|---|---|
| decode | ~16.8 t/s (15.4 at 4k context) | ~4.8 t/s |
| prefill (4096 tokens) | ~94 t/s | ~3-5 t/s |
| residency | fully memory-resident | streams experts from SSD |

Notes: the coordinator mirrors every prompt sync and eval to the worker, so
both KV caches stay in lockstep; prompt processing splits both the
routed-expert GEMMs (by expert ownership) and the attention heads (a
contiguous half per machine) with one bulk partial-sum exchange per
layer per stage (`--tensor-parallel-token-prefill` selects a slower
token-by-token prefill that exactly matches the single-machine arithmetic).
The split graph is deterministic, but its changed floating-point reduction
order is not generally byte-identical to single-machine execution.

### Running DeepSeek V4 Flash across two 128 GB MacBooks

The same mode runs DeepSeek V4 Flash (145 GB in the MXFP4 recipe, which does
not fit one 128 GB Mac without SSD streaming). Each Mac keeps its half of the
routed experts resident (~77 GiB) plus the replicated dense weights; the
per-layer exchanges are 16 KB and ride as single RDMA messages.

One-time setup per boot, on **both** machines (as for GLM above):

```sh
sudo sysctl iogpu.wired_limit_mb=120000
sudo ifconfig en1 inet 10.99.0.2/30 alias     # machine A (coordinator), its Thunderbolt member interface
sudo ifconfig en6 inet 10.99.0.1/30 alias     # machine B (worker)
rdma_ctl status                               # "enabled" on both
```

Both machines need the same tree, the same commit, and the model at the same
relative path. Start the worker in a terminal that stays open (an interactive
`ssh -tt` session is fine; a detached `nohup` worker cannot reach the
coordinator on macOS), then the coordinator; the worker retries until the
coordinator listens. Do not probe the coordinator's port with `nc` or `curl`
while it waits: it accepts the probe as the worker and fails.

```sh
MODEL=gguf/DeepSeek-V4-Flash-Vision-Exp-MXFP4Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out.gguf
VISION=gguf/DeepSeek-V4-Flash-Vision-Encoder.gguf   # omit --vision for the text-only model

# Machine B: worker.
./ds4 -m "$MODEL" --vision "$VISION" --tensor-parallel --role worker \
  --coordinator 10.99.0.2 9911 --transport rdma --rdma-device rdma_en6 --rdma-gid-index 1

# Machine A: coordinator, interactive chat (use /read photo.png to send an image).
./ds4 -m "$MODEL" --vision "$VISION" --tensor-parallel --role coordinator \
  --listen 10.99.0.2 9911 --transport rdma --rdma-device rdma_en1 --rdma-gid-index 1 -c 8192
```

The text-only `DeepSeek-V4-Flash-0731-MXFP4Experts-...-chat-v2-mxfp4.gguf`
file has the same layers and tensor types and runs identically. Other
quantization recipes (IQ2, Q4_K experts) run too, but the fastest paths gate
on MXFP4 experts with Q8 attention and shared-expert tensors.

Measured on two M5 Max 128 GB MacBooks over Thunderbolt 5 (MXFP4 recipe,
README prompt):

| | two Macs, tensor parallel | one Mac, SSD streaming |
|---|---|---|
| decode, 128-token context | ~47-48 t/s | ~11 t/s |
| decode, 2048-token context | ~48 t/s | |
| prefill | ~250 t/s at 128 tokens, ~850 t/s at 2048 | ~10 t/s |
| startup | ~10 s per Mac (pins its shard) | |
| DSpark (`--dspark`) | slower than plain decode here (30-40 t/s: each draft row re-streams its experts) | |

Things that cost speed on the coordinator: a remote screen-sharing session
or an animated wallpaper (the display compositor takes GPU time from the
decode, up to 10%), and anything else using the GPU. Benchmark with the
screen idle. The first run after swapping roles between the two Macs can fail
at the first prefill round with `timeout waiting for bulk RDMA round`; start
both again.

The fast paths are on by default; environment variables exist to opt out
for diagnosis (set them on both machines): `DS4_TP_DISABLE_POLL_GATES=1`
(shared-event gates instead of the poll gates), `DS4_TP_DISABLE_GATE_PREFETCH=1`,
`DS4_TP_STATIC_SHARED_SPLIT=1` (fixed shared-expert halves instead of the
per-token GPU-decided split), `DS4_TP_DISABLE_FLAG_FOLD=1`,
`DS4_METAL_DISABLE_KV_NORM_DEFER=1`, `DS4_METAL_DISABLE_M5_ROUTER_PROJECT_SELECT_FUSE=1`,
`DS4_TP_DISABLE_RDMA_WARMUP=1`, `DS4_METAL_DISABLE_MXFP4_MM_ID_MPP=1` (simdgroup
instead of TensorOps routed-expert prefill GEMMs), `DS4_METAL_DISABLE_QUEUE_KEEPALIVE=1`
(no idle keepalive on the Metal command queue: the first command buffer after
~3 s of GPU idleness then starts 600-800 ms late). `DS4_TP_GATE_PROFILE=1` prints per-gate wait
statistics on each rank at exit; `DS4_METAL_ENCODER_TIMELINE=<file>` writes
a per-kernel GPU timeline (`misc/tp_tools/tl_analyze.py` reads it).

## Tensor Parallelism across CUDA GPUs

On a single CUDA server, `--cuda-tensor-parallel` splits DeepSeek V4 Flash
tensor and routed-expert work across an even number of GPUs. This is separate
from the Mac-to-Mac mode above: it does not use `--role`, RDMA, or the
distributed layer pipeline. GPU placement and memory budgets are selected with
the normal `--gpu-devices` and `--gpu-vram` options.

The device order is significant. With `N` devices, the first `N/2` logical
tiers are contiguous layer-pipeline homes and the second `N/2` tiers are their
tensor-parallel partners. Specify all homes first and then all partners, with
the closest P2P pair at matching positions. For example, the tested L40S host
uses physical pairs `(0,1)`, `(2,3)`, `(4,5)`, and `(6,7)`, expressed as
`0,2,4,6,1,3,5,7`. Each pair stores a 50/50 split of the routed experts, and
the vocabulary head is row-sharded across the participating output tiers.
Those large tensors are not duplicated. Dense attention, router, and shared
expert weights are replicated within each pair.

For maximum throughput on eight 48 GB L40S cards, use the imatrix Q4 model.
Its routed `Q4_K` layout has the native grouped multi-session kernels; the Q2
model is the lower-memory choice (including tested four-card runs), but its
unsupported grouped routed shapes use the exact fallback and have lower
aggregate serving throughput. Download and build the L40S target with:

```sh
./download_model.sh ds4f-q4
make cuda CUDA_ARCH=sm_89
```

This is the interactive-agent setup used on the eight-L40S server:

```sh
MODEL=gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix-0731.gguf

./ds4-agent --cuda --cuda-tensor-parallel \
  --gpu-vram auto \
  --gpu-devices 0,2,4,6,1,3,5,7 \
  --model "$MODEL" \
  --ctx 100000
```

For serving, keep multiple KV sessions resident so decode rows can be grouped
across requests. The tested host is configured for up to 16 resident sessions:

```sh
./ds4-server --cuda --cuda-tensor-parallel \
  --gpu-vram auto \
  --gpu-devices 0,2,4,6,1,3,5,7 \
  --model "$MODEL" \
  --ctx 100000 \
  --batched-session 16 \
  --host 0.0.0.0
```

The equivalent local launchers are `./run-nvidia-tp-agent.sh` and
`./run-nvidia-tp-server.sh`. The server launcher also enables the on-disk KV
cache and defaults to the native 0731 MXFP4 GGUF. Set `DS4_MODEL` to use the Q4
file above instead. Reduce the session count or context size if the requested
resident KV caches do not fit after model loading. CUDA TP, half-resident expert
ownership, output sharding, pipelined prefill, and compatible grouped decode are
selected by `--cuda-tensor-parallel`; no `DS4_CUDA_*` environment tuning is required.
Without an explicit `--prefill-chunk`, this mode uses 2048-token chunks so the
tested 16-session, 100k-context layout retains enough VRAM for resident KV
caches. An explicit `--prefill-chunk` remains an override for other topologies.

Any even card count that can hold the selected model and graph scratch is a
valid topology. On this class of 48 GB card, the useful measured endpoints are
Q2 on four cards (two pipeline stages) and Q4 on eight cards (four stages).
For a four-card PIX-paired subset such as physical GPUs `0,1,4,5`, the ordered
list is `0,4,1,5`. Two cards do not have enough memory for these Flash models.

This mode currently requires DeepSeek V4 Flash and an even multi-GPU
placement. GLM 5.2 instead uses normal layer placement across the selected
CUDA devices. DGX Spark is a single-GPU target and must not be started with
`--cuda-tensor-parallel`.

## Reducing heat, power usage and fan noise

Long local inference runs can keep the GPU busy for extended periods. If you
care more about heat, fan noise, battery life on MacBooks, or reducing thermal
stress on the hardware than about maximum throughput, use `--power N`.

`--power 100` is the default and means full speed. Lower values ask DwarfStar to target
that percentage of GPU usage: `--power 70` targets about 70%, `--power 50`
targets about half usage, and so forth. DwarfStar does this by measuring GPU work time
and inserting small sleeps between work units: during prefill it sleeps between
layers, and during generation it sleeps between decoded tokens. This reduces
sustained load without changing model output.

The option is available on the CLI, server, agent, eval, and benchmark tools
for DeepSeek models. GLM 5.2 currently accepts only `--power 100`. For example:

```sh
./ds4 --power 50
./ds4-agent --power 70
./ds4-server --power 40 --ctx 100000
```

## Native agent

DwarfStar features a native coding agent that works in a different way
than most other systems: the inference is controlled from within the agent
itself, without socket/API boundaries, so the session is represented
by the on-disk KV cache itself. Moreover the tools and the system prompt
are all designed vertically for DeepSeek v4 Flash and PRO. This provides a
few advantages:

* Low latency experience, bounded mainly by the prefill speed limits. Displaying of generated text, tool calling, start of a new session are always instantaneous.
* Live progress bar during prefill time.
* No DSML tool calling conversion, the tools are handled natively in the LLM format.
* KV cache mismatch are impossible by construction, the current state is always the truth.
* Everything is tuned for this model.
* Ability to switch saved sessions with `/list` and `/switch`; full KV sessions resume without a prefill stage.

Agent sessions are stored in `~/.ds4/kvcache`. Use `/save` to persist the
current session, `/list` to show saved sessions sorted by recent update time,
and `/switch <sha>` to resume one of them. The session ID is stable across
future saves and is derived from the first user prompt and creation time.
`/del <sha>` removes a saved session. `/strip <sha>` keeps the rendered
conversation text and title but removes the heavy KV payload; switching to a
stripped session rebuilds the KV cache by prefilling the saved text.

Use `--chdir /path/to/ds4` when launching `ds4-agent` from another directory,
so relative runtime files such as `metal/*.metal` resolve from the project tree.

However while the system already works, there is a lot of work to do
in order to make it ready for prime time. When finally the agent will reach
the wanted shape, we will *likely* split the server and the client creating a stateful
session-based protocol that can recreate all that in a client-server way.

## Benchmarking

`ds4-bench` measures instantaneous prefill and generation throughput at context
frontiers instead of reporting one whole-run average. It loads the model once,
walks a fixed token sequence to frontiers such as 2048, 4096, 6144, and uses
incremental prefill so each row measures only the newly-added token interval.
After each frontier it saves the live KV state to memory, generates a fixed
greedy non-EOS probe, restores the memory snapshot, and continues prefill.

```sh
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

The example file is a cleaned public-domain Project Gutenberg text of
Alessandro Manzoni's *I Promessi Sposi* (ebook #45334), with the Gutenberg
header and footer removed: <https://www.gutenberg.org/ebooks/45334>.

Use `--step-incr N` for different linear spacing, or `--step-mul F` for
exponential sweeps. Output is CSV with one row per frontier: latest prefill
interval tokens/sec, generation tokens/sec at that frontier, and
`kvcache_bytes`.

Sessions prefill long prompts in 4096-token chunks by default. Use
`--prefill-chunk 2048`, for example, to match the strict official-vector
checkpoint path. Changing the chunk changes the KV checkpoint/logit path, so
compare it as an explicit run configuration.
Chunked Metal prefill reuses the same range-capable layer-major graph for each
chunk, preserving absolute compressor/indexer boundaries while avoiding the old
per-layer chunk dispatch path.

## Capability Evaluation

`ds4-eval` is a small real-model integration benchmark. It is not a leaderboard
runner and should not be reported as an official GPQA, SuperGPQA, AIME, or
security benchmark score: the questions are an embedded 92-item subset chosen
to make local regression testing useful and visually inspectable. The program
loads the real GGUF, renders DeepSeek chat prompts, streams sampled tokens in a split-screen TUI, grades
the final answer, and prints a per-question report with prompt tokens,
generated tokens, pass/fail state, the model answer, and the correct answer.

```sh
./ds4-eval -m ds4flash.gguf --trace /tmp/ds4-eval.txt
```

The default run uses `--tokens 16000`, thinking mode enabled, and a soft/hard
`</think>` budget cutoff so the model has room to produce a visible answer.
`ds4-eval` sizes the context internally from the largest selected prompt plus
the generation budget, and refuses runs that would need more than 1M context
tokens. Press `p` to pause, `q` to exit and print the report, Up/Down to
inspect or select another question, and Enter to run the selected question next.
`--plain` disables the TUI.

Use `--regrade-trace /path/to/trace.txt` to replay the current answer
extractor and scorer against a prior `--trace` file without loading the model
or regenerating tokens. This is useful when auditing evaluator changes: it
shows which cases changed, the old picked answer, the new picked answer, and a
pass/fail summary.

For inference changes that can affect generation drift, keep this deterministic
q1..q4 token-count gate in the test plan:

```sh
./ds4-eval \
  -m ds4flash.gguf \
  --plain \
  --questions 4 \
  --tokens 2048 \
  --temp 0 \
  --seed 1
```

The generated-token counts must stay aligned with the baseline:

| Question | Expected state | Expected generated tokens | Expected given/correct |
|---:|---|---:|---|
| 1 | `PASSED` | 2048 | `B` / `B` |
| 2 | `PASSED` | 438 | `C` / `C` |
| 3 | `PASSED` | 666 | `70` / `70` |
| 4 | `FAILED` | 2048 | `A` / `C` |

The first 75 embedded questions are interleaved as 25 GPQA Diamond, 25 audited
SuperGPQA, and 25 AIME 2025 problems. The final 17 are an audited COMPSEC
subset of reduced single-function C/C++ vulnerability-localization questions.
The model is asked for the single best source line, or the smallest exact line
set only when the bug cannot be localized to one line; the scorer accepts small
audited ranges only when adjacent lines are equivalent locations for the same
bug. The order is
intentionally progressive: early questions are useful smoke tests, while later
questions are hard enough that a strong reasoning model should still miss some
of them. The SuperGPQA slice is curated rather than blind: upstream rows with
wrong keys, missing figures, or underspecified prompts are replaced with cleaner
rows.

The set should be treated as a hard capability regression suite rather than
a pass/fail unit test.

- **GPQA Diamond** contributes graduate-level science questions with
  multiple-choice answers. DeepSeek's model card reports strong results
  on full GPQA Diamond in thinking mode, but individual items still require
  careful physics, chemistry, or biology reasoning and are easy to lose with a
  small prompt/rendering or sampling regression.
- **SuperGPQA** contributes broad specialist knowledge and domain-transfer
  questions. The model-card SuperGPQA number is much lower than GPQA Diamond,
  so these items are expected to be uneven: some look mundane, others require
  niche professional knowledge or exact interpretation of a translated-style
  exam question.
- **AIME 2025** contributes exact-answer contest math. These are often the most
  unforgiving items in the set: no multiple-choice prior, no partial credit, and
  a single arithmetic or algebraic slip changes the grade.
- **COMPSEC** contributes single-function C/C++ security reasoning items
  reduced from public CVE writeups. These are not exploit prompts: the task is
  to identify the best source line where the defensive code flaw is introduced,
  or return `0` for a safe function.

In practice this means `ds4-eval` should not be expected to produce a perfect
92/92 run. It is meant to answer a more useful engineering question: after a
kernel, quantization, prompt-rendering, KV-cache, or tool-streaming change, does
DeepSeek V4 Flash still solve a representative mix of hard science, broad
knowledge, exact math, and security-code problems while using the same inference
path users run?

## CLI

One-shot prompt:

```sh
./ds4 -p "Explain Redis streams in one paragraph."
```

No `-p` starts the interactive prompt:

```sh
./ds4
ds4>
```

The interactive CLI is a real multi-turn chat. It keeps the rendered chat
transcript and the live graph KV checkpoint, so each turn extends the previous
conversation. Useful commands are `/help`, `/think`, `/think-max`, `/nothink`,
`/ctx N`, `/read FILE`, and `/quit`. Ctrl+C interrupts the current generation
and returns to `ds4>`.

The CLI defaults to thinking mode. Use `/nothink` or `--nothink` for direct
answers. Models with a built-in draft block use `--mtp`; models with a separate
support GGUF use `--mtp-model MTP.gguf`. `--mtp-draft 2` sets the maximum draft
depth. At non-zero temperature, add `--mtp-exact-sampling` when the output must
preserve the ordinary target sampling distribution.

## Server

Start a local OpenAI/Anthropic-compatible server:

```sh
./ds4-server --ctx 100000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

Use `--chdir /path/to/ds4` when launching `ds4-server` from another directory,
so relative runtime files such as `metal/*.metal` resolve from the project tree.

By default the server keeps one mutable backend/KV checkpoint in memory, so
stateless clients that resend a longer version of the same prompt can reuse the
shared prefix instead of pre-filling from token zero.

`--batched-session N` preallocates `N` independent resident KV sessions. Ready
decode steps are evaluated together, while long prefills alternate in bounded
chunks so one request does not block every decoder. Requests beyond `N` wait
for a resident slot. If disk KV caching is enabled, an idle slot is persisted
before reuse and can be restored when that conversation returns; an active
request is never evicted. Choose `N` and `--ctx` so all resident KV allocations
fit in GPU memory. Without this option, inference retains the original
single-session behavior.

While generation is active, prefill yields every 128 tokens by default.
`--mixed-prefill-quantum N` changes that interval for testing; larger values
reduce scheduling handoffs but can make active decoders wait longer.

Native decode batching runs the same model graph, but grouping rows can change
floating-point reduction order slightly. When a native kernel is unavailable,
DwarfStar runs the rows in a fixed order and returns the same full logits as
separate session evaluations. The current backend behavior is:

| Backend and model | Session execution |
| --- | --- |
| Metal, resident DeepSeek Flash | Native shared-expert and QKV batching from two rows upward when supported; ordered fallback otherwise. |
| Metal, GLM 5.2 | Ordered exact fallback. |
| Metal, GLM 5.3 | Native decode batching below token 4096; ordered exact fallback at longer contexts. |
| CUDA, DeepSeek Flash on a supported multi-GPU TP/EP layout | Native decode and mixed prefill/decode, with exact fallbacks for unsupported kernel shapes. |
| CUDA single GPU, including DGX Spark | Ordered exact fallback. |

`N` resident sessions allocate `N` KV states, so a context size that fits once
may not fit eight times. Native batching can improve aggregate throughput; an
ordered fallback provides concurrency and fairness, but not the same speedup.
MTP speculative decoding is disabled while native session batching is active.

Supported endpoints:

- `GET /v1/models`
- `GET /v1/models/deepseek-v4-flash`
- `GET /v1/models/deepseek-v4-pro`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/completions`
- `POST /v1/messages`

The Flash and PRO model endpoints are compatibility aliases. They both report
the model currently loaded from the GGUF passed with `-m`; the endpoint name does
not select a different model.

`/v1/chat/completions` accepts the usual OpenAI-style `messages`,
`max_tokens`/`max_completion_tokens`, `temperature`, `top_p`, `top_k`, `min_p`,
`seed`, `stream`, `stream_options.include_usage`, `tools`, and `tool_choice`.
Tool schemas are rendered into DeepSeek's DSML tool format, and generated DSML
tool calls are mapped back to OpenAI tool calls.

`/v1/responses` accepts OpenAI Responses-style `input`, `instructions`,
`tools`, `tool_choice`, `max_output_tokens`, `temperature`, `top_p`, `stream`,
and `reasoning`. It is the preferred endpoint for Codex CLI. The server keeps
Responses continuations bound to live state when possible, and can fall back to
the same DSML rendering and KV prefix reuse used by chat completions.

`/v1/messages` is the Anthropic-compatible endpoint used by Claude Code style
clients. It accepts `system`, `messages`, `tools`, `tool_choice`, `max_tokens`,
`temperature`, `top_p`, `top_k`, `stream`, `stop_sequences`, and thinking
controls. Tool uses are returned as Anthropic `tool_use` blocks.

Default sampled API generation uses `temperature=1`, `top_p=1`, and
`min_p=0.05`, so the default filter is relative probability rather than
nucleus mass. In thinking mode DwarfStar applies those fixed sampling defaults
to any knob the request omits, matching DeepSeek's fixed-thinking API behavior,
but sampling parameters set explicitly in the request always win: a
`temperature=0` request is greedy through the whole reasoning phase, so
benchmark harnesses get deterministic thinking-mode output.

The chat, Responses, and Anthropic endpoints support SSE streaming. In thinking
mode, reasoning is streamed in the native API shape instead of being mixed into
final text. OpenAI chat streaming
also streams tool calls as soon as the DSML invocation is recognized: the tool
header is sent first, then parameter bytes are forwarded as
`tool_calls[].function.arguments` deltas while generation continues. The
Anthropic endpoint streams thinking and text live, then emits structured
`tool_use` blocks when the generated tool block is complete.
The Responses endpoint streams the Responses event lifecycle expected by Codex,
including `response.output_text.delta`, function-call argument events, and
terminal `response.completed` / `response.incomplete` / `response.failed`
events.

For browser JavaScript clients served from another origin, start the server with
`--cors` to emit `Access-Control-Allow-*` headers. This only changes HTTP
headers; it does not expose the server on the LAN. Use `--host 0.0.0.0`
explicitly when remote machines should be able to connect.

### Tool call handling and canonicalization

DeepSeek V4 emits tool calls as [DSML text](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/encoding/README.md). Agent clients do not send that
same text back on the next request: they send normalized OpenAI/Anthropic JSON
tool-call objects. **If the server re-rendered those objects slightly
differently, the rendered byte prefix would no longer match the live KV
checkpoint** and the next turn would have to be rebuilt.

The first line of defense is exact replay. Every tool call gets an unguessable
API tool ID, and the server remembers `tool id -> exact sampled DSML block` in
a bounded in-memory map backed by radix trees. When the client later sends that
tool ID back, the prompt renderer uses the exact DSML bytes the model sampled,
not a freshly formatted approximation. This map can also be saved inside KV
cache files, so exact replay survives server restarts for cached histories.

**Canonicalization is only the backup path**. If the exact DSML block is missing,
or exact replay is disabled with `--disable-exact-dsml-tool-replay`, the server
renders a deterministic DSML form from the JSON tool object. After a tool-call
turn, it compares the live sampled token stream with the prompt that the next
client request will render. If needed, it rewrites the live checkpoint, or
falls back to an older disk KV snapshot and replays only the suffix. This keeps
the model continuation aligned with the stateless API transcript.

During generation, the server also treats DSML syntax differently from payload.
When the model is emitting stable protocol structure such as DSML tags,
parameter headers, JSON punctuation, or closing markers, sampling is forced to
`temperature=0` so the tool call stays parseable. This greedy mode does **not**
apply to argument payloads: `string=true` parameter bodies and JSON string
values, including file contents and edit text, use the request's normal sampling
settings. That separation is important: deterministic decoding is helpful for
syntax, but can create repeated text when applied to long code or file bodies.

Minimal OpenAI example:

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"deepseek-v4-flash",
    "messages":[{"role":"user","content":"List three Redis design principles."}],
    "stream":true
  }'
```

### Agent Client Usage

`ds4-server` can be used by local coding agents that speak OpenAI-compatible
chat completions. Start the server first, and set the client context limit no
higher than the `--ctx` value you started the server with:

```sh
./ds4-server --ctx 100000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

You can use larger context and larger cache if you wish. Full context of
1M tokens is going to use more or less 26GB of memory (compressed indexer
alone will be like 22GB), so configure a context which makes sense in
your system. With 128GB of RAM you would run the 2-bit quants, which are
already 81GB, 26GB are going to be likely too much, so a context window
of 100~300k tokens is wiser. However users reported being able to run 2bit
quants with 250k ctx window in a Macs with just 96GB of system memory: make sure
to kill processes that use too much memory, if you plan doing so ;)

The `384000` output limit below avoids token caps since the model is able
to generate very long replies otherwise (up to 384k tokens). The server
still stops when the configured context window is full.

For **opencode**, add a provider and agent entry to
`~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ds4": {
      "name": "ds4.c (local)",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "dsv4-local"
      },
      "models": {
        "deepseek-v4-flash": {
          "name": "DeepSeek V4 Flash (ds4.c local)",
          "limit": {
            "context": 100000,
            "output": 384000
          }
        }
      }
    }
  },
  "agent": {
    "ds4": {
      "description": "DeepSeek V4 Flash served by local ds4-server",
      "model": "ds4/deepseek-v4-flash",
      "temperature": 0
    }
  }
}
```

For **Pi**, add a provider to `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "ds4": {
      "name": "ds4.c local",
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "dsv4-local",
      "compat": {
        "supportsStore": false,
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": true,
        "supportsUsageInStreaming": true,
        "maxTokensField": "max_tokens",
        "supportsStrictMode": false,
        "thinkingFormat": "deepseek",
        "requiresReasoningContentOnAssistantMessages": true
      },
      "models": [
        {
          "id": "deepseek-v4-flash",
          "name": "DeepSeek V4 Flash (ds4.c local)",
          "reasoning": true,
          "thinkingLevelMap": {
            "off": null,
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "xhigh"
          },
          "input": ["text"],
          "contextWindow": 100000,
          "maxTokens": 384000,
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          }
        }
      ]
    }
  }
}
```

Optionally make it the default Pi model in `~/.pi/agent/settings.json`:

```json
{
  "defaultProvider": "ds4",
  "defaultModel": "deepseek-v4-flash"
}
```

For **Codex CLI**, use the Responses wire API:

```toml
[model_providers.ds4]
name = "DS4"
base_url = "http://127.0.0.1:8000/v1"
wire_api = "responses"
stream_idle_timeout_ms = 1000000
```

Then run:

```sh
codex --model deepseek-v4-flash -c model_provider=ds4
```

For **Claude Code**, use the Anthropic-compatible endpoint. A wrapper like this
matches the local `~/bin/claude-ds4` setup:

```sh
#!/bin/sh
unset ANTHROPIC_API_KEY

export ANTHROPIC_BASE_URL="http://127.0.0.1:8000"
export ANTHROPIC_AUTH_TOKEN="dsv4-local"
export ANTHROPIC_MODEL="deepseek-v4-flash"

export ANTHROPIC_CUSTOM_MODEL_OPTION="deepseek-v4-flash"
export ANTHROPIC_CUSTOM_MODEL_OPTION_NAME="DeepSeek V4 Flash local ds4"
export ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION="ds4.c local GGUF"

export ANTHROPIC_DEFAULT_SONNET_MODEL="deepseek-v4-flash"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="deepseek-v4-flash"
export ANTHROPIC_DEFAULT_OPUS_MODEL="deepseek-v4-flash"
export CLAUDE_CODE_SUBAGENT_MODEL="deepseek-v4-flash"

export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1
export CLAUDE_STREAM_IDLE_TIMEOUT_MS=600000

exec "$HOME/.local/bin/claude" "$@"
```

Claude Code may send a large initial prompt, often around 25k tokens, before it
starts doing useful work. Keep `--kv-disk-dir` enabled: after the first expensive
prefill, the disk KV cache lets later continuations or restarted sessions reuse
the saved prefix instead of processing the whole prompt again.

## Thinking Modes

DeepSeek V4 Flash has distinct non-thinking, thinking, and Think Max modes.
The server defaults to thinking mode. `reasoning_effort=max` requests Think
Max, but it is only applied when the context size is large enough for the model
card recommendation; smaller contexts fall back to normal thinking. OpenAI
`reasoning_effort=xhigh` still maps to normal thinking, not Think Max.

For direct replies, use `thinking: {"type":"disabled"}`, `think:false`, or a
non-thinking model alias such as `deepseek-chat`.

## Disk KV Cache

Chat/completion APIs are stateless: agent clients usually resend the whole
conversation every request. `ds4-server` first tries the cheap exact token-prefix
check, then falls back to comparing rendered prompt bytes with decoded
checkpoint bytes. The live in-memory checkpoint covers the current session; the
disk KV cache makes useful prefixes survive session switches and server
restarts.

For RAM reasons there is currently only one live KV cache in memory. When a new
unrelated session replaces it, the old checkpoint can only be resumed without
re-processing if it was written to the disk KV cache. In other words, memory
cache handles the active session; disk cache is the resume mechanism for
different sessions.

Enable it with:

```sh
./ds4-server --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

The cache key is the SHA1 of the rendered byte prefix, and files are named
`<sha1>.kv`. The DS4 payload still stores the exact token IDs and graph state
for that prefix. This matters for continued chats: the model may have generated
one token whose decoded text is later sent back by a client as two canonical
prompt tokens. A rendered byte-prefix hit can still reuse the checkpoint and
tokenize only the new suffix.
The file is intentionally written with ordinary `read`/`write` I/O, not
`mmap`, so restoring cache entries does not add more VM mappings to a process
that already maps the model.

Tool calls also keep a bounded exact-DSML replay map keyed by unguessable tool
IDs, so client JSON history can be rendered back to the exact sampled text. The
RAM map keeps up to 100000 IDs by default; tune it with `--tool-memory-max-ids`.
Use `--disable-exact-dsml-tool-replay` to disable this and fall back to
canonical JSON-to-DSML rendering.

On disk, a cache file is:

```text
KVC fixed header, 48 bytes
u32 rendered_text_bytes
rendered_text_bytes of UTF-8-ish token text
DS4 session payload, payload_bytes from the KVC header
optional tool-id map section
```

The fixed header is little-endian:

```text
0   u8[3]  magic = "KVC"
3   u8     version = 1
4   u8     routed expert quant bits, currently 2 or 4
5   u8     save reason: 0 unknown, 1 cold, 2 continued, 3 evict, 4 shutdown
6   u8     extension flags, bit 0 = appended tool-id map
7   u8     reserved
8   u32    cached token count
12  u32    hit count
16  u32    context size the snapshot was written for
20  u8[4]  reserved
24  u64    creation Unix time
32  u64    last-used Unix time
40  u64    DS4 session payload byte count
```

The rendered text is the tokenizer-decoded text for the cached token prefix.
It is both the human-inspectable prefix and the lookup identity: its SHA1 is
the filename, and a file is reusable only when those bytes are a prefix of the
incoming rendered prompt. After load, the exact checkpoint tokens from the DS4
payload remain authoritative, and only the incoming text suffix after the cached
bytes is tokenized.

The optional tool-id map is present only when header extension bit 0 is set.
Appended sections use fixed bit order, so future extension bits can add fields
without ambiguity. The map stores unguessable API tool call IDs back to the
exact DSML block the model sampled. Only mappings whose DSML block is present
in the rendered cached text are stored. This lets restarted servers render
later client history byte-for-byte like the original model output, even if the
client reorders JSON arguments.

The current tool-id map section is:

```text
0   u8[3]  magic = "KTM"
3   u8     version = 1
4   u32    entry count

For each entry:
0   u32    tool id byte length
4   u32    sampled DSML byte length
8   bytes  tool id
... bytes  exact sampled DSML block
```

The section is auxiliary replay memory, not model state. A cache hit restores
the session payload first, then loads the map if present. Before rendering a
request, the server can also scan cache files for the tool IDs present in the
client history and load just those mappings, so an exact DSML replay can survive
server restarts even when the matching KV snapshot is not the one ultimately
used for the rendered-prefix hit.

The DS4 session payload starts with thirteen little-endian `u32` fields:

```text
0   magic = "DSV4"
1   payload version = 2
2   saved context size
3   prefill chunk size
4   raw KV ring capacity
5   raw sliding-window length
6   compressed KV capacity
7   checkpoint token count
8   layer count
9   raw/head KV dimension
10  indexer head dimension
11  vocabulary size
12  live raw rows serialized below
```

Then it stores:

- `u32[token_count]` checkpoint token IDs.
- `float32[vocab_size]` logits for the next token after that checkpoint.
- `u32[layer_count]` compressed attention row counts.
- `u32[layer_count]` ratio-4 indexer row counts.
- For every layer: the live raw sliding-window KV rows, written in logical
  position order rather than physical ring order.
- For compressed layers: live compressed KV rows and compressor frontier
  tensors.
- For ratio-4 compressed layers: live indexer compressed rows and indexer
  frontier tensors.

The logits are raw IEEE-754 `float32` values from the host `ds4_session`
buffer. They are saved immediately after the checkpoint tokens so a loaded
snapshot can sample or continue from the exact next-token distribution without
running one extra decode step. MTP draft logits/state are not persisted; after
loading a disk checkpoint the draft state is invalidated and rebuilt by normal
generation.

Distributed coordinator sessions use the same `DSV4` payload. Worker-owned
layer tensors are pulled during save and merged into the normal layer-ordered
tensor stream; during load the coordinator splits that stream into the current
route and pushes the relevant layer tensors back to the workers. The saved file
does not retain the distributed topology.

The tensor payload is DS4-specific KV/session state, not a generic inference
graph dump. It is expected to be portable only across compatible `ds4.c`
builds for this model layout.

The cache stores checkpoints at four moments:

- `cold`: after a long first prompt reaches a stable prefix, before generation.
- `continued`: when prefill or generation reaches the next absolute aligned frontier.
- `evict`: before an unrelated request replaces the live in-memory session.
- `shutdown`: when the server exits cleanly.

Cold saves intentionally trim a small token suffix and align down to a prefill
chunk boundary. This avoids common BPE boundary retokenization misses when a
future request appends text to the same prompt. The defaults are conservative:
store prefixes of at least 512 tokens, cold-save prompts up to 30000 tokens,
trim 32 tail tokens, and align to 2048-token chunks. The important knobs are:

Continued saves use the same alignment and are written only when the live graph
naturally reaches an absolute frontier. With the defaults this means roughly
every 10k tokens, independent of where the first cold checkpoint landed, so long
generations leave restart points behind without persisting the fragile final few
tokens.

- `--kv-cache-min-tokens`
- `--kv-cache-cold-max-tokens`
- `--kv-cache-continued-interval-tokens`
- `--kv-cache-boundary-trim-tokens`
- `--kv-cache-boundary-align-tokens`
- `--tool-memory-max-ids`
- `--disable-exact-dsml-tool-replay`

By default, checkpoints may be reused across the 2-bit and 4-bit routed-expert
variants if the rendered prefix matches. Use `--kv-cache-reject-different-quant`
when you want strict same-quant reuse only.

The cache directory is disposable. If behavior looks suspicious, stop the
server and remove it. You can investigate what is cached with hexdump as
the kv cache files include the verbatim prompt cached.

## Backends

The default graph backend is Metal on macOS and CUDA in CUDA builds:

```sh
./ds4 -p "Hello" --metal
./ds4 -p "Hello" --cuda
```

On Linux, plain `make` prints the available build targets instead of selecting a
CUDA target implicitly. Use `make cuda-spark` for DGX Spark / GB10. It omits an
explicit `nvcc -arch` because that is currently the fastest path on GB10. Use
`make cuda-generic` for a normal local CUDA build, or set `CUDA_ARCH` explicitly
when cross-building or when you need a known target:

```sh
make cuda CUDA_ARCH=sm_120
make cuda CUDA_ARCH=native
```

CUDA builds accept `--gpu-vram N[,N,...]` and `--gpu-devices N[,N,...]` in the
CLI, server, agent, and benchmark. VRAM values are per-device GiB budgets;
`--gpu-vram auto` uses the free memory reported by CUDA. The device list controls
the placement order and must have the same number of entries as an explicit
budget list. Placement reserves graph and KV memory for the requested context
and refuses to start if model layers would spill to the CPU.

Without `--cuda-tensor-parallel`, CUDA uses normal layer placement across the
listed devices. This is also the supported multi-GPU layout for GLM 5.2:

```sh
./ds4 -m gguf/GLM-5.2-UD-Q2_K_RoutedQ2K.gguf \
  --gpu-vram auto --gpu-devices 0,2,4,6,1,3,5,7 \
  --ctx 32768 -p "Hello"
```

For the DeepSeek Flash tensor/expert-parallel layout, see
"Tensor Parallelism across CUDA GPUs" above.

There is also a CPU reference/debug path:

```sh
./ds4 -p "Hello" --cpu
make cpu
./ds4
./ds4 -p "Hello"
```

Do not treat the CPU path as the production target. The CLI and `ds4-server`
support the CPU backend for reference/debug use and share the same KV session
and snapshot format as Metal and CUDA, but normal inference should use Metal or
CUDA.

## Steering

DeepSeek V4 Flash and GLM 5.3 Flash support steering with single-vector
activation directions; see the `dir-steering` directory for examples,
model-specific shapes, and build instructions. This follows the core idea of the
[Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)
paper. You can use it to make the model more or less verbose, less likely to
answer programming questions if it is a chatbot for your car rental web site,
and so forth, much faster than fine-tuning.
This is also useful for cybersecurity researchers who want to reduce a model's
willingness to provide dual-use or offensive security guidance.

Load a direction when starting either interactive client:

```sh
./ds4 -m model.gguf --dir-steering-file direction.f32
./ds4-agent -m model.gguf --dir-steering-file direction.f32
```

The default FFN scale is `1`. At an interactive prompt, `/steer` shows the
current scale, `/steer 0` disables it, and `/steer F` sets a value from `-100`
to `100` for subsequent tokens. The existing KV cache is kept. Live changes
are currently limited to local sessions, not distributed inference or network
tensor parallelism.

## Conversation Prefixes

`ds4` and `ds4-agent` can preload a conversation with `--prefix-file FILE`.
The file contains complete `USER:` / `ASSISTANT:` pairs:

```text
USER: Write terse answers.
This line is still part of the user turn.
ASSISTANT: Understood.
USER: Return code without an introduction.
ASSISTANT: I will return only the code.
```

A marker starts a new turn only at the beginning of a line. The prefix must
start with `USER:`, alternate roles, and end with `ASSISTANT:`. It is rendered
with the model's chat template and prefilled before the live prompt.

```sh
./ds4 --prefix-file prefix.txt
./ds4-agent --prefix-file prefix.txt
```

This can also overcome refusal behavior. For example, when a multi-turn prefix
shows the assistant writing exploits as part of computer-security work, most
models will comply with the next computer-security requests. The effect is
studied in *Many-shot Jailbreaking*.

## Test Vectors

`tests/test-vectors` contains short and long-context continuation vectors
captured from the official DeepSeek V4 Flash API. The requests use
`deepseek-v4-flash`, greedy decoding, thinking disabled, and the maximum
`top_logprobs` slice exposed by the API. Local vectors are generated with
`./ds4 --dump-logprobs` and compared by token bytes, so tokenizer/template or
attention regressions show up before they become long generation failures. The
C runner pins a 2048-token prefill chunk for this strict API-vector comparison.

The core local tests are driven by the C runner, with a small `ds4-eval`
extractor self-test run first:

```sh
make test                  # ./ds4-eval --self-test-extractors && ./ds4_test --all
./ds4_test --logprob-vectors
./ds4_test --server
```

The batching tests are model-backed and must run on the matching GPU backend:

```sh
# Metal, with DS4_TEST_SESSION_COUNT set to 2, 4, 8, and 16.
DS4_TEST_MODEL=/path/to/model.gguf DS4_TEST_SESSION_COUNT=4 \
  make test-metal-session-batch

# GLM 5.3 native row batching uses a bounded numerical comparison.
DS4_TEST_MODEL=/path/to/GLM-5.3-Flash-Q2.gguf \
  DS4_TEST_SESSION_COUNT=4 DS4_TEST_LOGIT_TOLERANCE=0.001 \
  make test-metal-session-batch

# CUDA multi-GPU Flash.
DS4_TEST_MODEL=/path/to/model.gguf make test-cuda-session-batch
DS4_TEST_MODEL=/path/to/model.gguf make test-cuda-mixed-batch
```

For GLM, run the same Metal session test with a GLM GGUF and run
`tests/glm_long_context_smoke.sh /path/to/model.gguf`. The official 100-case
quality scorers, two-Mac TCP/RDMA tests, CUDA matrix, and manual agent checks are
release gates rather than quick local tests; follow
[QA_BEFORE_RELEASES.md](QA_BEFORE_RELEASES.md).

## Debugging Notes

When a generation looks wrong, three small tools are usually enough to get a
first answer:

```sh
./ds4 --dump-tokens -p "..."
./ds4 --dump-logprobs /tmp/out.json --logprobs-top-k 20 --temp 0 -p "..."
./ds4 --dump-logits /tmp/logits.json --metal --nothink --prompt-file prompt.txt
./ds4-server --trace /tmp/ds4-trace.txt ...
```

- `--dump-tokens` prints the exact chat prompt token stream the CLI would use,
  then exits before inference starts. Add `--raw` to tokenize the `-p` or
  `--prompt-file` text literally. Protocol specials are recognized in both
  modes; for example, the DSML tool close marker starts as two tokens: `</`
  and `｜DSML｜`.
- `--dump-logprobs` stores a greedy continuation with the top local
  alternatives at each step, which helps separate sampling choices from
  logit/model issues.
- `ds4-server --trace` writes the rendered prompts, cache decisions, generated
  text, and tool-parser events for a whole agent session.

## Logo

The DwarfStar logo was designed by hand by Salvatore Sanfilippo, made more
graphical with AI, and manually reworked by Ben Gnomino, whose human touch made
it rock.
