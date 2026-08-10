# QA Before Releases

This is the release gate for DwarfStar.  Run it before tagging or pushing a
release build.  The goal is not to prove every code path exhaustively; it is to
exercise the paths that have historically regressed: Metal graph inference,
CUDA, ROCm, SSD streaming, distributed execution, disk KV cache, server APIs, and the
agent TUI/tool state machine.

Do not run multiple huge model processes at the same time.  Record the commit,
hardware, GGUF file, context size, and any non-default flags for every manual
run.

Preferred release test hosts:

- CUDA / DGX Spark: `toor@192.168.60.184`.
- Metal / distributed Mac testing: `mac-m5max-it` and `mac-m5max-us`.
- ROCm: The Strix Halo system at antirez@strixhalo (Framework Desktop).

`192.168.60.250` is permission-only. Never connect to it for QA, stop or start
its server, build there, or run tests or benchmarks there without asking
Salvatore and receiving explicit permission for that specific QA pass. Earlier
permission does not carry over to later work.

The Mac hosts have DNS entries and are reached through an internet VPN.  They
are connected to each other over WiFi and also through a Thunderbolt 5
point-to-point link.  The TB5 route is the preferred distributed-inference
network when it is available, but it can be fragile and sometimes only works
when `ds4` is executed in the foreground.  Prefer these machines for release
testing, especially distributed inference.  Local fallback testing on this
machine is acceptable when needed; it is an M3 Max with 128 GB RAM.
The Strix Halo system is reachable via the VPN as well and has a local WiFi
address in the same lan of the M5 Max systems. The CUDA hosts are in a
different remote lan and are accessible via a different VPN active
in this system.

## 1. Repository And Build Sanity

- Start from a clean tree except intentional release notes:
  `git status --short`.
- Build the normal local target:
  `make clean && make`.
- Build CPU-only binaries as a compile check only:
  `make clean && make cpu`.
- Treat compiler warnings as build failures. Save each release and test build's
  complete output and require no `warning:` or NVCC `warning #` lines. Fix the
  source when possible; use a narrow target-specific suppression only when a
  test deliberately compiles a partial translation unit.
- Repeat the warning-free build gate on the release hardware:
  `make clean && make` on Metal,
  `make clean && make cuda-spark` on DGX Spark,
  `make clean && make cuda-generic CUDA_HOME=/usr` on the multi-GPU CUDA host
  only after receiving permission for `192.168.60.250`,
  `make clean && make strix-halo` on Strix Halo.
- Run whitespace checks before committing:
  `git diff --check`.
- Confirm `./ds4 --help`, `./ds4-server --help`, and `./ds4-agent --help` render
  cleanly, with readable section colors and no broken wrapping.

## 2. Core Regression Tests

- Run the default suite:
  `make test`.
- Run `tests/test_gpu_args_cli.sh` explicitly after changing executable option
  parsing or multi-GPU placement. Invalid values and device/budget count
  mismatches must reach the shared GPU parser in all four binaries; an
  `unknown option` response from a binary that advertises the flag is a
  release blocker. On CUDA, also start `ds4-server` once with
  `--gpu-vram auto` and the intended `--gpu-devices` list and preserve the
  resolved layout line.
- Run the vector checks explicitly after any tokenizer, template, KV, kernel,
  quantization, or prompt-rendering change:
  `DS4_TEST_MODEL=/path/to/0731.gguf
  DS4_TEST_VECTOR_FILE=tests/test-vectors/flash-0731/official.vec
  ./ds4_test --logprob-vectors`
  and
  `DS4_TEST_MODEL=/path/to/0731.gguf
  DS4_TEST_LOCAL_GOLDEN_FILE=tests/test-vectors/flash-0731/local-golden.vec
  ./ds4_test --local-golden-vectors`.
- Run server tests when HTTP, SSE, prompt rendering, cache policy, or tool-call
  replay changed:
  `./ds4_test --server`.
- Run `./ds4-eval --self-test-extractors`.

### Critical Input And Server Regression Pass

Run these checks after changing parsers, server generation, model loading,
distributed snapshots, caches, DSpark, or CUDA build rules. Keep the item
numbers in the QA report so omissions are visible.

1. Send malformed OpenAI, Responses, and Anthropic requests with repeated
   owned string or array fields under ASan. Each request must fail cleanly and
   a following valid request must still work. Run `./ds4_test --server` too.
2. Replay at least 4,096 assistant/tool-result pairs through the Responses and
   Anthropic validators. Validation must remain linear-time and preserve the
   same accepted and rejected histories as a short replay.
3. Feed a distributed worker a snapshot header whose declared lengths exceed
   the configured and protocol limits. It must reject the header before a
   large allocation or payload read, without growing RSS materially.
4. Run malformed safetensors and GGUF fixtures through the loader and
   `gguf-tools/deepseek4-quantize` under ASan and UBSan. Truncated files,
   impossible dimensions, and overflowing tensor sizes must be rejected.
5. With a checkpoint-matched DSpark drafter, compare temperature-zero output
   against a no-drafter run for 400 and 800 generated tokens. Record the first
   output difference, acceptance, direct commits, replay fallbacks, and decode
   speed. Byte identity is not required: DSpark commits the batched verifier
   state, whose floating-point operation order differs from one-token decode.
   Verifier errors, invalid text, or a material continuation-quality regression
   remain release blockers. Use `--dspark-strict` for the target-only control.
6. Exercise unterminated and twice-closed reasoning in streaming and
   non-streaming OpenAI, Responses, and Anthropic requests, with and without
   tools. Reasoning must never leak into answer content.
7. On real Blackwell hardware, build the CUDA targets for `sm_120` or `sm_120a`
   and for DGX Spark `sm_121`. Confirm the emitted architecture flags retain
   the architecture-specific feature suffix and run `make cuda-regression`.
8. Build with CUDA 12.8 or newer and require the CUDA translation units to
   compile warning-free, including the `FLT_MAX` users.
9. Force a conversation past the in-memory KV threshold, restore the same disk
   checkpoint twice, and confirm the checkpoint file remains present after
   both successful loads. Corrupt checkpoints must still be rejected.
10. Run `make dspark-verify-depth` with matching 0731 target and drafter files.
    Strict capture must skip layers without a compressor and compare every
    captured compressor layer.
11. Send the same long GLM 5.2 prompt twice to one server session. The second
    request must report `cache_source: memory-rewind`, reuse through one token
    before the prompt boundary, and produce the same greedy output as a fresh
    session.
12. Run `./ds4_test --think-tool-recovery`, then repeat through all three HTTP
    APIs. A complete tool block inside unclosed reasoning must be recovered
    once, preceding prose must remain reasoning, and no synthetic continuation
    may be generated.
13. Run `./ds4_agent_test` under ASan with agent-cache strings whose declared
    lengths exceed the remaining file. Loading must fail without allocating
    the declared size, and a valid cache must still load.
14. Run the server parser tests under UBSan with `NaN`, positive infinity, and
    negative infinity where integer JSON fields are expected. Conversion must
    be defined and clamped, with no sanitizer report.

## 3. Official Continuation Quality Gates

These tests are release-blocking after tokenizer, template, KV-cache, attention,
MoE routing, quantization, logit, or model-graph changes.  They are
teacher-forced continuation checks against hosted-model output and API
top-logprob slices, so do not replace them with one sampled chat answer.

- Build the scorer:
  `make -C gguf-tools quality-score`.
- Match every Flash GGUF to the fixture captured from the same checkpoint.
  The current release checkpoint is 0731 and uses
  `tests/test-vectors/flash-0731/`; the older undated GGUF uses the preserved
  `tests/test-vectors/flash-pre-0731/` fixture. Never report a cross-checkpoint
  failure as a quality regression. New checkpoints require a new
  `flash-CHECKPOINT/` directory before release QA; do not replace an older
  fixture. Checkpoint-labelled GGUFs such as `-0731` must use the fixture with
  the same label.
- Run the tracked DeepSeek V4 Flash 0731 smoke vectors:
  `DS4_TEST_MODEL=/path/to/0731.gguf
  DS4_TEST_VECTOR_FILE=tests/test-vectors/flash-0731/official.vec
  ./ds4_test --logprob-vectors`.
  This covers short prompts and long-prompt attention cases. The runner defaults
  to this fixture, but release logs should keep the path explicit.
- Run the 100-case DeepSeek V4 Flash fixture for every released Flash GGUF:
  `gguf-tools/quality-testing/score_official /path/to/deepseek-v4-flash.gguf gguf-tools/quality-testing/data/flash/manifest.tsv /tmp/flash.tsv 4096`.
  This manifest is also for the 0731 checkpoint. A later checkpoint needs a
  separately named 100-case fixture and must not be scored against this one.
- Treat the native MXFP4 Flash GGUF as a separate release artifact. Run the
  same 100-case fixture on Metal, resident CUDA, and CUDA SSD streaming when
  those backends are advertised; compare each result with the Metal baseline.
  For resident multi-GPU CUDA, pass the normal placement flags to the scorer,
  for example `--gpu-vram auto --gpu-devices 0,2,4,6,1,3,5,7
  --cuda-tensor-parallel`.
- Run the 100-case GLM 5.2 OpenRouter fixture for every released GLM GGUF:
  `gguf-tools/quality-testing/score_official models/GLM-5.2-UD-Q4_K_XL.gguf gguf-tools/quality-testing/data/glm52-openrouter-100/manifest.tsv /tmp/glm52-q4.tsv 4096`.
  Current Q4 XL reference band: first-token match `95/100`, API top-1 agreement
  about `0.942`, and API pair-order agreement about `0.880`.
- Run the same GLM fixture for reduced-precision GLM release files.  The Q2
  routed reference is lower quality but should stay near first-token match
  `92/100`, API top-1 agreement about `0.890`, and API pair-order agreement
  about `0.800` unless the quantization changed deliberately.
- Run the 100-case DeepSeek V4 PRO fixture for every released PRO GGUF:
  `gguf-tools/quality-testing/score_official /path/to/deepseek-v4-pro.gguf gguf-tools/quality-testing/data/pro/manifest.tsv /tmp/pro.tsv 4096`.
- For SSD streaming, run the same official-continuation scorer once with full
  residency and once with `--ssd-streaming` for the release model.  The summary
  and API agreement should stay in the same quality band.
- Compare any candidate against the previous release or last-known-good output:
  `python3 gguf-tools/quality-testing/compare_scores.py /tmp/old.tsv /tmp/new.tsv`.
  Treat a large first-token-match drop, a clear NLL regression, or a material
  API top-1/pair-order regression as a blocker unless the release notes call out
  an intentional quality tradeoff.
- Keep the raw `summary` and `api_summary` lines in the release notes or QA log.
  Do not use stale manifests from `misc/` as release evidence.

## 4. Metal Flash Path

Use the normal Flash GGUF that 128 GB users run.

- One-shot CLI:
  `./ds4 -m ds4flash.gguf --ctx 32768 --nothink -p "Explain C pointers in one paragraph."`
- Thinking and max-thinking prompts:
  run one short coding prompt with default thinking and one with max thinking.
- Long-context recall:
  run the long name/number or archive recall test used for catching attention
  and MoE routing drift.
- Logprob sanity:
  `./ds4 --nothink --temp 0 --dump-logprobs /tmp/ds4-logprobs.json --logprobs-top-k 20 -p "..."`
  and inspect that the continuation is sane.
- Speed sanity:
  run `ds4-bench` with `speed-bench/promessi_sposi.txt` and compare prefill,
  generation speed, and KV bytes with the last known good numbers for the same
  machine.
- For native MXFP4 changes, run `make mxfp4-dot-test test-mxfp4-metal`, then a
  short greedy prompt and the section 3 continuation fixture with the MXFP4
  GGUF. The synthetic fused-MoE test and full-model quality gate must both pass.

### DSpark / DeepSpec Runtime

DSpark is opt-in, but it mutates the verifier, target-hidden capture, support
model loading, and proposal paths. Run these whenever DSpark support,
speculative verification, confidence policy, target hidden capture,
tiny routed-MoE verifier kernels, or shared `--mtp` support-model code changes:

Use the 0731 DSpark support GGUF only with a Flash 0731 target. A support model
from another checkpoint can have plausible acceptance statistics while
producing a different greedy continuation.

Normal DSpark runs commit accepted target-verifier state directly. The batched
verifier and one-token decode use the same graph with different floating-point
operation order, so `output_match=0` against the baseline is diagnostic rather
than a failure. `--dspark-strict` remains the byte-identical target-only mode.

- Default greedy acceptance fixture:
  `DS4_DSPARK_MODEL=/Users/antirez/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf DS4_DSPARK_SUPPORT=/Users/antirez/ds4/gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf make dspark-acceptance`.
- 64-token guardrail:
  `DS4_DSPARK_FIXTURE_TOKENS=64 DS4_DSPARK_MODEL=/Users/antirez/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf DS4_DSPARK_SUPPORT=/Users/antirez/ds4/gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf make dspark-acceptance`.
- Fixed-block direct partial commit:
  `DS4_DSPARK_FIXTURE_CONFIDENCE=0 DS4_DSPARK_FIXTURE_TOKENS=8 DS4_DSPARK_FIXTURE_REQUIRE_PARTIAL=1 DS4_DSPARK_MODEL=/Users/antirez/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf DS4_DSPARK_SUPPORT=/Users/antirez/ds4/gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf make dspark-acceptance`.
- DSpark verifier invariant smoke:
  `DS4_TEST_MODEL=/Users/antirez/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf DS4_DSPARK_SUPPORT=/Users/antirez/ds4/gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf make dspark-verify-depth`.
- For Metal DSpark verifier/proposer/replay changes, run this same-machine A/B
  matrix with `DS4_DSPARK_STATS=1`, greedy decoding, the same prompt and token
  limit, and no other environment changes:

  | Target expert cache | Expected DSpark depth | Legacy control | Candidate |
  | ---: | ---: | --- | --- |
  | 16 | 2 | `DS4_METAL_DSPARK_PROPOSER_BLOCK_MAX=0 DS4_METAL_DSPARK_ACCEPTANCE_ONLY_VERIFY=0 DS4_METAL_DSPARK_HEADLESS_REPLAY=0` | Leave proposer/headless unset; keep acceptance-only `=0` |
  | 32 | 5 | `DS4_DSPARK_SSD_VERIFY_BLOCK_MAX=5 DS4_METAL_DSPARK_ACCEPTANCE_ONLY_VERIFY=0 DS4_METAL_DSPARK_HEADLESS_REPLAY=0` | Keep the verifier cap, set acceptance-only `=1`, and leave proposer/headless unset |

  Use `--ssd-streaming-cache-experts 16` or `32` to match the row. The 0731
  top-6 verifier needs 30 effective slots for five draft rows; 32 leaves a small
  margin. Require byte-identical stdout between control and candidate and
  `errors=0`, `verifier_unavailable=0`, `proposed>0`, and
  `accepted_draft>0`. Record generation t/s, acceptance, `propose`, `verify`,
  `replay`, `prop_capped`, `prop_scheduled_rows`, `metal_accept_only`,
  `metal_verify_rows_saved`, and `metal_replay_headless`. In the candidate,
  eligible `N >= 3` verification cycles should save one target row; aligned
  ratio-4 boundaries intentionally remain on the legacy path. The
  depth-2 run exercises proposer capping and headless replay while retaining
  the legacy verifier; the depth-5 run should retain the checkpoint's native
  five proposal rows and exercise acceptance-only verification.
  On low-memory Metal, repeat the depth-5 candidate once with
  `DS4_METAL_DSPARK_PIN_MAIN_PROJ=1`. Require a startup log confirming the
  locked byte count, identical stdout and acceptance, and compare
  `prop_setup`, total `propose`, page faults, and generation t/s. A lock
  failure or a slower median keeps this optimization opt-in.
- For the experimental Metal SSD exact-2 verifier, repeat the depth-2 row
  above with `DS4_METAL_DSPARK_EXACT2=0` as the control and `=1` as the only
  candidate change. Set `DS4_DSPARK_FIXTURE_REQUIRE_EXACT2=1` only on the
  candidate. Require byte-identical stdout against both control and
  target-only output, `exact2_attempt>0`, `exact2_full>0`,
  `exact2_fallback=0`, and `errors=0`. Record generation t/s, `verify`, and
  `replay`; then repeat for at least 100 generated tokens to catch cumulative
  state drift. Do not infer that the generic five-row batch state is directly
  committable from this two-row result.
- For Metal exact-union or the AProjQ4/HC decode fusions, first run the
  model-backed oracle with the target AProjQ4 GGUF:
  `DS4_TEST_MODEL=/path/to/deepseek-v4-flash-aprojq4.gguf make test-metal-exactn-oracle`.
  Require its N=2..5 cases to be byte-identical to sequential decode for
  serialized KV/compressor state, logits, and the four-token continuation.
  The matrix must include full accepts for N=2,3,4,5, all N=5 partial prefixes
  1..4, and EOS in the first and a middle row. This is a correctness gate, not
  evidence of a speedup.
- The Q8 Q-A/KV compound rows below require a separate AProjQ8 target whose
  metadata includes both ratio-4 and ratio-128 compressor layers. An AProjQ4
  oracle cannot exercise that compound and is a failed coverage gate even if
  greedy output remains correct.
- Then run isolated, same-machine greedy A/B pairs with identical prompt,
  context, cache, token limit, and `DS4_DSPARK_STATS=1`. Change only the gate
  named by the row:

  | Metal fusion | Reference control | Candidate |
  | --- | --- | --- |
  | HC RMSNorm + F16 mixer on M1-M4 | `DS4_METAL_DISABLE_PRE_M5_HC_NORM_MIX_FUSE=1` | Leave the disable switch unset |
  | HC RMSNorm + F16 mixer on another Apple generation | Leave both HC norm/mix switches unset | `DS4_METAL_ENABLE_HC_NORM_MIX_FUSE=1` |
  | HC producer + split/Sinkhorn/destination RMSNorm on M1-M5 | `DS4_METAL_DISABLE_HC_PRODUCER_PRE_NORM_FUSE=1` | Leave the disable switch unset |
  | Q4 Q-A/KV + compressor store in exact-union | `DS4_METAL_DSPARK_EXACTN_UNION=1` with the Q4 enable switch unset | Keep exact-union `=1`; set `DS4_METAL_ENABLE_Q4_QKV_COMPRESSOR_FUSE=1` |
  | Q4 Q-A/KV + compressor store in ordinary `FULL` decode | Leave `DS4_METAL_ENABLE_Q4_QKV_COMPRESSOR_FUSE` unset | Set `DS4_METAL_ENABLE_Q4_QKV_COMPRESSOR_FUSE=1` |
  | Q8 Q-A/KV + compressor store in SSD `FULL` (AProjQ8) | Leave the Q8 enable/require switches unset | Set `DS4_METAL_ENABLE_Q8_QKV_COMPRESSOR_FUSE=1 DS4_METAL_REQUIRE_Q8_QKV_COMPRESSOR_FUSE=1` |
  | Q8 Q-A/KV + compressor store in SSD exact-union (AProjQ8) | `DS4_METAL_DSPARK_EXACTN_UNION=1` with the Q8 enable/require switches unset | Keep exact-union `=1`; set `DS4_METAL_ENABLE_Q8_QKV_COMPRESSOR_FUSE=1 DS4_METAL_REQUIRE_Q8_QKV_COMPRESSOR_FUSE=1` |
  | Q4 attention-output tiny batch in the generic verifier | Set `DS4_METAL_DSPARK_EXACTN_UNION=0 DS4_METAL_DSPARK_EXACTN=0 DS4_METAL_DSPARK_EXACT2=0`; leave tiny enable/require unset | Keep all three exact gates `=0`; set `DS4_METAL_REQUIRE_Q4_ATTN_OUT_TINY_BATCH=1` and require at least one proposed block of depth 3–5 (the acceptance-only suffix evaluates one fewer row) |
  | F16 attention+indexer quad compressor store in `FULL` decode | `DS4_METAL_DISABLE_COMPRESSOR_QUAD_STORE=1` | Leave the disable switch unset |
  | F16 attention+indexer quad compressor store in exact-union | `DS4_METAL_DSPARK_EXACTN_UNION=1 DS4_METAL_DISABLE_COMPRESSOR_QUAD_STORE=1` | Keep exact-union `=1`; leave the quad disable switch unset |
  | Exact ratio-4 one-row compressor pool on M1-M5 | `DS4_METAL_DISABLE_COMPRESSOR_EXACT_POOL_RATIO4=1` | Leave the disable switch unset |
  | Q4 attention-output B + HC expansion | `DS4_METAL_DISABLE_Q4_ATTN_OUT_HC_FUSE=1` | Leave the disable switch unset |
  | FlashAttention pad/block PSO memo | `DS4_METAL_DISABLE_PRE_M5_FLASH_ATTN_PAD_BLK_MEMO=1` | Leave the disable switch unset |
  | FlashAttention batched/vector PSO memo | `DS4_METAL_DISABLE_PRE_M5_FLASH_ATTN_BATCHED_MEMO=1` | Leave the disable switch unset |
  | Exact-union asynchronous routed tails | `DS4_METAL_DSPARK_EXACTN_UNION=1` with `DS4_METAL_DSPARK_EXACT_ROWS_ASYNC_TAILS` unset | Keep exact-union `=1`; set `DS4_METAL_DSPARK_EXACT_ROWS_ASYNC_TAILS=1` |

  The Q4 Q-A/KV compound is opt-in in both exact-union and ordinary `FULL`
  decode; it is enabled only when the explicit enable variable is present.
  Require byte-identical stdout and `errors=0`; for exact-union also
  require `exactn_union_attempt>0` and `exactn_union_error_fallback=0`.
  Partial-accept fallback is expected when the draft diverges. Record
  `exactn_union_full`, `exactn_union_partial_fallback`, `propose`, `verify`,
  `replay`, stage timings, page faults, and generation t/s. A candidate that
  is correct but slower remains disabled or opt-in according to its gate.
  For asynchronous tails, also repeat the model-backed oracle with the switch
  set and run enough exact-union cycles to cross cache eviction and raw-ring
  wrap boundaries. The candidate removes a CPU wait but retains private expert
  buffers until command-buffer completion; serialized state and process memory
  after synchronization must match the synchronous control.
  Before the model-backed runs, build `ds4_test` and run
  `./ds4_test --metal-kernels`. This covers the isolated compound HC, F16 quad
  compressor-store, exact ratio-4 pool, and tie-heavy Metal routing kernels.
  For the exact one-row pool candidate, repeat once with
  `DS4_METAL_REQUIRE_COMPRESSOR_EXACT_POOL_RATIO4=1`; the run must exercise the
  specialization instead of silently falling back. Also exercise the global
  kill switches plus the matching pre-M5 or M5 HC/pool rollback on the target
  machine. Treat the FlashAttention memo rows as host-dispatch A/B tests: the
  selected specialization and output must remain identical, and any timing
  comparison must use repeated warm runs.
- For the removed Metal 512-column streaming top-k path, there is no runtime
  candidate gate. Compare the current binary with a build immediately before
  its removal only if historical timing is needed. First require
  `./ds4_test --metal-kernels` to pass, including tie-heavy routing cases, then
  require identical selected expert ids and greedy output. Correct deterministic
  ordering takes precedence over a timing difference.
- For the default CPU unrolled argmax, run `tests/test_sampling`, then compare
  an otherwise identical greedy workload with
  `DS4_CPU_DISABLE_UNROLLED_ARGMAX=1` (scalar control) and with the variable
  unset (candidate). Require identical tokens for ordinary, excluded-id,
  cross-lane-tie, and vocabulary-tail cases; record median generation t/s over
  repeated runs without claiming a speedup from the implementation alone.
- For the experimental resident-CUDA exact-2 verifier, use three controlled
  runs with verifier cap two on the same single-GPU host: native proposer plus
  legacy verifier
  (`DS4_CUDA_DSPARK_EXACT2=0 DS4_CUDA_DSPARK_PROPOSER_BLOCK_MAX=0 DS4_DSPARK_SSD_VERIFY_BLOCK_MAX=2`),
  two-row proposer plus legacy verifier
  (`DS4_CUDA_DSPARK_EXACT2=0 DS4_CUDA_DSPARK_PROPOSER_BLOCK_MAX=2 DS4_DSPARK_SSD_VERIFY_BLOCK_MAX=2`),
  and two-row proposer plus exact-2
  (`DS4_CUDA_DSPARK_EXACT2=1 DS4_CUDA_DSPARK_PROPOSER_BLOCK_MAX=2 DS4_DSPARK_SSD_VERIFY_BLOCK_MAX=2`).
  This separates the non-causal proposer-width change from the verifier and
  replay change. Then compare uncapped legacy DSpark against exact-2 as an
  end-to-end policy test. Keep SSD streaming and TP disabled. Require
  byte-identical stdout, `errors=0`, and `verifier_unavailable=0` from every
  run; set `DS4_DSPARK_FIXTURE_REQUIRE_EXACT2=1` on the exact-2 run so the
  fixture enforces `exact2_attempt>0` and `exact2_fallback=0`.
  Record `prop_scheduled_rows/cycles`, `propose`, `verify`, `replay`, `net_saved`,
  `miss_first`, `no_draft`, `avg_accept`, and generation t/s from every run.
- For resident CUDA exact-N, keep exact-2 disabled and compare
  `DS4_CUDA_DSPARK_EXACTN=0` against `=1` with the native five-row proposer
  and `DS4_DSPARK_SSD_VERIFY_BLOCK_MAX=5`. Repeat N=2,3,4,5 with explicit
  proposer/verifier caps, then exercise the kill switch with both
  `DS4_CUDA_DSPARK_EXACTN=1` and
  `DS4_CUDA_DISABLE_DSPARK_EXACTN=1`. Require byte-identical greedy stdout,
  `errors=0`, `verifier_unavailable=0`, `cuda_exactn_attempt>0`, and at least
  one `cuda_exactn_full`; partial cases must increment the partial and
  aggregate fallback counters, never the error counter, and continue
  identically through legacy replay. Include
  EOS as the first and a middle draft, a raw-ring wrap boundary, a context
  capacity cut, and prefill workspaces below five rows. Record
  `cuda_exactn_rows`, its full/partial/error counters, `snapshot`, `verify`,
  `replay`, acceptance, and generation t/s. Run with CUDA decode graphs both
  enabled and disabled. Do not promote the gate without a CUDA device build
  and serialized KV/compressor-state oracle; host syntax tests do not execute
  this path. On candidate fixture runs set
  `DS4_DSPARK_FIXTURE_REQUIRE_CUDA_EXACTN=1`; it requires aggregate
  `cuda_exactn_attempt>0` and `cuda_exactn_error_fallback=0`. It reports but
  does not reject aggregate `cuda_exactn_fallback`, because valid partial
  matches increment both the partial and aggregate fallback counters before
  legacy replay.
- For CUDA DSpark non-causal proposer attention, compare the reference with
  `DS4_CUDA_ENABLE_DSPARK_NONCAUSAL_ONLINE=0` against the candidate with `=1`.
  Repeat at proposal depths two and five, across every raw-ring start index,
  and once with both the enable variable and
  `DS4_CUDA_DISABLE_DSPARK_NONCAUSAL_ONLINE=1` to prove the kill switch restores
  the reference dispatch. On the short diagnostic runs also set
  `DS4_DSPARK_VERIFY_NONCAUSAL=1`; record all three reported `max_abs` and
  `max_rel` comparisons and reject non-finite values or a material error
  regression. Then run the acceptance fixture without the diagnostic host
  readbacks and require byte-identical target stdout, `errors=0`, and
  `verifier_unavailable=0`. Record proposal time, acceptance, generation t/s,
  and the startup dispatch log. Draft logits or acceptance may differ slightly
  because online softmax changes the floating-point reduction order; that is
  not permission for the verified target continuation to differ.
- For CUDA HC and tiny routed-MoE kernel changes, keep
  `DS4_CUDA_DSPARK_EXACT2` unset and repeat the resident acceptance fixture
  with these explicit A/B pairs: HC control
  `DS4_CUDA_DISABLE_HC_SPLIT_NORM_FUSED=1` versus candidate with that variable
  absent; routed-MoE control `DS4_CUDA_DSPARK_TINY_ALIGNED_VEC=0` versus
  candidate `=1`. Require byte-identical stdout, `errors=0`, and
  `verifier_unavailable=0`; record `prop_chain`, `verify_layer`, total
  proposal/verify time, acceptance, and generation t/s. Also run
  `--decode-consistency 64` and the logprob-vector regression before enabling
  a numerically different kernel by default.
- For the CUDA AProjQ4 ports, run an isolated A/B for each dispatch:
  Q-A/KV pair control `DS4_CUDA_DISABLE_Q4_DENSE_PAIR=1` versus candidate with
  that variable absent; HC norm/mix control
  `DS4_CUDA_DISABLE_HC_NORM_MIX_FUSE=1` versus candidate
  `DS4_CUDA_ENABLE_HC_NORM_MIX_FUSE=1 DS4_CUDA_NO_F16_CUBLAS_ONE=1`; and Q4
  attention-output/HC control `DS4_CUDA_DISABLE_Q4_ATTN_OUT_HC_FUSE=1` versus
  MMVQ-safe candidate `DS4_CUDA_ENABLE_Q4_ATTN_OUT_HC_FUSE=1`. Run a separate
  non-captured diagnostic with `DS4_CUDA_Q4_ATTN_OUT_HC_ORACLE=1`; require
  the summary to be present with `calls>0`, `skips=0`, and
  `epilogue_mismatches=0`, while `q8k_mismatches` records the expected
  numerical distance from the optional one-dispatch Q8_K experiment. A
  zero-call summary is a failed coverage gate. Only
  test `DS4_CUDA_Q4_ATTN_OUT_HC_Q8K_EXPERIMENT=1` as a promotion candidate if
  its oracle mismatches are also zero. Repeat the pair and attention-output
  cases with `DS4_CUDA_MMQ=0` to exercise the canonical Q8_K fallback
  separately from the default MMVQ/Q8_1 path. Require byte-identical stdout
  and full-logit/tensor equivalence before promoting an opt-in gate. Run with
  decode graphs both enabled and disabled, and record target, proposer,
  verifier, replay, acceptance, and generation t/s. A CUDA build and hardware
  run are mandatory; a host-only build does not compile the device kernels.
- When DSpark, support-model mapping, or SSD streaming changes, repeat both
  the acceptance fixture and verifier invariant on every advertised graph
  backend. Apply the backend and SSD options to the target-only baseline as
  well as the DSpark run:

  ```sh
  DS4_DSPARK_MODEL=/path/to/flash-0731.gguf \
  DS4_DSPARK_SUPPORT=/path/to/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  DS4_DSPARK_FIXTURE_BACKEND=cuda \
  DS4_DSPARK_FIXTURE_SSD_STREAMING=1 \
  DS4_DSPARK_FIXTURE_SSD_STREAMING_CACHE_EXPERTS=32 \
  DS4_DSPARK_FIXTURE_CONFIDENCE=0 \
  make dspark-acceptance

  DS4_TEST_MODEL=/path/to/flash-0731.gguf \
  DS4_DSPARK_SUPPORT=/path/to/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  DS4_TEST_SSD_STREAMING=1 \
  DS4_TEST_SSD_STREAMING_CACHE_EXPERTS=32 \
  make dspark-verify-depth
  ```

  On Strix Halo use the same variables with
  `DS4_DSPARK_FIXTURE_BACKEND=rocm make rocm-dspark-acceptance` and
  `make rocm-dspark-verify-depth`. Do not use the generic targets after a ROCm
  build: on non-Apple hosts their default object set is CUDA. For the 0731
  Flash layout, ROCm needs at least 30 expert slots; use 32 in release tests.
- The fixture must report aggregate `proposed>0`, `accepted_draft>0`,
  `verifier_unavailable=0`, and `errors=0`; stdout must remain byte-identical
  to the target-only SSD baseline. The verifier smoke must report
  `max_chunk>1`, `nspec>64`, and `worst_argmax_gap<=2`.
- Preserve baseline and DSpark `generation` t/s from the same fixture run,
  with the same host, model, cache, runtime settings, thermal state, and
  background load. A DSpark path that is materially slower than the target-only
  SSD path without a documented correctness tradeoff is a release blocker.
- On ROCm, also run one `DS4_DSPARK_PROBE=1`
  generation and require the non-causal attention and stage-chain probes to
  pass. This covers the HIP draft-attention kernel before the end-to-end
  verifier gate.
- If shared support-model or verifier structures changed, also run legacy MTP:
  `make mtp-verify-depth` with `DS4_TEST_MTP` set to a one-stage MTP support
  GGUF, or confirm the target skips only because the optional file is missing.
- Record `c_add` `accepted_draft`, `direct_full`, `direct_partial`,
  `replay_fallbacks`, `errors=0`, `verify_layer`, `net_saved`, and
  `output_match` for both 32-token and 64-token runs. At least one direct commit
  must occur. A faster run with lower proposal quality is a regression unless
  it was an intentional confidence-policy change.
- If verifier MoE kernels changed, run one diagnostic `c_add` profile with
  `DS4_DSPARK_VERIFY_SELECTED_PROFILE=1` or the Metal MoE stage profiler and
  record the selected-expert footprint or stage timing in the DSpark log.

### Session Microbatching And Metal TP

Run these gates whenever session scheduling, batched decode, mixed
prefill/decode, QKV projection, shared or routed experts, tensor parallelism,
or backend fallback selection changes.

- On a single Metal machine, run the full-vocabulary exact-logit oracle with
  2, 4, 8, and 16 sessions:
  `DS4_TEST_MODEL=/path/to/ds4flash.gguf DS4_TEST_SESSION_COUNT=N make test-metal-session-batch`.
  Compatible resident Q8 runs must report `native_shared=1 native_qkv=1` at
  every tested count. The 16-session run covers row counts above the old
  artificial eight-row limit.
- Repeat the four-session oracle with
  `DS4_METAL_SESSION_BATCH_SHARED=0` and with
  `DS4_METAL_SESSION_BATCH_QKV=0`. The first run must use the complete fallback;
  the second may batch the shared expert only. Both must remain bit-exact.
- The oracle must cover reversed row ordering, at least six decode steps, and a
  mixed prefill/decode call. Any nonzero differing-logit count is a blocker;
  argmax-only agreement is insufficient.
- Benchmark 1, 2, 4, 8, and 16 simultaneous resident sessions on the same host
  and model. Record model-step latency and aggregate decode tokens/second, not
  only request completion speed. The current Metal path batches QKV and part of
  the shared expert, but still runs attention, routed experts, shared down, and
  the output head per session. Treat flat aggregate scaling as unfinished
  implementation work, not evidence that Metal cannot benefit from batching.
- On `mac-m5max-it` and `mac-m5max-us`, run the same oracle in physical TP mode
  over explicit `tcp` and `rdma` transports. Set `DS4_TEST_TP_MODE=leader` on
  the leader and `DS4_TEST_TP_MODE=worker DS4_TEST_TP_LEADER_HOST=HOST` on the
  worker, with a unique `DS4_TEST_TP_PORT`. Run at least 2 and 4 sessions and
  preserve both logs. TP currently uses the ordered per-session fallback, so
  native single-machine row-grid flags must remain off in the TP logs.
- For the current TB5 MacBook link, US is `10.99.0.2` on `en1`/`rdma_en1` and
  IT is `10.99.0.1` on `en6`/`rdma_en6`; both use GID index 1. Before testing,
  require `rdma_ctl status` to report `enabled` and `ibv_devinfo -v` to show
  `PORT_ACTIVE` plus the corresponding `::ffff:10.99.0.x` GID. Force the
  device and GID with `--rdma-device NAME --rdma-gid-index 1` if automatic
  selection is ambiguous. A working TB IP ping alone is not RDMA evidence.
- Kill the TP worker during one batch with `DS4_TEST_TP_DISCONNECT=1` on the
  leader. The operation must fail cleanly, invalidate every affected session,
  and return control without hanging.
- Verify unsupported combinations explicitly: GLM, DSpark/MTP support models,
  SSD streaming, quality/reference modes, steering, resident Q4 expert
  overlap, and CPU-router modes must use the established exact fallback or
  reject the combination before evaluation. They must not partially activate
  native batching.

## 5. Metal PRO Path

PRO support is experimental, but release builds must not break it silently.

- If a PRO-capable machine is available, run a short PRO q2 prompt and verify
  the correct template, thinking behavior, and endpoint aliases.
- For PRO Q4 distributed builds, test only on the intended high-memory machines.
- If PRO cannot be run locally, at least build all binaries and review changes
  touching model shape, tensor lookup, routed expert mapping, template logic,
  and KV payload compatibility.

## 6. GLM 5.2

GLM has a different template, model shape, MTP block, attention layout,
tensor-parallel gate width, and streaming policy. Flash or PRO success does not
substitute for this matrix.

- On a 512 GB Metal machine, run short greedy prompts with both the Q4 XL and
  reduced-precision Q2 release GGUFs. Cover thinking and no-thinking templates,
  and verify the server reports the GLM model family rather than a DeepSeek
  alias internally.
- Run the OpenRouter smoke vectors explicitly:
  `DS4_TEST_MODEL=/path/to/glm.gguf
  DS4_TEST_VECTOR_FILE=tests/test-vectors/glm-openrouter/official.vec
  ./ds4_test --logprob-vectors`.
  Preserve the report as a diagnostic. The hosted vectors include very
  low-probability top-20 tails whose membership is not stable after GLM routed
  expert quantization, so an individual `official top token missing locally`
  assertion is not by itself a release blocker. Selected-token mismatches must
  remain consistent with the model's 100-case first-token band, and the
  section 3 scorer is the release gate for aggregate GLM quality.
- Run the 100-case Q4 XL and Q2 official fixtures from section 3 and preserve
  both `summary` and `api_summary` lines. Compare against the documented Q4 and
  Q2 reference bands independently.
- Run `tests/glm_long_context_smoke.sh` with the release-advertised context on
  the 512 GB Metal host. The generated continuation must begin with `>` and
  contain none of the known token-corruption markers.
- Exercise integrated GLM MTP with `--glm-mtp-timing` on a deterministic
  prompt. Compare the greedy text to
  a non-MTP run, require clean speculative cycles, and record acceptance and
  timing. Also run once with MTP disabled to prove ordinary decode remains the
  default.
- Run the Metal session oracle with 2 and 4 GLM sessions. It must report
  `family=glm native_shared=0 native_qkv=0` and remain exact, including mixed
  prefill/decode; the DeepSeek-only row-grid kernels must not activate.
- Run resident and SSD-streaming GLM Q2 prompts with the same greedy input.
  Compare first token and top-logprob sanity, and record the selected full-layer
  prefix and dynamic expert-cache budget.
- Run physical two-machine GLM TP over TCP and RDMA with short and long prompts.
  Record prefill/decode speed, transport, rank residency, and clean shutdown.
  Repeat one run with `--tensor-parallel-token-prefill` as the exact-arithmetic
  diagnostic. Use a GGUF whose routed-expert type has ownership-aware GLM TP
  kernels. Also test a Q4-routed GLM file as a negative gate: until Q4 ownership
  kernels are implemented, both ranks must reject it clearly before evaluation
  rather than loading a partial split or hanging.
- With explicit permission for the current QA pass, run one resident GLM Q2
  prompt, a long-context prompt, integrated GLM MTP, and concurrent server
  requests on the eight-GPU CUDA host. Use ordinary eight-GPU layer placement
  for GLM; do not pass the Flash-specific
  `--cuda-tensor-parallel` option. Multi-tier GLM prefill must
  report progress through the tier-switching token-major path, and decode,
  cache updates, and output-head/logit assembly must complete without CPU spill.
  Auto-placement must reserve each layer's compact DSA/indexer cache and the
  graph workspace before loading weights; a late graph-allocation failure is a
  release blocker. Confirm the long-context layout stays within every device's
  budget and uses additional tiers when the cache no longer fits on the earlier
  ones.
  The long-context harness can select this backend with
  `DS4_GLM_BACKEND=cuda` and pass placement flags through
  `DS4_GLM_EXTRA_ARGS="--gpu-vram auto --gpu-devices 0,2,4,6,1,3,5,7"`.
- Through `ds4-server`, exercise OpenAI chat, Responses, and Anthropic requests
  against GLM, including thinking and SSE. DeepSeek compatibility endpoint
  aliases may resolve to the loaded model, but rendered prompts and generated
  text must use the GLM template.

## 7. SSD Streaming

SSD streaming is a capacity path, so test both correctness and user experience.

- Flash q2/q2-q4 streaming:
  `./ds4 -m ds4flash.gguf --ssd-streaming --ssd-streaming-cache-experts 32GB -p "..."`
- Regression test mixed-quant Flash SSD streaming. Use the mixed q2/q4 GGUF
  with boosted Q4 routed-expert layers and a prompt long enough to exercise the
  selected-address prefill path; it must not fail with "model range is not
  covered by mapped model views":
  `./ds4 -m gguf/DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf --ssd-streaming --ssd-streaming-cache-experts 16GB --ctx 4096 --tokens 1 --nothink --prompt-file /tmp/ds4_600tok_prompt.txt`.
- Cold streaming measurement:
  run once with `--ssd-streaming-cold` and verify no deadlock, missing expert,
  or impossible slowdown.
- Confirm startup reports cache budget and that generation does not stall on
  repeated expert misses for a small interactive prompt.
- If streaming cache internals changed, test the same prompt twice and compare
  first-token/logprob sanity between runs.

## 8. CUDA / DGX Spark

Before a release, ask the user for CUDA access if it is not already configured.
Use the DGX Spark / GB10 host `toor@192.168.60.184`.  Do not claim CUDA is
release-ready without this pass.

- Fetch or push the exact release commit to the CUDA machine.
- Build:
  `make clean && make cuda-spark`.
- Require both the DGX Spark build and the eight-GPU CUDA build to complete
  without compiler warnings. The eight-GPU build is performed only after
  receiving explicit permission to use `192.168.60.250` for this QA pass.
- Run:
  `make cuda-regression`.
- On a single GB10 (`sm_121`), validate the imported Q2 decode fast paths with
  the AProjQ8/OutQ8 Flash GGUF.  Compare the default against a rollback process
  that sets all of:
  `DS4_CUDA_NO_DIRECT_Q2_PREFILL=1`,
  `DS4_CUDA_NO_F16_PAIR_COMPRESSOR_STORE=1`,
  `DS4_CUDA_NO_F16_PAIR_COMPRESSOR_TRANSPOSE=1`,
  `DS4_CUDA_NO_F16_PAIR_COMPRESSOR_TRANSPOSE_PREFETCH8=1`,
  `DS4_CUDA_NO_Q8_FUSED_ALIGNED=1`,
  `DS4_CUDA_NO_Q8_ALIGNED_PERSISTENT=1`,
  `DS4_CUDA_NO_Q8_ALIGNED_DENSE_SCRATCH=1`, and
  `DS4_CUDA_NO_HC_SPLIT_NORM_SPLIT4096=1`.  Use separate processes, require
  byte-identical greedy stdout and per-token logprobs, then run the same pair
  under Compute Sanitizer.  Record prefill, decode, and steady decode rather
  than copying the upstream PR numbers into a release claim.
- Repeat the GB10 comparison with AProjQ4/OutQ8.  First run
  `make test-mmq-parity-cuda CUDA_ARCH=sm_121`; its Q4 cases must report zero
  bit mismatches for persistent scratch, grouped attention-A, and the
  opt-in K1024 persistent kernel.  For the model A/B, use
  `DS4_CUDA_NO_Q4_GB10_FAST=1` in the control and leave it unset in the
  candidate.  Run a separate candidate with
  `DS4_CUDA_Q4_GROUPED_ATTN_A_ORACLE=1` and require `calls>0`,
  `mismatches=0`, and `skips=0`.  Benchmark the K1024 persistent kernel as a
  separate fail-closed arm with both
  `DS4_CUDA_ENABLE_Q4_K1024_PERSISTENT=1` and
  `DS4_CUDA_REQUIRE_Q4_K1024_PERSISTENT=1`; its rollback is
  `DS4_CUDA_NO_Q4_K1024_PERSISTENT=1`.  The persistent OutQ8
  vocabulary, compressor, HC split, direct routed-MoE paths, Q4 scratch,
  grouped attention-A, and canonical B+HC epilogue remain relevant, while
  the Q8-only attention-projection consumers are intentionally ineligible.
- Exercise CUDA DSpark at verifier/proposer depth 5 with the fast paths enabled
  and disabled.  Require identical final output, zero verifier errors, and
  matching full/partial acceptance histograms.  Test both the generic batch
  verifier (direct Q2 path) and CUDA exact-N (one-row decode paths); do not
  infer speculative speedup from the target-only benchmark.
- For native MXFP4 changes, run
  `make test-mxfp4-cuda CUDA_ARCH=native` on the multi-GPU CUDA host only after
  receiving explicit permission for `192.168.60.250`, and
  `make test-mxfp4-cuda CUDA_ARCH=sm_121` on DGX Spark. Dense MMQ, routed MMQ,
  routed MMVQ, fused gate/up, and fused down must pass. The Spark run must also
  pass the Blackwell K-tile guard. This synthetic parity test does not replace
  full-model continuation scoring.
- With that permission, run the native MXFP4 GGUF resident on the multi-GPU
  host, and run it with `--ssd-streaming` on DGX Spark. Use the same greedy prompt and continuation
  fixture on both. Record prefill and generation speed, require finite logits,
  and compare quality with the Metal MXFP4 result. Blackwell MMQ quantizes
  activations to native FP4 for batched work; decode MMVQ keeps Q8 activations,
  so quality must be checked rather than inferred from kernel-only parity.
- Run a short CLI prompt with the Flash GGUF and record generation t/s.
- Run a longer prompt that exercises routed experts past a few thousand tokens.
- With explicit permission for this QA pass, run the full-vocabulary decode
  oracle on the eight-GPU CUDA host:
  `DS4_TEST_MODEL=/path/to/flash.gguf make test-cuda-session-batch`.
  Preserve the per-batch timing for 2, 4, and 8 rows and require
  `nonexact_logits=0`. Run the released Q4 file and the reduced-precision Q2
  file: Q4 exercises grouped routed/shared stages, while unsupported Q2 native
  MoE shapes must retain the ordered exact fallback.
- With CUDA TP attention enabled, compatible Q4 runs must use grouped
  attention-core, QKV, KV-store, and attention-post by default and remain
  full-vocabulary exact against isolated decode. On the eight-L40S host, the
  16-row decode step must remain above 110 aggregate tokens/s. Repeat once with
  `DS4_CUDA_TP_ATTN=0` only as rollback coverage; it is not the production
  configuration.
- Run native mixed prefill/decode at the default frontier and at compressed
  context:
  `DS4_TEST_MODEL=/path/to/flash.gguf make test-cuda-mixed-batch` and
  `DS4_TEST_CONTEXT=4096 DS4_TEST_MIXED_INITIAL=2048 DS4_TEST_MIXED_ROUNDS=8
  DS4_TEST_MODEL=/path/to/flash.gguf make test-cuda-mixed-batch`.
  Every round must report exact logits and `mode=native`; a serialized fallback
  is a failure for the eight-GPU TP/EP topology. Under CUDA TP attention, the
  native mixed step must use the same exact grouped decode stages when their
  capability checks pass; record correctness and speedup separately. Also
  force an 800-row prefill quantum with
  `DS4_TEST_ALLOW_FALLBACK=1`; it must report the serialized safety fallback.
- With explicit permission for the eight-GPU host, start `ds4-server` with 8
  and 16 batched sessions and issue at least that many simultaneous requests
  with mixed prompt lengths. Verify no session mix-up, deadlock, or starvation
  and record aggregate generation throughput.
- On DGX Spark, verify the same public batch API and server concurrency use the
  single-GPU fallback without creating peer-only TP/EP state. The eight-GPU
  native oracle is not a valid Spark test because its topology is intentionally
  unavailable there.
- If CUDA Q4, distributed, streaming hooks, tensor span loading, or model cache
  code changed, test the specific GGUF and split mode that uses that path.
- Verify that any CUDA-only warning fixes are also clean on macOS and do not
  change Metal behavior.

## 9. ROCm / Strix Halo

Use the Strix Halo Framework Desktop via the VPN hostname `strixhalo`
(`antirez@strixhalo`).  This host validates the ROCm backend; do not use it as
a substitute for CUDA or Metal release testing.

- Fetch or push the exact release commit to the Strix Halo machine.
- Build:
  `make clean && make strix-halo`.
- Require the ROCm build to complete without compiler warnings.
- Use the q2 Flash imatrix GGUF for release smoke tests:
  `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`.
- Do not use the mixed q2-q4 or Q4 Flash GGUFs for routine Strix Halo QA yet.
  They are dangerous on this machine for now because the ROCm path can hit
  system OOM instead of failing cleanly.
- Run a short CLI prompt:
  `./ds4 -m gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf --ctx 4096 --nothink -p "Reply with exactly: OK"`.
- For DeepSeek Flash decode, confirm the default path uses prequantized Q8
  activations. Repeat the same greedy run with
  `DS4_ROCM_DSV4_PREQUANT_DECODE=0` only as a diagnostic control. The default
  must be materially faster and must still pass the continuation-quality gate.
  GLM and `--quality` must stay on the full-FP32 activation path.
- Test DSpark with the matched 0731 target and support files:
  `DS4_BIN=./ds4 DS4_DSPARK_MODEL=gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf DS4_DSPARK_SUPPORT=gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf DS4_DSPARK_FIXTURE_TOKENS=64 sh tests/dspark_acceptance_fixture.sh`.
  Require proposals, accepted draft tokens, at least one direct state commit,
  zero verifier errors, and zero replay fallbacks. Record ordinary and DSpark
  generation speed separately. When direct verifier-state handling changes,
  also compare with a test-only build of its immediate replay predecessor; the
  direct build must be faster. DSpark is not currently expected to beat
  ordinary ROCm decode, so do not describe it as a ROCm speedup without a new
  measurement.
- Run one longer prompt if ROCm kernels, backend hooks, tensor loading, model
  cache, KV cache, or graph prefill code changed.
- Run the GLM Q2 release model through ROCm SSD streaming with at least four
  generated tokens:
  `./ds4 --rocm -m gguf/GLM-5.2-UD-Q2_K_RoutedQ2K.gguf --ssd-streaming --ctx 4096 --nothink --tokens 4 -p "Reply with exactly: OK"`.
  Startup must select a cache budget that passes the memory guard without an
  override, and both compact indexed prefill and decode must complete.
- Run one longer GLM prompt with the release-advertised Strix context after
  changes to GLM attention, typed quantized projections, streaming expert
  caches, or memory budgeting. Record the context, cache split, and whether
  the continuation stays free of token-corruption markers.
- Run the same GLM model with `--glm-mtp-timing --temp 0`. At least one draft
  verification cycle must complete without a `glm mtp step failed` message.
- Record startup memory/cache messages, prefill speed, generation speed, and
  whether the backend reports `ROCm backend initialized`.

## 10. Distributed Inference

Distributed code has regressed around route setup, KV snapshots, request IDs,
and split model loading.  Test it whenever distributed, KV, session, or model
loading code changes.

- Prefer `mac-m5max-it` and `mac-m5max-us` for Metal distributed tests.  Use the
  TB5 point-to-point link when it is working; otherwise note that the run used
  WiFi/VPN routing.
- Start workers first, then the coordinator.
- Test a small prompt and a longer prompt.
- Verify the coordinator waits for a complete route and exits cleanly.
- Verify `Ctrl+C` returns control after the current distributed token or chunk
  drains.
- Save and restore a distributed KV snapshot if that code changed.
- If CUDA distributed is relevant, test across the CUDA hosts and record
  generation speed, not just "it works".

## 11. Disk KV Cache

Disk KV cache bugs are high impact for server users.

- Start the server with:
  `./ds4-server --ctx 100000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192`.
- Run the same request twice and verify the second request hits cache.
- Fill the cache enough to trigger eviction; verify the newly-written entry is
  not evicted and useful anchors are retained.
- Test rejection of incompatible checkpoints when model, quantization, context,
  or raw/compressed KV layout changes.
- Test stripped agent sessions: `/strip <id>` then `/switch <id>` should rebuild
  by prefill and render sane history.

## 12. Server APIs

The server must keep compatibility across OpenAI, Responses, and Anthropic
clients.

- `GET /v1/models/deepseek-v4-flash` and `GET /v1/models/deepseek-v4-pro`
  should both serve whichever GGUF is loaded.
- Test OpenAI chat completion, OpenAI Responses, and Anthropic messages.
- Test SSE streaming with thinking enabled and disabled.
- Test keepalive during long prefill and confirm clients do not time out.
- In batched mode, close clients while their requests are queued, prefilling,
  and streaming decode. Repeat across OpenAI chat, Responses, Anthropic, and
  completions. Abandoned work must stop at the next backend-safe boundary, and
  a valid request after each cancellation must complete normally.
- Only after receiving explicit permission for this QA pass, start
  `ds4-server` on the eight-L40S CUDA TP target with the release TP options and
  verify all 16 100k-context sessions allocate. Startup must report a
  2048-token prefill cap; a silent fallback to 4096 is an OOM regression.
- Test `--trace` and confirm rendered prompts, cache decisions, generated text,
  and tool-parser events are useful without leaking unrelated state.

## 13. ds4-agent

The agent is the most stateful component.  Test it manually, not only by build.

- Startup banner, status bar, help, `/power`, `/save`, `/list`, `/switch`,
  `/history`, `/compact`, `/new`, `/del`, and `/strip`.
- Ctrl+C during generation, during prefill, during a web fetch, and during a
  long tool call.  After `Stopped by user`, typing a new prompt must work.
- Queue messages while the model is busy.  Queued messages must not skip tool
  execution; after tool results, the queued user text must be provided.
- Read/search/edit/write tools:
  create a temp project and ask for edits. By default, verify that exact old/new
  replacements work and the tool prompt does not advertise `[upto]`. In a
  separate `--edit-upto` run, verify anchored edits fail safely on ambiguous
  matches and do not require retyping whole files.
- Real coding edit loop:
  delete `/tmp/mymandel`, ask ds4-agent to create a small C ASCII Mandelbrot
  program there, build and run it, then in a second user turn ask for a small
  modification that should naturally use the edit tool, such as changing the
  ASCII character ramp or output dimensions.  Verify the agent edits the
  existing file instead of rewriting the whole project, and that the final
  program still builds and runs.
- Bash tools:
  test short output, large output truncation, non-zero exit output, long-running
  jobs, `bash_status`, and `bash_stop`.
- Web tools:
  `google_search` and `visit_page` should ask for visible Chrome approval with
  timeout, open pages without stealing focus when possible, extract Markdown,
  close tabs, and handle consent/privacy walls as tool errors the model can see.
- TUI:
  test multiline prompt editing, history navigation, queued prompt display,
  status bar fill to terminal width, syntax highlighting in Markdown/code blocks,
  and SSH/remote terminal flicker.

## 14. Download Script And Model Files

- Test `download_model.sh` in a temporary directory so local weights are not
  overwritten.
- Test one Flash target and one PRO target enough to verify URL, resume, Hugging
  Face CLI/curl behavior, file naming, and symlink policy.
- Verify legacy removed targets fail clearly.
- Verify README model names match the script and Hugging Face repository.

## 15. Performance And Power

- Run `ds4-bench` on the release machine and compare with tracked CSV baselines.
- Test `--power 100` is not throttled.
- Test `--power 50` visibly reduces duty cycle in CLI, server, agent, eval, and
  bench where practical.
- Confirm context buffer size, raw KV rows, compressed KV rows, and mmap behavior
  match expectations for 32k, 100k, and any release-advertised context size.

## 16. Speed Regression

Performance is a release gate. A correct result that is unexpectedly much
slower still needs an explanation before release.

Use the same commit, GGUF checksum, prompt, context frontier, generated-token
count, power setting, and backend flags as the reference run. Let the machine
become idle, discard the first warm-up run, then record the median of three
runs. Do not compare different model checkpoints or quantizations. For batched
tests, record aggregate and per-session decode speed.

- A slowdown over 5% requires a clean rerun and investigation.
- A repeatable slowdown over 10% in prefill, decode, or aggregate batched
  decode is a release blocker unless the change and tradeoff are documented.
- Keep the complete `ds4-bench` CSV. A single short-prompt average is not enough
  to detect a context-dependent regression.
- Compare startup time and peak memory as well as tokens per second when model
  loading, caches, streaming, or temporary arenas changed.
- Run the backend-specific batch tests in sections 4 and 8. Fast single-session
  decode does not substitute for aggregate multi-session throughput.

These are the last known good observations available when this gate was added.
They are reference points for matching hardware and workloads, not performance
claims across different models or contexts.

| System and backend | Model and workload | Prefill | Decode |
| --- | --- | ---: | ---: |
| MacBook Pro M3 Max 128 GB, Metal | Flash q2, 11,709-token prompt | 250.11 t/s | 21.47 t/s |
| MacBook Pro M5 Max 128 GB, Metal | Flash q2, 11,707-token prompt | 463.44 t/s | 25.90 t/s |
| Mac Studio M3 Ultra 512 GB, Metal | Flash q2, 11,709-token prompt | 468.03 t/s | 27.39 t/s |
| Mac Studio M3 Ultra 512 GB, Metal | Flash q4, 12,018-token prompt | 448.82 t/s | 26.62 t/s |
| Two M5 Max 128 GB Macs, Metal TP over TB5 RDMA | GLM 5.2 IQ2_XXS, 4,096-token context | about 94 t/s | 15.4 t/s |
| DGX Spark GB10, CUDA | Flash q2, 7,047-token prompt | 343.81 t/s | 13.75 t/s |
| DGX Spark GB10, CUDA | Flash q2 DSpark, 64-token C fixture | - | 24.48 t/s direct; 13.93 t/s replay predecessor |
| Strix Halo gfx1151, ROCm | Flash IQ2 resident, short section 9 smoke | - | 17.27 t/s; FP32 rollback 9.70 t/s |
| Strix Halo gfx1151, ROCm | Flash IQ2 resident, 4,096-token context | - | 14.82 t/s; FP32 rollback 8.76 t/s |
| Strix Halo gfx1151, ROCm | Flash IQ2 DSpark, 64-token C fixture | - | 11.40 t/s direct; 9.77 t/s replay predecessor; 16.70 t/s ordinary |
| 8x L40S, CUDA TP | Flash q4, 2,048-token prefill benchmark | 1,524.84 t/s | 46.93 t/s |
| 8x L40S, CUDA TP | Flash q4, 16-row decode oracle | - | 126.0 aggregate t/s |

The 8x L40S values are retained from the last recorded run on `192.168.60.250`.
They are historical references only: never connect to that host or interrupt
its production server without explicit permission for the current QA pass. If
permission is granted, the existing hard floor remains 110 aggregate t/s for
the 16-row decode oracle.

## 17. Release Sign-off

Do not sign off until:

- macOS Metal Flash passed.
- GLM 5.2 Metal, official-quality, MTP, batching-fallback, and applicable TP or
  CUDA gates passed.
- Official continuation quality gates passed for every released model family.
- CUDA was tested on the CUDA machine or the release notes explicitly say CUDA
  was not validated.
- ROCm was tested on Strix Halo or the release notes explicitly say ROCm was
  not validated.
- Metal, CUDA, ROCm, CPU-only, and test builds completed without compiler
  warnings on every release target that was validated.
- Disk KV cache was exercised.
- Server API streaming was exercised.
- Agent interruption and tool loops were exercised manually.
- The speed-regression gate passed on every validated backend, with any skipped
  baseline or intentional slowdown documented.
- Metal 2/4/8/16-session exactness and forced fallback gates passed.
- Physical Metal TP batching and CUDA native decode/mixed batching passed when
  those backends are part of the release.
- Any skipped item is written down with the reason.
