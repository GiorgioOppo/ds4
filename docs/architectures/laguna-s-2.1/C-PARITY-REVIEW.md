**English** | [Italiano](C-PARITY-REVIEW.it.md)

# Laguna S 2.1 — C-parity review (branch `laguna-s2.1` @ `448d569`)

Line-by-line review of the Swift port against the reference C engine
(`antirez/ds4`, feature branch `laguna-s2.1`, head `448d569` — the same
baseline pinned in [`UPSTREAM-SYNC.md`](../../UPSTREAM-SYNC.md)). Six axes
were compared (geometry/config, tokenizer, chat/tool protocol, tensor
schema/catalog, attention path, MoE/FFN/dequant) and every claimed divergence
was independently re-verified against the code on both sides before being
acted on.

## Verified identical (no action)

- **Geometry and configuration** — all 29 validated `laguna.*` GGUF keys,
  expected values, required types, the 48/72 head alternation
  (`il % 4 == 0`), YaRN requirement and the relative 1e-6 float tolerance
  (`config_validate_laguna_model` ↔ `LagunaConfiguration`). Only the order
  and classification of diagnostics differ; no file is accepted or rejected
  differently.
- **Tokenizer core** — the LF-run pre-split plus the GLM4-shape segment with
  single-digit groups is identical character for character, including every
  Unicode class table; byte-level BPE, special-token loading and the
  EOS/`</assistant>` stop policy match exactly.
- **Tensor schema, quant layouts, catalog** — tensor names, shapes and
  dimension order, the layout markers (Q8_0 signal vs legacy Q4_K/F16), the
  per-group allowed types including the legacy Q6_K down exception, and the
  three catalog artifacts (repos, revision pin, file names, sizes) are all
  value-identical. Swift is stricter on diagnostics only (duplicate tensors,
  `partialOutputHead`, per-tensor missing errors).
- **Attention path** — full YaRN formulas (corr-dims, ramp with integer
  division, mscale ≈ 1.34657 applied to cos/sin of Q and K on full-attention
  layers), NeoX rotation over the 64/128-dim prefix, per-head Q/K RMS-norm
  before RoPE at eps 1e-6, softplus gate (guard at 20) applied per head after
  normalization and before the output projection, softmax scale 1/√128
  without mscale, the 512-key sliding window including the current position,
  F16 ring stores with RNE — operation-identical on the oracle, the engine
  and the kernels. `metal/laguna/laguna.metal` is byte-identical to upstream
  apart from a documented local prelude (`block_q6_K`). Laguna has **no**
  attention sink upstream; neither side implements one.
- **MoE/FFN/routing/dequant** — sigmoid routing with bias-only selection,
  top-10 bitonic sort with lowest-id tie-break, normalization over the
  unbiased probabilities with the 2⁻¹⁴ clamp, ×2.5 scale, SwiGLU without
  clamp with the route weight applied to the mid vector before the down
  projection, unweighted shared expert, `add3` residual order, Q8_0 embedding
  and LM head, and byte-exact Q8_0/Q2_K/Q3_K/Q4_K block layouts (84/110/144/34
  bytes, scale/min packing verified bit by bit).

## Divergences found and fixed in this pass

| # | Severity | Divergence | Fix |
|---|----------|------------|-----|
| F3 | high | Think-mode history: the server renders `<assistant><think>reasoning</think>content` (empty reasoning included, reasoning dropped in nothink); the Swift renderer emitted a bare `</think>` and had no reasoning field | `LagunaChatMessage` (the `chat_msg` shape: separate reasoning + raw tool text) with server-exact rendering; `ChatTurn` mapping splits embedded think prefixes like `split_reasoning_content` |
| F4 | high | Tool-call arguments rendered in alphabetical order; the reference uses the schema's property *declaration* order, then unmapped keys in original JSON order, with non-string values as minified raw JSON | Ordered raw-JSON scanning (`parse_schema_properties`/`json_args_parse`/`json_minify_raw_value` ports) in `LagunaToolCodec.renderToolCalls` |
| F5 | medium | Parsers did not decode the renderer's XML entities (`dsml_unescape_text`), so render→parse round trips were not stable | `dsmlUnescape` applied to parsed keys and values in both parsers |
| — | — | No port of the reference server parser (lenient: last-`</think>` scoping, any tool name, duplicate keys, trailing text dropped, all-string C-spaced arguments, `raw_tool_text` capture) | `LagunaToolCodec.parseServer`, exact port of `parse_glm_generated_message_ex`; `parseStrict` keeps the agent-grade validations on the same grammar |
| — | — | Streaming parser was accumulate-and-finish only; the reference agent errors on completed malformation immediately and exposes calls as they complete | `LagunaIncrementalToolParser` now classifies incomplete vs malformed per chunk and exposes `completedCalls` (plus a chunk-boundary fix: end-of-buffer on a tag edge now waits instead of failing) |
| F6 | medium | Sampling defaults (0.7/20/0.95/0.05) were defined and tested but wired to no inference path | The demo Laguna branch now samples with the family defaults (env overrides win, `DS4_DEMO_TEMPERATURE=0` for greedy parity runs) |
| F1 | medium | The upstream rendered-chat scanner keeps five cross-family literals active for Laguna (`[gMASK]`, `<｜begin▁of▁sentence｜>`, `<｜end▁of▁sentence｜>`, `<｜Assistant｜>`, `<|assistant|>`); Swift only scanned the seven native ones | Literals added to the scanner (and to the neutralization set, so untrusted content cannot inject them) |
| F2 | low | `encodeChatPrompt` rendered the server template and scanned it (EOS id first, default system prompt); the reference CLI path pushes the BOS id and has no default system | `encodeChatPrompt` now mirrors `encode_chat_prompt`/`laguna_chat_append_wrapped` exactly; the demo passes the Poolside default system like `ds4_cli.c` does |
| — | low | No think-mode-aware stop policy (`ds4_token_is_stop_for_think_mode`) | `LagunaTokenizer.isStopToken(_:reasoning:)`, used by the demo |
| — | — | No port of the live tool tail and the malformed-tool-call recovery suffix (`render_laguna_live_tool_tail`, `build_invalid_laguna_tool_error_suffix`) | `LagunaChatRenderer.liveToolTail` and `invalidToolCallRecoverySuffix` (pure helpers; server-loop wiring is still open, see below) |
| F7 | low | The CPU oracle's sliding-window spec pinned the ring at 512 rows without the upstream `min(512, n_ctx)` clamp (the engine already clamped) | Clamp added to `LagunaAttentionSpec.slidingWindow` |
| F8 | low | Stale engine header claimed the mixed Q2_K/Q3_K file is refused (it is accepted since the K-quant matvecs were wired) | Comment corrected |
| — | low | ASCII vs Unicode whitespace in the renderer/parsers (C uses `isspace`) | ASCII whitespace helpers used throughout the Laguna backend |

Additionally, `LagunaRuntimeGate` now honours `DS4_LAGUNA_RUNTIME=1` as a
per-process bring-up override; the committed default stays off until the
logits-parity gate passes.

## Deliberate deviations kept (documented, not bugs)

- **Untrusted-content neutralization on by default** (U+2060 inside control
  literals). The C reference renders user/tool content verbatim, so a literal
  `</assistant>` in user text becomes a real control token there. Byte parity
  with C is available via `neutralizeUntrustedContent: false`.
- **`parseStrict` extras** — undeclared tools/arguments, duplicate keys,
  identifier charset, required arguments and schema-typed values are
  validations the C server does not perform; `parseServer` provides the exact
  reference behavior.
- **Deterministic `<available_tools>` re-encoding** (sorted keys) versus the
  C server's verbatim client JSON: the port has no HTTP client JSON to
  replay, and a stable prompt prefix is worth more than emulating bytes it
  never receives.

## Still open (needs a Mac and/or upstream-shaped work)

- The Gap 4 hardware steps: first compile, kernel parity tests, end-to-end
  logits parity vs `ds4 --temp 0`, batched prefill and the flash split-K
  decode dispatch (`kernel_laguna_flash_attn_reduce_gate_f32` is ported but
  not yet dispatched), the legacy F16/Q6_K recipe, DFlash, catalog digests.
- Server-level Laguna behaviors found by the completeness sweep and not yet
  declared anywhere: the model-alias system (`laguna-s-2.1-chat/-nothink/
  -reasoner` think-mode mapping and the `/v1/models` listing), wiring the
  malformed-tool-call recovery suffix into the server loop, the Laguna
  session-cache suffixes (`</assistant>\n` checkpoints + live tool tail), and
  the upstream `laguna-openrouter-100` QA fixture flow.
