**English** | [Italiano](README.it.md)

# Laguna S 2.1 backend — staged frontend, decoder pending

Laguna S 2.1 is Poolside's GQA + MoE model, supported natively by the
reference C engine on the `laguna-s2.1` branch of `antirez/ds4`. In this port
the family is **recognized and fully validated, but not runnable by
default**: the complete frontend is implemented and unit-tested, the
first-cut resident Metal engine is written (pending its on-hardware parity
gate), and inference stays refused behind `LagunaRuntimeGate` —
`DS4_LAGUNA_RUNTIME=1` opts a single process in for bring-up runs.

The whole frontend has been re-reviewed line by line against the reference C
at head `448d569`, with every divergence fixed or documented as a deliberate
deviation: see [`C-PARITY-REVIEW.md`](C-PARITY-REVIEW.md).

## Geometry (exact `DS4_SHAPE_LAGUNA_S21`)

48 blocks · 3072 embedding · 100352 vocab · GQA with 8 KV heads, head-dim 128
· per-layer query heads: 48 on every fourth block (full attention, 64 YaRN
RoPE dims, base 500000, scale 32 over an 8192 original context), 72 elsewhere
(512-token sliding window, 128 RoPE dims, base 10000) · gated attention with
per-head Q/K RMS-norm · one leading dense block (FFN 12288) · 256 routed
experts, top-10, gating function 2, scale 2.5, expert FFN 1024, plus one
shared expert · RMS epsilon 1e-6 · shipped context 262144.

## Already in place (unit-tested, no model files required)

- `laguna` identification and family detection with a dedicated
  not-implemented error (`ModelArchitectureID.laguna`);
- exact geometry/metadata validation including the 48/72 head-count
  alternation and the YaRN requirement (`LagunaConfiguration`);
- BPE tokenizer with the Laguna pre-split (LF runs first, then GLM4-shape
  groups with single digits) and the family's control tokens
  (`LagunaTokenizer`);
- native chat template: `〈|EOS|〉` opener, `<system>`/`<user>`/
  `<tool_response>` text tags, `<assistant>…</assistant>` turns with
  server-exact `<think>` reasoning framing (`LagunaChatMessage` carries the
  reasoning and raw tool text separately, like upstream `chat_msg`), plus the
  live tool tail and the malformed-tool-call recovery suffix
  (`LagunaChatRenderer`);
- tagged tool calls (`<tool_call>name<arg_key>…<arg_value>…`) with schema
  declaration-order arguments and XML-entity decoding: the exact
  reference-server parser (`parseServer`), the stricter agent-grade parser
  (`parseStrict`), and a chunk-safe incremental parser
  (`LagunaToolCodec`);
- reference sampling defaults: temperature 0.7, top-k 20, top-p 0.95,
  min-p 0.05;
- tensor schema of the published recipes — Q8_0 signal path, legacy Q4_K/F16,
  mixed RoutedQ2_K/Last27Q3_K (`LagunaTensorSchema`);
- download catalog: official Poolside Q4_K_M (revision-pinned), the mixed
  requant, and the DFlash Q8_0 draft as an accessory.

## Not yet implemented

- the Metal decoder (`metal/laguna.metal` + driver) and its logits-parity
  gate — the turnkey plan is Gap 4 in
  [`../../PORTING-GAPS.md`](../../PORTING-GAPS.md);
- DFlash speculative decoding (after the decoder);
- pinned byte counts/SHA-256 for the catalog artifacts (required before the
  entries may become `runnable`);
- SSD streaming, distributed inference and tensor parallelism (upstream also
  requires full residency for Laguna).
