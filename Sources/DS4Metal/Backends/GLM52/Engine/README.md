# GLM 5.2 engine

`GLM52ResidentModel` is the real engine over the real GGUF: it validates the
schema, builds the weight map and loads the resident decode graph — attention,
dense/shared FFN and the output head uploaded once per layer, routed experts
streamed per token through `GLM52StreamedExpertProvider` (planner + LRU slot
cache, byte-exact record slicing) — then drives token-by-token prefill and
greedy decode over the growing resident caches. Embedding rows are read
per token straight from the Q8_0 `token_embd` tensor via the reader's
bounded ranged read.

Deliberate scope limits:

- `layerCount` may truncate the stack from the front. The three leading
  dense layers are Q8_0 end to end and run against the published file today;
  sparse layers additionally need a routed-expert type with a validated
  kernel — the published IQ2_XXS routed experts are REFUSED at load
  (`GLM52StreamedExpertProviderError.unsupportedExpertType`) until that
  kernel tranche lands, because silently wrong output is worse than an
  error.
- This is roadmap wiring, not enablement: `BackendSelector` still refuses
  `glm-dsa`, and the catalog stays `downloadOnly`, until the full-model
  real-GGUF logits parity gate passes.

`GLM52GreedyDecoding` keeps argmax (lowest-index ties) and the generation
loop free of Metal so both are unit-tested without a device.
