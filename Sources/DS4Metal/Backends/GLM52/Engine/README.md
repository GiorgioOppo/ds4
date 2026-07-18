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
  routed experts need a type with a validated kernel — Q8_0, the four
  K-quants and IQ2_XXS (the published routed format) are supported; any
  other type is refused at load
  (`GLM52StreamedExpertProviderError.unsupportedExpertType`), because
  silently wrong output is worse than an error.
- This is roadmap wiring, not enablement: `BackendSelector` still refuses
  `glm-dsa`, and the catalog stays `downloadOnly`, until the full-model
  real-GGUF logits parity gate passes.

`GLM52GreedyDecoding` keeps argmax (lowest-index ties) and the generation
loop free of Metal so both are unit-tested without a device.

# SSD streaming

`GLM52LayerStreamer` is the GLM analog of the DeepSeek StreamingDecoder:
sparse layers past `residentLayerCount` keep only their small state resident
(norms, router rows, indexer proj, decode caches — ~12 MiB/layer) while the
big Q8_0 tensors are pread DIRECTLY into two reusable staging buffer sets,
with the next layer's fill overlapping the current layer's GPU compute.
Streaming is a memory strategy, never a numeric one: the streamed path runs
the exact same resident-graph functions on the staged buffers.

Knobs (demo: DS4_GLM_RESIDENT_LAYERS / DS4_GLM_ACTIVE_EXPERTS /
DS4_GLM_EXPERT_SLOTS): resident-layer budget, routed-expert cap (rank-order
truncation — less I/O, lower quality), expert slot-cache size. After every
token the engine warms each sparse layer's slot cache with that token's
selected experts (`GLM52StreamedExpertProvider.prefetch`, serialized against
the decode thread). Honest arithmetic: a fully streamed 78-layer pass reads
~36 GiB/token — the resident budget, the expert-record coalescing already
done by `read(plan:)`, and the future MTLIO/bundle paths are what make the
16-32 GiB machines viable.

MetalIO (`DS4_GLM_MTLIO=1`): the streamer fills its staging slots through an
`MTLIOCommandQueue` (SSD → MTLBuffer, no CPU pread copy), with a load-time
warm-up probe and permanent per-run fallback to pread on any anomaly — the
same discipline as the DeepSeek ExpertBundle backend.

Expert BUNDLES are a deliberate non-goal for now: repacking gate|up|down
records contiguously would duplicate ~190 GiB on disk to turn three adjacent
preads into one, and `read(plan:)` already coalesces each expert's reads
into one packed destination. Revisit only if profiling shows the expert
path seek-bound rather than bandwidth-bound.
