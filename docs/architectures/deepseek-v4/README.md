# DeepSeek V4 backend

DeepSeek V4 is the reference operational backend. Flash and Pro share the same
concrete Metal decoder, but each instance receives the validated geometry of
its own GGUF. The multi-model restructuring preserves its hot path and makes
the parts that belong to the family explicit.

## Family-specific components

- `deepseek4.*` metadata and Flash/Pro validation;
- DeepSeek tokenizer and control tokens;
- conversational template and DSML tool calling;
- Hyper-Connection, NSA, compressors and top-6 MoE routing;
- expert cache and streaming, side bundle and MTP;
- Metal decoder, KV payload and numerical diagnostics;
- distributed pipeline with 43/256 geometry for Flash or 61/384 for Pro;
  protocol, slices and masks are tested for both profiles.

## Profile status

- **Flash**: local inference, demo, GUI and distribution supported for the
  three single-file quantizations in the catalog.
- **Pro Q2**: the single GGUF is downloadable, selectable and locally
  runnable from GUI and demo. `DSV4RuntimeGeometry` carries into the decoder
  the 61 layers, 7168 channels, 128 heads, 384 experts, top-1024 indexer,
  router scale 2.5 and the profile's per-layer compression ratios.
- **Pro Q4 split**: the two shards are downloadable as a package but remain
  `downloadOnly`; the local loader does not combine them into a single model.
- **Distributed Pro**: the complete Q2 GGUF is accepted by pipeline and
  expert parallelism. Protocol and geometry are covered by tests; the
  multi-Mac numerical and performance validation with the real file remains
  to be done.

The router uses 256 lanes for Flash and a 512-lane bitonic network for Pro; in
the latter case indices 384...511 are padded to `-inf`, while top-6
normalization and weight scaling receive the values of the active geometry.

The `DS4_*` variables remain valid for compatibility. Controls specific to
experts, NSA, Q4 and streaming must be exposed by the GUI only when the
backend declares the corresponding capability.

## Regression

Every structural change must keep the existing tokenizer, DSML, shape,
weights, prefill, decode, KV, MoE and distribution tests passing. These tests
remain explicitly DeepSeek: renaming them "generic" would hide the
assumptions they must protect.
