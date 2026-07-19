**English** | [Italiano](README.it.md)

# DeepSeekV4/Architecture

Geometry and derived dimensions of the DeepSeek-V4 profiles served by the
Metal kernels. The static Flash profile remains available for source
compatibility, but model-aware construction uses the instance-based geometry.

## Main files

- [`DSV4Shape.swift`](DSV4Shape.swift): layers, heads, experts, compression
  ratios and legacy Flash constants.
- [`DSV4RuntimeGeometry.swift`](DSV4RuntimeGeometry.swift): instance-based
  runtime geometry derived from the Flash/Pro profile or the GGUF
  configuration.
- [`DSV4Dims.swift`](DSV4Dims.swift): derived runtime dimensions and flags for
  configurable fusions/kernels.
- [`RopeParams.swift`](RopeParams.swift): parameters for RoPE and context scaling.

## Flow and dependencies

The loader validates the GGUF metadata in `DeepSeekV4Configuration`; from this
a `DSV4RuntimeGeometry` is built. Decoder, scratch, graph and kernel wrappers
can thus share the selected profile's dimensions, compression and RoPE.
`DSV4Shape` keeps the previous Flash API for tests and legacy callers; it must
not be used to size a Pro GGUF. The local Pro Q2 profile uses 61 layers, 384
experts and its own compression ratios through the same runtime geometry.

## Change rules

Do not fix an incompatible shape with silent fallbacks. Distinguish model
constants from hardware tuning. Every new field must state its unit, origin
and consumers; update validation and allocation tests together.
