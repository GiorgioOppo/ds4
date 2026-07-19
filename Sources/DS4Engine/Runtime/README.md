**English** | [Italiano](README.it.md)

# Multi-backend runtime

This layer separates the portable inspection of a GGUF from the construction
of the specific decoder. `RuntimeBackendFactory` first reads
`general.architecture`, produces a `RuntimeModelDescriptor` and only then
enables a concrete backend.

The current numerical path remains DeepSeek V4 and keeps using
`StreamingDecoder` directly: the Runtime layer introduces no dynamic dispatch
in the generation loop. Flash and single-file Pro Q2 are selected locally and
build a distinct immutable geometry; the Pro Q4 split package remains
download-only. Distributed Pro is under verification and is not part of the
declared local support. Qwen is recognized to allow clear messages and a
capability-driven UI. GLM 5.2 already has a native detector and frontend, but
construction is still rejected until the Metal decoder passes the end-to-end
numerical gates.

The historical public APIs of `InferenceService` and the `DS4_*` variables
remain compatible.
