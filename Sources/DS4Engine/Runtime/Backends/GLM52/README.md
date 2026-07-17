# GLM 5.2 runtime registration

This directory owns the engine-facing registration for GGUF models declaring
`general.architecture = "glm-dsa"`.

`GLM52BackendDefinition` currently advertises portable model capabilities only.
Its runtime capability set intentionally remains empty, and backend selection
returns `backendNotImplemented`, until the resident Metal graph, tokenizer and
quality fixtures pass together. This boundary prevents a GLM GGUF from being
parsed or executed as DeepSeek V4.

The numerical implementation belongs in a sibling
`Sources/DS4Metal/Backends/GLM52` tree; architecture metadata and tensor schemas
belong in DS4Core. Do not add GLM conditionals to the DeepSeek hot loop.
