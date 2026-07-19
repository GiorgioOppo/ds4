**English** | [Italiano](README.it.md)

# Runtime/Common

- `ModelInspector` converts GGUF metadata into DS4Core's neutral descriptors
  and the capabilities actually implemented by DS4Engine.
- `BackendSelector` distinguishes an available backend, a recognized but
  unimplemented family and an unknown architecture.
- `RuntimeBackendFactory` is the boundary to call before `ModelConfig`,
  tokenizer or architecture-specific allocations.
- `BackendCapabilities` drives GUI, diagnostics and optional services. It must
  not be used to describe theoretical properties not yet supported by the
  runtime.

The types in this folder do not run inference and are not in the per-token
path.
