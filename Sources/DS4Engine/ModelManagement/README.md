# ModelManagement

Gathers model operations that are not part of the inference cycle.

## Components

- [`Catalog`](Catalog/README.md): cross-family registry, DeepSeek V4 and
  GLM 5.2 catalogs, artifacts and availability in the current runtime.
- [`Download`](Download/README.md): credentials, resumable GGUF download and
  state consumed by the GUI.
- `ExpertBundleTool.swift`: verifies or builds the expert sidecar without
  loading the full decoder.
- `ModelFileDiagnostics.swift`: pre-flight of the model path — explains the
  real cause of a failed open (missing file, orphaned `.part` to resume, file
  in the legacy Application Support invisible to the sandbox) with the remedy
  in the message; used by `InferenceService` before opening the GGUF.

The operating procedure is described in
[`GESTIONE-MODELLI.md`](GESTIONE-MODELLI.md).

## Dependencies and flow

The catalog is the single source and does not depend on the GUI. The
downloader uses Foundation/CryptoKit; the token store uses Security. Bundle
construction uses `DS4Core` metadata and `DS4Metal` logic. The result is then
consumed by [`Inference`](../Inference/README.md) or
[`Distributed`](../Distributed/README.md).

## Extension

A new transformation must produce a deterministic, verifiable artifact kept
separate from the original GGUF. Do not store secrets in UserDefaults or logs
and do not replace a final file before download/verification are complete.

Being in the catalog means being acquirable, not necessarily runnable. The
three DeepSeek V4 Flash entries and the single-GGUF Pro Q2 are `runnable`;
the Pro Q4 remains `downloadOnly` because it is a multi-shard package. The
three monolithic GLM 5.2 GGUFs are also `downloadOnly` until a verified
`glm-dsa` backend exists.
