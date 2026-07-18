# DS4Engine

`DS4Engine` is the application layer between the portable data of `DS4Core`,
the GPU backend of `DS4Metal` and clients such as the GUI. The target contains
no views: it coordinates inference, tools, agents, persistence, models and
distributed nodes.

## Dependencies

```text
DS4Core ──┐
          ├── DS4Engine ──> GUI / application services
DS4Metal ─┘
```

- `DS4Core`: GGUF, tokenizer, conversation, sampling and shared formats.
- `DS4Metal`: Metal runtime, decoder, caches and layer execution.
- Foundation, Network, CryptoKit and Security are used only in the areas that
  need them.

## Folders

- [`Runtime`](Runtime/README.md): GGUF inspection, backend selection and the
  capabilities actually exposed to services and the GUI.
- [`Inference`](Inference/README.md): API and the actor that owns the decoder.
- [`Distributed`](Distributed/README.md): protocol, transport, coordinator and worker.
- [`Tools`](Tools/README.md): function calling, built-in tools and MCP.
- [`Persistence`](Persistence/README.md): checkpoints and persistent caches.
- [`ModelManagement`](ModelManagement/README.md): model download and sidecar.
- [`Projects`](Projects/README.md): secure index of imported projects.
- [`Agents`](Agents/README.md): agent profiles and registry.

## Architectural rules

1. Public inference types go in `Inference/API`, not in the GUI.
2. The decoder's mutable state stays isolated from `InferenceService`.
3. Data transmitted over the network lives in `Distributed/Protocol`;
   coordinator and worker consume its types but do not define the format.
4. A tool declares its contract and execution via `ToolRegistry`; reusable
   integrations must not be duplicated in individual built-ins.
5. Persistence must not hold complete snapshots in RAM when it can process
   them in streaming.
6. Extensions of a main type follow `Tipo+Responsabilita.swift`.
7. Every GGUF goes through `RuntimeBackendFactory` before any specific
   tokenizer, configuration or decoder; recognizing a family does not mean
   implementing it.

## Verifying changes

After a change to the target, run at least `swift build --disable-sandbox` and
the tests in `Tests/DS4CoreTests/Engine`. Protocol changes also require
encode/decode tests and a bump of `Dist.protocolVersion` if they are not
compatible with existing nodes.
