# DwarfStar

The SwiftUI macOS app for Apple Silicon. It is driven by `DS4Engine`; a sidebar
selects the main panels. Model path and context length are configured once in
**Settings** through `AppSettings`, then inherited by every controller. Every
persisted setting (UserDefaults key, default value, engine environment
variable) is documented in the root
[Configuration Reference](../../README.md#configuration-reference).

The app is organized by **feature**, with one folder per tab or area under
`Features/`:

- **`App/`**: entry point, shared settings, root view, and environment helpers.
- **`Features/Chat/`**: models, persistence, view model and views for streaming
  chat with Markdown, reasoning, live tool calls,
  attachments, and the `ChatStore` view model.
- **`Features/ModelManagement/`**: catalog-backed GGUF discovery, validated
  manual selection, resumable downloads and progress UI. Flash and the
  single-file Pro Q2 are runnable; the two-file Pro Q4 package and the three
  GLM 5.2 variants are download-only.
- **`Features/Project/`**: project library, with sandbox-bookmarked folders indexed for
  agent tools.
- **`Features/Tuning/`**: expert-cache slots, mixed-quant pool policy, hit-rate,
  routing concentration, and agent editor.
- **`Features/Server/`**: API adapters, networking, concurrency and UI for the
  native in-process HTTP server compatible with OpenAI and
  Anthropic-style endpoints, exposing the shared Settings-loaded engine.
- **`Features/Distributed/`**: UI for worker/coordinator distributed inference.
- **`Features/Benchmark/`**: local or distributed prefill/generation benchmarks over growing
  context sizes.
- **`Features/Diagnostics/`**: token and chat-template dumps.
- **`Features/Settings/`**: global model, context, execution mode, and memory/I/O
  settings such as layer-aware mixed-quant expert cache, expert bundle, dense
  streaming, Q4 dense cache, disk KV, and raw-KV ring.
- **`Shared/Support/`**: cross-cutting utilities such as engine logs and process streams.
- **`Assets.xcassets/`**: app icon assets.

## Dependency and state flow

`DwarfStarApp` creates `AppSettings`, `ChatStore`, and MCP state. `RootView`
passes those shared objects to feature controllers and views. Chat owns the one
local `InferenceService`; Server and local Benchmark borrow that instance and
serialize work instead of loading duplicate weights.

Application code may adapt APIs from `DS4Engine` and `DS4Core`. Reusable model,
inference, protocol, storage, and tool behavior belongs in those modules, not in
the SwiftUI target.

Downloaded GGUFs live in `~/Library/Application Support/DwarfStar/models/`.
The app renders the catalog from `DS4Engine`; it does not duplicate remote
filenames, SHA-256 values or runtime support decisions.

## Documentation map

- [`App/README.md`](App/README.md): startup and shared settings.
- [`Features/README.md`](Features/README.md): feature index and boundaries.
- [`Features/Chat/FLOW.md`](Features/Chat/FLOW.md): message, tool, persistence,
  and shared-engine flow.
- [`Features/ModelManagement/README.md`](Features/ModelManagement/README.md):
  GUI catalog, reuse, resume, selection and runtime boundary.
- [`Features/Server/HTTP-API.md`](Features/Server/HTTP-API.md): endpoints and
  request lifecycle.
- [`Shared/README.md`](Shared/README.md): cross-feature support rules.

Every source directory has a local README. Update the nearest README when files
move or ownership changes; do not document environment defaults in multiple
places when the root Configuration Reference is authoritative.
