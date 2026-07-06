# DwarfStar

The SwiftUI macOS app for Apple Silicon. It is driven by `DS4Engine`; a sidebar
selects the main panels. Model path and context length are configured once in
**Settings** through `AppSettings`, then inherited by every controller. Every
persisted setting (UserDefaults key, default value, engine environment
variable) is documented in the root
[Configuration Reference](../../README.md#configuration-reference).

The app is organized by **feature**, with one folder per tab or area:

- **`App/`**: entry point, shared settings, root view, and environment helpers.
- **`Chat/`**: streaming chat with Markdown, reasoning, live tool calls,
  attachments, and the `ChatStore` view model.
- **`Models/`**: GGUF selection, scanning, and downloads.
- **`Project/`**: project library, with sandbox-bookmarked folders indexed for
  agent tools.
- **`Tuning/`**: expert-cache slots, hit-rate, routing concentration, and agent
  editor.
- **`Server/`**: native in-process HTTP server compatible with OpenAI and
  Anthropic-style endpoints, exposing the shared Settings-loaded engine.
- **`Distributed/`**: UI for worker/coordinator distributed inference.
- **`Bench/`**: local or distributed prefill/generation benchmarks over growing
  context sizes.
- **`Diagnostics/`**: token and chat-template dumps.
- **`Settings/`**: global model, context, execution mode, and memory/I/O
  settings such as expert bundle, dense streaming, Q4 dense cache, disk KV, and
  raw-KV ring.
- **`Support/`**: cross-cutting utilities such as engine logs and process streams.
- **`Assets.xcassets/`**: app icon assets.
