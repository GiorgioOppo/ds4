**English** | [Italiano](GUI-SERVER-E-API.it.md)

# GUI, local server, and API

The SwiftUI GUI and the HTTP server are two entry points into the same engine.
Both must go through `InferenceService`; no feature may create a second copy of
the model to serve a request.

## App structure

`Sources/DwarfStar/App` contains only bootstrap, environment, global settings,
and navigation. Behavior is organized under `Sources/DwarfStar/Features`:

| Feature | Responsibility |
|---|---|
| `Chat` | sessions, rendering, attachments, tool loop, and generation |
| `Settings` | model, architecture inspection, capability-gated controls, MCP, and distribution |
| `ModelManagement` | GGUF catalog and downloads |
| `Project` | selection and inspection of the active project |
| `Tuning` | expert cache and profiles, only for backends with expert routing |
| `Server` | HTTP listener and OpenAI/Anthropic adapters |
| `Distributed` | coordinator, workers, and vertical benchmark |
| `Benchmark` | shared engine throughput |
| `Diagnostics` | tokenizer, templates, and operational status |

`Shared/Support` hosts only helpers used by multiple features. A helper
specific to the chat or the server must stay in its owning feature.

## Shared dependencies

`AppEnvironment` builds and distributes the long-lived objects. The most
important is `InferenceService`, an actor that serializes access to the decoder
and to KV state. Chat, server, and benchmark may present different interfaces,
but they do not run concurrent inference on the same decoder.

This constraint protects:

- Metal buffers and the expert cache;
- KV position and NSA recurrent state;
- resident memory on systems with little RAM;
- the ordering of generation events.

## ChatStore

`ChatStore` is the chat's `@MainActor` view model. The main file contains
observable state and initialization; extensions separate:

- model lifecycle;
- generation and cancellation;
- persistent sessions;
- attachments;
- agents and tool loop;
- benchmark and tuning;
- application of performance settings.

The view model converts service events into presentation models. It does not
implement tokenizers, samplers, or wire parsing. Views must not call the Metal
backend directly.

Before load, `ChatStore` uses `InferenceService`'s metadata-only inspection to
obtain architecture and capabilities without allocating Metal. Settings, Chat,
and Tuning therefore show only the features declared by the backend: a Qwen
model that is recognized but not implemented keeps its path and context, but
does not receive DeepSeek controls for expert cache, NSA, Q4, bundles, or
MetalIO.

## Model download and selection

The **Download…** button in Settings opens `DownloadView`. The view does not
own a copy of the remote names: it renders `ModelCatalogRegistry.entries`,
defined in `DS4Engine`, and uses `DownloadRunner` only as a `@MainActor`
adapter between downloader events and SwiftUI state.

The main catalog presents eight logical choices coming from two Hugging Face
repositories:

| Edition | Variants | Download | Select/run |
|---|---|---:|---:|
| DeepSeek V4 Flash | Q2, mixed Q2/Q4, Q4 | yes | yes |
| DeepSeek V4 Pro | single Q2 | yes | yes |
| DeepSeek V4 Pro | two-shard Q4 split | yes | no |
| GLM 5.2 | monolithic IQ2_XXS, Q2_K, Q4_K | yes | no |

For each entry the GUI shows approximate size, local status, free space,
aggregate progress, and runtime availability. A complete Flash entry or a
single Pro Q2 can be selected; when the download finishes it becomes the
active model and the choice persists without a security-scoped bookmark,
because the file belongs to the app container. Pro Q4 split and all GLM
entries remain `downloadOnly`: the selection button does not appear and the
download does not change the active model. MTP is an accessory and does not
appear in the main catalog.

### Paths and reuse

The writable destination is:

```text
~/Library/Application Support/DwarfStar/models/
```

Before touching the network, the runner looks for the catalog's exact filename
in the destination, in the development root and its `gguf/` subfolder, and in
the directory of the currently configured model. A regular, non-empty final
file is reused in place and is not downloaded again; when the catalog knows
the exact size, as for GLM, the byte count must also match. The model menu's
automatic scans include only the three Flash variants and Pro Q2 declared
selectable by the catalog. Complete GLM files remain visible in the download
sheet, not in the load menu.

An incomplete transfer remains as `<name>.part`; **Resume** uses HTTP Range.
The UI allows cancellation and keeps the partial file. The downloader checks
disk space, validates the size returned by the server, verifies the SHA-256
digest pinned in the catalog, and publishes the final filename only after
verification.

### Manual browse

**Browse** remains the advanced path for external GGUFs. `ModelPicker`
acquires security-scoped access, runs `InferenceService.inspectModel`, and
passes the descriptor to `BackendSelector` before saving the choice. If the
profile is not runnable by the current build, it shows an error and keeps the
previous active model. The bookmark is used only for external files; choosing
an app-managed model removes any old bookmark, preventing it from taking
precedence again at relaunch.

The optional Hugging Face token is configured in Settings, lives in the
Keychain, and is passed explicitly to the downloader. The panel shows only the
active source or the masked form, never the full secret.

## Local server

`LocalServer` uses `Network.framework` and exposes the already-loaded engine.
Responsibilities are split into:

- `Networking` — accept, HTTP reading, JSON, and responses;
- `API` — adapters for the various contracts;
- `Concurrency` — request limiting and serialization;
- `Services` — shared listener state;
- `Controllers` and `Views` — control from the GUI.

Implemented endpoints:

| Method | Path | Compatibility |
|---|---|---|
| `GET` | `/v1/models`, `/v1/models/{id}` | OpenAI |
| `POST` | `/v1/chat/completions` | OpenAI Chat Completions |
| `POST` | `/v1/responses` | OpenAI Responses |
| `POST` | `/v1/completions` | OpenAI legacy |
| `POST` | `/v1/messages` | Anthropic Messages |

The adapters translate the request into shared inference types and convert the
events back into JSON or SSE. They must not duplicate the generation loop.

## SSE streaming

For a streaming response the server:

1. validates method, path, authentication, and body limit;
2. builds the normalized request;
3. acquires the engine gate;
4. sends `text/event-stream` headers;
5. translates each service event into the chosen API's format;
6. sends the terminator required by the protocol;
7. releases the gate even on cancellation or error.

Client disconnection must propagate cancellation to the generation task,
without leaving a partially committed turn as valid KV.

## Parameters and precedence

The server has its own settings for host, port, API key, CORS, and default
token limit. Sampling parameters and limits present in the request body
override the server defaults for that turn. The requested model name is a
compatibility identifier: the server always uses the single GGUF already
loaded.

The complete table of accepted fields is in the
[Configuration Reference](../README.md#http-server-server-tab).

## Security

The listener uses plain HTTP. The default `127.0.0.1` restricts access to the
local machine; a LAN bind must be protected externally with TLS or a tunnel.
The application API key prevents accidental requests but, without TLS, does
not protect the token from interception.

The body is size-limited and the parser rejects malformed requests. CORS is
disabled by default. See [CRITTOGRAFIA.md](CRITTOGRAFIA.md).

## Adding a GUI feature

1. Create `Features/<Name>` with models, controllers/view models, services,
   and views only when they are genuinely needed.
2. Add a `README.md` declaring ownership and dependencies.
3. Expose an application-level API from the engine in `DS4Engine`; do not
   import `DS4Metal` in the GUI.
4. Register the feature in the root navigation.
5. Persist only settings that must survive a relaunch.
6. Test the pure parts in the test target and manually verify the SwiftUI
   lifecycle that depends on the system.

## Adding or changing an endpoint

1. Keep generic HTTP parsing in `Networking`.
2. Add the contract mapping under `API`.
3. Normalize toward the types in `DS4Engine/Inference/API`.
4. Cover stream and non-stream, errors, and cancellation.
5. Do not introduce a parallel decoder or sampling queue.
6. Update this guide, the server README, and the examples in the main README.

## Key files

- `Sources/DwarfStar/App/AppEnvironment.swift`
- `Sources/DwarfStar/Features/Chat/ViewModels/ChatStore.swift`
- `Sources/DwarfStar/Features/ModelManagement/Views/DownloadView.swift`
- `Sources/DwarfStar/Features/ModelManagement/Services/DownloadRunner.swift`
- `Sources/DS4Engine/ModelManagement/Catalog/ModelCatalog.swift`
- `Sources/DS4Engine/ModelManagement/Download/ModelDownloader.swift`
- `Sources/DwarfStar/Features/Server/Services/LocalServer.swift`
- `Sources/DwarfStar/Features/Server/Concurrency/RequestGate.swift`
- `Sources/DS4Engine/Inference/Service/InferenceService.swift`

For the internal flow see [PIPELINE-INFERENZA.md](PIPELINE-INFERENZA.md).
