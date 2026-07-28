**English** | [Italiano](STRUTTURA-PROGETTO.it.md)

# Project structure

This guide describes where to place new code and which dependencies are
allowed. The structure separates pure data, GPU execution, orchestration,
app features and the CLI, so that every change has a clear owner.

## Modules and dependencies

```text
DS4Metal  = DS4Core
DS4Engine = DS4Core + DS4Metal
DwarfStar = DS4Core + DS4Engine
DS4Demo   = DS4Core + DS4Metal
```

In particular:

| Target | Dependencies | Contents |
|---|---|---|
| `DS4Core` | no internal ones | Portable formats and contracts, architecture detection, sampling and the Core components of the backends. |
| `DS4Metal` | `DS4Core` | Shared Metal runtime and kernels; decoder, weights and GPU state kept separate per backend. |
| `DS4Engine` | `DS4Core`, `DS4Metal` | Backend selection, inference service, tools, persistence, downloads and distribution. |
| `DwarfStar` | `DS4Engine`, `DS4Core` | SwiftUI interface, feature state and the HTTP server exposed by the app. |
| `DS4Demo` | `DS4Core`, `DS4Metal` | Diagnostic CLI that uses the engine directly, without the application layer. |

Subfolders are not Swift modules. They exist to make code ownership
visible; the real boundaries are the targets declared in
`Package.swift`.

## Responsibility map

### DS4Core: portable contracts and backend frontends

- `Model/Common`: `general.architecture` identifier, family, descriptor,
  capabilities and detection with no Metal dependencies.
- `Model/Backends/DeepSeekV4`: shape, Flash/Pro profiles, defaults and
  validation of `deepseek4.*` metadata.
- `Model/Backends/Qwen`: documented extension point; no Qwen shape is
  implemented yet.
- `Conversation/Models`: shared turns, specs and tool calls.
- `Conversation/Backends/DeepSeekV4`: renderer, DSML markup and call parsing
  specific to the DeepSeek template.
- `Tokenization/Common`: reusable byte-level primitives.
- `Tokenization/Backends/DeepSeekV4`: DeepSeek tokenizer, special tokens and
  thinking mode; the Qwen path stays separate and non-operational.
- `Formats/GGUF`: GGUF types, binary cursor and memory-mapped model.
- `Formats/KVCheckpoint`: shared persistent wrapper; payloads belong to the
  individual backends.
- `Formats/Quantization`: numeric conversions and CPU quantization.
- `Generation`: sampling and token-choice policies.
- `Storage` and `Diagnostics`: cache/SSD planning and load progress.

Reusable structures with no Metal, network or UI dependencies must live here.
A type does not become common just because two models share a concept with the
same name: layout, special tokens and semantics must be genuinely compatible.

### DS4Metal: shared runtime and concrete GPU backends

- `Runtime/Core`: device, command queue, pipelines and `GPUTensor`.
- `Runtime/Generated`: embedded and generated Metal sources.
- `Kernels/<Area>`: Swift wrappers for reusable operations, grouped by
  attention, compression, dense, MoE and tensors.
- `Graph/Core` and `Graph/Operations`: graph infrastructure currently shared
  with the DeepSeek backend; it does not host architecture selection.
- `Model/Quantization`: shared GPU quantization descriptors.
- `Backends/Common`: high-level execution boundary and rules that
  prevent dynamic dispatch inside the per-layer loop.
- `Backends/DeepSeekV4/Architecture`: Flash dimensions and RoPE parameters.
- `Backends/DeepSeekV4/Weights`, `Streaming`, `Experts`, `MTP`: GGUF mapping,
  dense and expert weights, sidecars and DeepSeek-specific streaming.
- `Backends/DeepSeekV4/Decode`: execution, generation, attention, cache,
  KV, prefill, diagnostics, reference and state of the concrete decoder.
- `Backends/GLM52`: tensor schema, DSA/IndexShare references, the streaming
  engine (`GLM52ResidentModel` + `GLM52ChainedDecode`) and the per-family
  Metal kernels; runnable end to end (chat, demo, server, benchmark).
- `Backends/Laguna`: tensor schema of the published S 2.1 recipes and the
  runtime gate (off); the decoder is not ported yet.
- `Backends/Qwen`: documented placeholder; no fake kernels or decoder.

A data structure holding Metal buffers or resources belongs in this
target. Architecture choice happens before entering the hot path:
do not add `if qwen` to `StreamingDecoder` or its layers. An application-level
choice about how to use the model belongs in `DS4Engine` instead.

### DS4Engine: API and application features

- `Runtime/Common`: model inspection, backend selection, descriptor and
  capabilities consumed by clients.
- `Runtime/Backends/DeepSeekV4`: construction and settings of the operational
  backend without touching the decoder's hot path.
- `Runtime/Backends/Laguna`: capability registration behind the Laguna
  runtime gate; selection refuses the family until the decoder lands.
- `Runtime/Backends/Qwen`: explicit error and extension point, not a
  simulated implementation.
- `Inference/API`: public DTOs for requests, events, results and benchmarks.
- `Inference/Service`: main actor and its extensions for conversation,
  generation and agents.
- `Inference/Benchmark`, `Diagnostics`, `Subagents`, `Tuning`: separate
  features that compose the service.
- `Distributed/Protocol`: network protocol data split by framing,
  handshake, files, KV, work, experts, codecs and serialization.
- `Distributed/Coordinator`: central state and separate extensions for
  connections, files, KV, chat, expert parallelism and benchmarks.
- `Distributed/Worker`: node state and extensions grouped into
  `Assignments`, `Files`, `KV`, `Lifecycle`, `Serving` and `Concurrency`.
- `Distributed/Transport`, `Execution`, `Files`: networking, per-node engine
  and model distribution, kept separate from the wire messages.
- `Tools/Core`: registry and shared contracts; `Tools/Builtins`: one tool per
  file; `Tools/Integrations` and `Tools/MCP`: external clients and protocols.
- `Agents`, `ModelManagement`, `Persistence`, `Projects`: application services
  with autonomous responsibilities.

Types transmitted over the network must not depend on the coordinator or the
worker. The transport must not define message semantics. Service extensions
follow the `InferenceService+Responsibility.swift` naming.
`InferenceService` remains the public façade: it inspects and selects the
backend once, then holds the concrete decoder. Demo, GUI and diagnostics must
not duplicate heuristics based on the file name.

### DwarfStar: interface features

Every visible area lives in `Features/<Feature>`:

- `Chat`: `Models`, `Persistence`, `ViewModels`, `Views`;
- `Server`: `API`, `Networking`, `Concurrency`, `Services`, `Controllers`,
  `Views`;
- `Benchmark`, `Diagnostics`, `Distributed`: separate controllers and views;
- `ModelManagement`: models, services and views;
- `Project`, `Settings`, `Tuning`: feature-specific content.

`App` contains only bootstrap, environment, global settings and root
navigation. `Shared/Support` is reserved for helpers genuinely shared across
multiple features; it must not become a catch-all folder.

### DS4Demo: CLI and diagnostics

`Command/main.swift` handles arguments and the execution loop. Logging, model
audit and disk benchmarks live in `Diagnostics`, so the CLI does not
duplicate engine logic.

### Tests: structure mirroring the domains

The SwiftPM target stays single (`DS4CoreTests`), but files are grouped by
module and responsibility:

```text
Tests/DS4CoreTests/
  Core/                 shared contracts and Core backend regressions
  Metal/                shared runtime/kernels and GPU backend regressions
  Engine/               backend selection, inference and application services
```

Subfolders are discovered recursively by both SwiftPM and XcodeGen.
A new test goes next to the domain of the code under test; the target name
does not imply it must concern only `DS4Core`.

## Rules for new files

1. Place a data structure next to the domain that owns it, not next to
   its first caller. Public DTOs go in `Inference/API`; wire messages
   in `Distributed/Protocol`; UI-only models in the corresponding feature.
2. Separate data definition, serialization, I/O and execution when
   they change for different reasons.
3. Prefer one main type or one cohesive extension per file. To extend
   a type use `Type+Feature.swift`.
4. Keep dependencies pointing downward: Core knows nothing about Metal,
   Engine or UI; Metal knows nothing about Engine or UI; Engine knows
   nothing about SwiftUI.
5. Architecture inspection and description live in the common layer; shape,
   tokenizer, templates, tensors, decoder and KV payloads live in the backend.
6. A reusable Metal wrapper goes in `Kernels/<Area>`; the source executed
   by the GPU goes in `metal/*.metal`. An architecture-specific graph goes in
   `Backends/<Architecture>`, not in the shared runtime.
7. A new GUI feature gets its own folder, with subfolders created
   only when distinct responsibilities exist (models, services, controllers,
   views or persistence).
8. Never edit generated files by hand. Always document the command that
   regenerates them.
9. Every new source, test or operational folder must contain a
   `README.md` with purpose, dependencies, main files and verification rules.
   Cross-cutting details go in a thematic document under `docs/`.

Markdown files placed inside a target are excluded automatically by the
`markdownFiles` helper in `Package.swift`; XcodeGen uses the recursive
exclusion `**/*.md`. Do not maintain manual README lists in the manifest.

## Generated kernels

The `metal/*.metal` files are the source of truth. The command:

```sh
make embed-kernels
```

generates
`Sources/DS4Metal/Runtime/Generated/KernelSources.swift`, embedded in the
SwiftPM binaries and in the app. Every kernel change must therefore follow
this flow:

1. edit the `.metal` file;
2. update the wrapper in `Sources/DS4Metal/Kernels/<Area>` if the
   signature changes;
3. run `make embed-kernels`;
4. build and run the tests.

## Build, tests and Xcode project

From the repository root:

```sh
# Debug build of all targets
swift build --disable-sandbox

# Test suite
swift test --disable-sandbox

# Release build of the demo
swift build -c release --product DS4Demo --disable-sandbox

# Regenerate the project after adding, removing or moving files
xcodegen generate

# Launch the CLI or the GUI through SwiftPM
swift run DS4Demo
swift run DwarfStar
```

On macOS, if the active toolchain does not point to the full Xcode
installation, prefix with:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift build --disable-sandbox
```

After a reorganization verify both SwiftPM and the regenerated project:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcodebuild build \
  -project DwarfStar.xcodeproj \
  -scheme DwarfStar \
  -destination 'platform=macOS' \
  -derivedDataPath /tmp/DwarfStarDerivedData \
  CODE_SIGNING_ALLOWED=NO
```

`xcodegen generate` uses `project.yml` as the source of truth: do not add
files manually to the `.pbxproj`.
