# Sources

The source tree is organized first by **Swift target**, then by feature or
technical responsibility. Target boundaries are enforced by SwiftPM; folders
inside a target are organizational and do not create additional Swift modules.

The exact SwiftPM target graph is below. An arrow `A -> B` means **B imports and
depends directly on A**; transitive availability is not a license to skip the
declared boundary.

```text
DS4Core  -> DS4Metal
DS4Core  -> DS4Engine       DS4Metal -> DS4Engine
DS4Core  -> DwarfStar       DS4Engine -> DwarfStar
DS4Core  -> DS4Demo         DS4Metal -> DS4Demo

DS4Core + DS4Metal + DS4Engine -> DS4CoreTests
```

Consequently, `DS4Core` has no project-target dependency; `DS4Metal` can only
build upward from `DS4Core`; and `DS4Engine` is the first layer allowed to join
portable data with the Metal backend and application policy. `DwarfStar`
imports `DS4Engine` and `DS4Core` directly but not `DS4Metal`. `DS4Demo`
deliberately bypasses `DS4Engine` and drives `DS4Core` + `DS4Metal` for bring-up
and performance diagnostics. `DS4CoreTests` is the shared test target and may
import all three libraries.

| Target | Type | Direct target dependencies | Responsibility |
|---|---|---|---|
| `DS4Core/` | library | none | Pure Swift formats, architecture inspection, portable contracts and backend-owned tokenizer/conversation frontends. No Metal or SwiftUI. |
| `DS4Metal/` | library | `DS4Core` | Shared Metal runtime and kernels plus concrete, physically separated model decoders and GPU state. |
| `DS4Engine/` | library | `DS4Core`, `DS4Metal` | Backend selection and application orchestration: inference service, DTOs, tools, persistence, downloads and distribution. |
| `DwarfStar/` | executable | `DS4Engine`, `DS4Core` | Native SwiftUI application and its app-facing HTTP server, grouped by user-visible feature. |
| `DS4Demo/` | executable | `DS4Core`, `DS4Metal` | Small CLI that drives the core and GPU runtime directly for audit, diagnostics and generation. |

## Directory Map

```text
DS4Core/
  Conversation/
    Models/             chat turns, tool calls and shared conversation data
    Backends/
      DeepSeekV4/DSML/  DeepSeek template, markup and tool-call parser
      Qwen/             documented placeholder, not implemented
  Diagnostics/          load/progress reporting without UI dependencies
  Formats/
    GGUF/                GGUF types, cursor and mapped model
    KVCheckpoint/        on-disk KV checkpoint format
    Quantization/        numeric conversion and CPU quantization helpers
  Generation/           sampling
  Model/
    Common/             architecture id, family, descriptor and capabilities
    Backends/
      DeepSeekV4/       Flash/Pro shape and metadata validation
      Qwen/             documented placeholder, not implemented
  Storage/              SSD/cache planning and memory-lock simulation
  Tokenization/
    API/                minimal architecture-neutral tokenizer contract
    Common/             byte-level helpers
    Backends/
      DeepSeekV4/       concrete tokenizer, special tokens and thinking mode
      Qwen/             documented placeholder, not implemented

DS4Metal/
  Runtime/
    Core/                Metal device, pipelines and GPU tensor primitives
    Generated/           generated embedded Metal source; never hand-edit
  Model/Quantization/   architecture-independent quantization descriptors
  Backends/
    Common/             high-level boundary; never dispatches inside a layer
    DeepSeekV4/
      Architecture/     Flash dimensions and RoPE parameters
      Weights/          GGUF-backed layer weights and providers
      Experts/ MTP/     expert bundle, MetalIO and optional MTP sidecar
      Streaming/        dense-weight streaming and requant caches
      Decode/           Execution, Generation, Attention, Cache, KV, Prefill,
                        State, Diagnostics and Reference
    Qwen/               documented placeholder, no decoder or fake kernels
  Graph/
    Core/                graph context
    Operations/          focused graph stages (attention, MoE, RoPE, output…)
  Kernels/
    Attention/ Compression/ Dense/ MoE/ Tensor/
                        Swift dispatch wrappers grouped by operation

DS4Engine/
  Runtime/
    Common/             model inspection, descriptor, capabilities and selector
    Backends/
      DeepSeekV4/       registration of the operational concrete backend
      Qwen/             recognized but deliberately unavailable backend
  Inference/
    API/                 public request/result/event data structures
    Service/             inference actor and conversation/generation extensions
    Benchmark/           benchmark operations
    Diagnostics/         tokenizer/template diagnostics
    Subagents/           isolated sub-agent execution
    Tuning/              runtime tuning operations
  Distributed/
    Protocol/            wire types grouped by framing, handshake, files,
                         KV, work, experts, codec and serialization
    Coordinator/         connection, file, KV, chat and benchmark orchestration
    Worker/              worker state plus Assignments, Files, KV, Lifecycle,
                         Serving and Concurrency subdomains
    Transport/           network transport
    Execution/           per-node model execution
    Files/               model distribution
  Tools/
    Core/                registry and common tool contracts
    Builtins/            one implementation per built-in tool, grouped by area
    Integrations/        shared Git/GitHub/web clients
    MCP/                 MCP configuration, protocol and transports
  Agents/                agent profiles
  ModelManagement/       model download and expert-bundle operations
  Persistence/           disk-backed application state and KV
  Projects/              project indexing/cache

DwarfStar/
  App/                   application entry point, environment and root view
  Features/
    Chat/                 Models, Persistence, ViewModels and Views
    Server/               API adapters, networking, concurrency, services, UI
    Benchmark/            Controllers and Views
    Diagnostics/          Controllers and Views
    Distributed/          Controllers and Views
    ModelManagement/      Models, Services and Views
    Project/ Settings/ Tuning/
                         feature-specific views and state
  Shared/Support/         GUI helpers shared by multiple features
  Assets.xcassets/        app assets (Xcode build only)

DS4Demo/
  Command/               executable entry point and argument handling
  Diagnostics/           logging, disk benchmark and model audit helpers
```

## Placement Rules

- Put reusable, hardware-independent data in `DS4Core`; do not import Metal,
  Network or SwiftUI there. Keep architecture identity common but tokenizer,
  chat format and shape in the backend that owns them.
- Put GPU-backed state and operations in `DS4Metal`. Model-visible application
  policy belongs in `DS4Engine`, not in a kernel wrapper. Never add model-family
  conditionals to the per-layer hot loop; select a concrete backend first.
- Put public inference request/result/event types in `DS4Engine/Inference/API`.
  Keep service behavior in a focused `InferenceService+Area.swift` extension.
- Put distributed wire data under `DS4Engine/Distributed/Protocol`, grouped by
  protocol concern. Transport and coordinator/worker behavior stay outside the
  protocol folder.
- Put UI-only models, persistence adapters, controllers/view models and views
  inside the owning `DwarfStar/Features/<Feature>` subtree. Use
  `DwarfStar/Shared` only when more than one feature owns the dependency.
- Keep generated files under a `Generated/` directory and document their
  generator. `DS4Metal/Runtime/Generated/KernelSources.swift` is generated from
  `metal/*.metal` by `make embed-kernels`; edit the `.metal` files instead.
- Prefer one primary type or one cohesive extension per file. Name extensions
  as `Type+Responsibility.swift` and keep DTOs separate from I/O or execution.

See [`../docs/STRUTTURA-PROGETTO.md`](../docs/STRUTTURA-PROGETTO.md) for the
dependency rules, contribution workflow and build commands.
See [`../docs/ARCHITETTURE-SUPPORTATE.md`](../docs/ARCHITETTURE-SUPPORTATE.md)
for the support matrix and the checklist for adding Qwen.

## Local Documentation Rule

Every directory in `Sources` has a `README.md`. Read the target README first,
then the nearest local README before changing a type. Local files document
ownership, dependencies, main files and extension/testing rules; cross-cutting
behavior belongs under [`../docs/`](../docs/README.md).

All Markdown files colocated with target sources are discovered and excluded by
the helper in `Package.swift`; `project.yml` applies the equivalent recursive
exclusion for XcodeGen. Adding a guide next to code therefore does not copy it
into the executable or produce unhandled-resource warnings.
