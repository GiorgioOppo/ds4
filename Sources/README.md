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
| `DS4Core/` | library | none | Pure Swift data formats, tokenizer, sampler, model metadata, conversation models and DSML rendering. No Metal or SwiftUI. |
| `DS4Metal/` | library | `DS4Core` | Metal runtime, GPU tensors, kernel dispatch, decode/prefill graph, model weights, expert streaming and caches. |
| `DS4Engine/` | library | `DS4Core`, `DS4Metal` | Application orchestration: inference service, public DTOs, tools, agents, persistence, downloads and distributed inference. |
| `DwarfStar/` | executable | `DS4Engine`, `DS4Core` | Native SwiftUI application and its app-facing HTTP server, grouped by user-visible feature. |
| `DS4Demo/` | executable | `DS4Core`, `DS4Metal` | Small CLI that drives the core and GPU runtime directly for audit, diagnostics and generation. |

## Directory Map

```text
DS4Core/
  Conversation/
    Models/             chat turns, tool calls and shared conversation data
    DSML/               chat-template renderer, markup and tool-call parser
  Diagnostics/          load/progress reporting without UI dependencies
  Formats/
    GGUF/                GGUF types, cursor and mapped model
    KVCheckpoint/        on-disk KV checkpoint format
    Quantization/        numeric conversion and CPU quantization helpers
  Generation/           sampling
  Model/                hardware-independent model shape
  Storage/              SSD/cache planning and memory-lock simulation
  Tokenization/         tokenizer, byte-level helpers and thinking mode

DS4Metal/
  Runtime/
    Core/                Metal device, pipelines and GPU tensor primitives
    Generated/           generated embedded Metal source; never hand-edit
  Model/
    Architecture/        DeepSeek-V4 dimensions and RoPE parameters
    Weights/             GGUF-backed layer weights and providers
    Quantization/        GPU/model quantization descriptors
    Experts/             expert bundle and MetalIO safety state
    Streaming/           dense-weight streaming
  Decode/
    Execution/           decoder state, forward pass and per-layer execution
    Generation/          output-head and generation operations
    Attention/           indexer/attention selection support
    Cache/               expert cache and usage statistics
    KV/                  KV snapshots
    Prefill/             prefill stages, orchestration and I/O gathering
    State/               transient decode state
    Diagnostics/         decode profiling
    Reference/           parity/reference decoder
  Graph/
    Core/                graph context
    Operations/          focused graph stages (attention, MoE, RoPE, output…)
  Kernels/
    Attention/ Compression/ Dense/ MoE/ Tensor/
                        Swift dispatch wrappers grouped by operation

DS4Engine/
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
  Network or SwiftUI there.
- Put GPU-backed state and operations in `DS4Metal`. Model-visible application
  policy belongs in `DS4Engine`, not in a kernel wrapper.
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

## Local Documentation Rule

Every directory in `Sources` has a `README.md`. Read the target README first,
then the nearest local README before changing a type. Local files document
ownership, dependencies, main files and extension/testing rules; cross-cutting
behavior belongs under [`../docs/`](../docs/README.md).

All Markdown files colocated with target sources are discovered and excluded by
the helper in `Package.swift`; `project.yml` applies the equivalent recursive
exclusion for XcodeGen. Adding a guide next to code therefore does not copy it
into the executable or produce unhandled-resource warnings.
