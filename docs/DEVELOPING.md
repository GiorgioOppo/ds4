# Developing

How to extend the project: setup, conventions, recipes for adding
kernels / layers / tests / CLI flags, and the common pitfalls that
have actually bitten us. Open this before submitting a change.

## At a glance

| Task | Section |
|---|---|
| **Engine** | |
| Set up your dev environment | §1 |
| Where things live, naming conventions | §2 |
| Add a new Metal kernel | §3 |
| Add a new Layer (composition) | §4 |
| Add a CLI flag | §5 |
| Support a new weight-name layout | §6 |
| Debug a forward pass | §7 |
| Common pitfalls (real ones we've hit) | §8 |
| **Desktop app** | |
| Add a new `ModelEndpoint` backend (e.g. Anthropic direct) | §9 |
| Add a Settings tab | §10 |
| Add a new MCP transport (HTTP/SSE) | §11 |
| Wire a new InferenceService method through the chat | §12 |
| Add a native tool to `DeepSeekTools` | §13 |

## 1. Dev environment

Required:

- macOS 14+ (Sonoma/Sequoia) with Apple Silicon
- Xcode 15+ command-line tools (`xcode-select --install`)
- Swift 5.10+ (ships with Xcode 15+)
- `git`, `git-lfs` (for downloading reference weights)

Optional but useful:

- VSCode + Swift extension for inline diagnostics
- [Instruments](https://help.apple.com/instruments/) for profiling
- [`huggingface-cli`](https://huggingface.co/docs/huggingface_hub/guides/cli)
  for downloading model checkpoints

Build flags to know:

```bash
swift build                              # debug (slow, useful for assertion checks)
swift build -c release                   # production speed
swift build -c release -v                # verbose; see MetalLibPlugin invocations
swift test -c release                    # run all XCTests
swift test --filter HadamardTests        # run one class
swift test --filter Hadamard/testFWHT    # run one function
```

After cloning, make sure the build-tool plugin's shell helper is
executable:

```bash
chmod +x Plugins/MetalLibPlugin/build_metallib.sh
```

Otherwise SwiftPM will fail with "Permission denied" when it tries to
compile the `.metal` files.

## 2. Code conventions

### Directory layout

| Directory | What goes here |
|---|---|
| `Sources/DeepSeekKit/` | Top-level types: Tensor, Device, Config, IO, helpers |
| `Sources/DeepSeekKit/Encoding/` | Chat encoder + Message types |
| `Sources/DeepSeekKit/Kernels/` | `.metal` shader sources |
| `Sources/DeepSeekKit/Layers/` | Swift wrappers around kernels + composition (MLA, MoE, Block, …) |
| `Sources/deepseek/` | Inference CLI executable |
| `Sources/converter/` | HF → Mac-friendly format CLI executable |
| `Tests/DeepSeekKitTests/` | XCTest target (one file per kernel/module) |
| `Reference/` | Read-only Python source-of-truth (do not modify) |
| `Plugins/MetalLibPlugin/` | SwiftPM build-tool plugin |

### Naming patterns

- **Wrapper types**: `enum Foo { static func apply(...) }` for stateless
  in-place ops; `final class Foo { ... func callAsFunction(...) }` for
  stateful (constructor takes weights/config).
- **Public method names**:
  - `apply(_:in:)` — kernel that modifies its input in place.
  - `callAsFunction(_:in:)` — kernel that takes input and returns a
    fresh tensor.
  - `forward(_:in:)` — same as `callAsFunction` but for clarity in
    deeper composition layers.
  - `referenceCPU(_:...)` — pure-Swift reference for tests.
- **Files**: one type per file, file named after the type
  (`SoftmaxAxis.swift` contains `enum SoftmaxAxis`).
- **Docstring convention**: each file header references the Python
  source-of-truth, e.g.
  ```swift
  /// Walsh-Hadamard transform along the last axis. Mirrors
  /// `rotate_activation` in `Reference/inference/model.py` lines 247–251.
  ```

### Avoid

- Adding new abstractions when an existing type fits.
- Speculative future-proofing. Three similar lines are fine; abstract
  on the third concrete case.
- Comments restating what the code obviously does. Use comments for
  *why* something is done.

## 3. Recipe: add a new Metal kernel

You'll usually add three things in sequence: kernel, wrapper, test.

### Step 1 — Write the kernel

`Sources/DeepSeekKit/Kernels/foo.metal`:

```metal
#include <metal_stdlib>
using namespace metal;

// Brief description of what this computes.
// Mirrors `<function>` in `Reference/inference/<file>.py:<line>` (if any).
//
// Inputs:
//   x: [N, D] f32
//   ...
// Outputs:
//   y: [N, D] f32
//   ...

kernel void foo_f32(
    device const float* x       [[buffer(0)]],
    device float*       y       [[buffer(1)]],
    constant uint2&     dims    [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint n = gid.y, d = gid.x;
    uint N = dims.x, D = dims.y;
    if (n >= N || d >= D) return;
    y[n * D + d] = x[n * D + d] * 2.0f;   // your math here
}
```

Conventions:
- Buffer 0 is the primary input. Add buffers in declaration order.
- Use `uint` (or `uint2`/`uint3`) for all `[[thread_position_*]]`
  attributes. Mixing dimensionalities in the same kernel is a compile
  error — see §8.
- Function constants get global indices across the whole library. Add
  any you use at index 7+ (indices 0-6 are reserved — see
  [KERNELS.md](KERNELS.md#conventions)).

The `MetalLibPlugin` picks up the file automatically on the next
`swift build`. No Package.swift edit needed.

### Step 2 — Write the Swift wrapper

`Sources/DeepSeekKit/Layers/Foo.swift`:

```swift
import Foundation
import Metal

/// Multiplies each element of x by 2. Demonstrative.
public enum Foo {
    private static let pipeline = Device.shared.makePipeline("foo_f32")

    public static func apply(_ x: Tensor, in cmd: MTLCommandBuffer) -> Tensor {
        precondition(x.dtype == .f32 && x.shape.count == 2)
        let N = x.shape[0], D = x.shape[1]
        let y = Tensor.empty(shape: [N, D], dtype: .f32)

        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(x.buffer, offset: x.offset, index: 0)
        enc.setBuffer(y.buffer, offset: 0, index: 1)
        var dims = SIMD2<UInt32>(UInt32(N), UInt32(D))
        enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 2)
        enc.dispatchThreads(MTLSize(width: D, height: N, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
        return y
    }

    /// Pure-Swift reference for testing.
    public static func referenceCPU(_ x: [Float]) -> [Float] {
        x.map { $0 * 2 }
    }
}
```

Conventions:
- `private static let pipeline = Device.shared.makePipeline(...)` —
  one pipeline per type, cached at static init.
- If your kernel uses function constants, use the `makeFunction(name:
  constantValues:)` path (see `Layers/HCSinkhorn.swift` for an example).
- `dispatchThreads` is preferred over `dispatchThreadgroups` unless
  you specifically need fixed threadgroup-position semantics.
- Always end with `enc.endEncoding()`.

### Step 3 — Write the test

`Tests/DeepSeekKitTests/FooTests.swift`:

```swift
import XCTest
import Metal
@testable import DeepSeekKit

final class FooTests: XCTestCase {

    func testMatchesCPU() throws {
        let N = 4, D = 8
        let input = (0..<N*D).map { Float($0) * 0.1 }
        let t = input.withUnsafeBytes {
            Tensor.from(bytes: $0, shape: [N, D], dtype: .f32)
        }
        let cmd = Device.shared.queue.makeCommandBuffer()!
        let y = Foo.apply(t, in: cmd)
        cmd.commit(); cmd.waitUntilCompleted()

        let gpu = y.toFloatArray()
        let cpu = Foo.referenceCPU(input)
        for i in 0..<gpu.count {
            XCTAssertEqual(gpu[i], cpu[i], accuracy: 1e-5)
        }
        _ = input
    }
}
```

See [TESTING.md](TESTING.md) for tolerance choices and patterns.

### Step 4 — Run + iterate

```bash
swift build -c release
swift test -c release --filter FooTests
```

If the build fails on the `.metal`, the error usually points to a
specific line and column. If the test fails, the assertion message
includes the index — print neighbors to diagnose.

## 4. Recipe: add a new Layer (composition)

A Layer that composes existing kernels (no new `.metal`) typically:

1. Declares its weight tensors and helper modules as `let` properties.
2. Constructs `MTLComputePipelineState`s for any kernels it uses
   (often lazy on `Device.shared.makePipeline`).
3. Implements `callAsFunction(_:...)` or similar by chaining
   already-tested kernels.
4. Provides a `referenceCPU(...)` for the smallest interesting input.

Pattern reference: `Sources/DeepSeekKit/Layers/HyperConnections.swift`
is a clean example — it composes `Linear`, `HCSinkhorn`, and two
small composition kernels (`hc_collapse_f32`, `hc_post_compose_f32`)
into `pre` and `post`.

For composition layers, the reference CPU is built from each
sub-Layer's reference. The test then composes those references and
compares to the composed GPU forward.

## 5. Recipe: add a CLI flag

Two files might need touching:

- `Sources/deepseek/main.swift` — inference CLI
- `Sources/converter/main.swift` — converter CLI

Both use a simple `while !args.isEmpty` loop. To add `--my-flag <value>`:

```swift
case "--my-flag":
    guard !args.isEmpty, let v = Int(args.removeFirst()) else { usage() }
    myFlag = v
```

Update the usage string at the top of the file, then update
[USAGE.md](USAGE.md) with the new flag's row in the CLI flag table.

## 6. Recipe: support a new weight-name layout

When the model release changes its tensor naming (HF rename pass, new
checkpoint format), update `Sources/DeepSeekKit/Assembly.swift`:

```swift
// existing:
let embedW = (try loader.tryLoad(["embed.weight", "model.embed.weight"]))
    ?? AssemblyHelpers.randomTensor([config.vocabSize, dim], rng: &rng, scale: 0.02)

// extend with the new name:
let embedW = (try loader.tryLoad([
    "embed.weight",
    "model.embed.weight",
    "model.embed_tokens.weight",        // ← new
])) ?? AssemblyHelpers.randomTensor(...)
```

`WeightLoader.tryLoad(_:[String])` returns the first match. Old names
keep working as fallbacks. The "missing tensor" summary in the load
log tells you which fallbacks fired.

For systematic renames (e.g. a whole subtree moved), add the rename
to `renameKey` in `Sources/converter/main.swift:121`. The converter
applies it once and writes back the canonical form, so the loader's
fallback list stays short.

## 7. Recipe: debug a forward pass

### Print intermediate tensor stats

```swift
import Foundation

func dump(_ t: Tensor, name: String, n: Int = 16) {
    let arr = t.toFloatArray()
    let head = arr.prefix(n).map { String(format: "%+.4f", $0) }.joined(separator: " ")
    let mean = arr.reduce(0, +) / Float(arr.count)
    let absMax = arr.map { abs($0) }.max() ?? 0
    print("\(name) shape=\(t.shape) mean=\(mean) absmax=\(absMax) head=[\(head)]")
}
```

Call `dump(...)` after any kernel output. Forces a CPU/GPU sync (you
need a `cmd.commit(); cmd.waitUntilCompleted()` first), so use
sparingly during normal runs.

### Isolate a single layer

Build a tiny `ModelConfig`, randomly initialize one Block via
`Assembly.swift:randomInit` patterns, run it standalone:

```swift
var cfg = ModelConfig()
cfg.dim = 64
cfg.nLayers = 1
cfg.nRoutedExperts = 4
let m = Transformer.randomInit(config: cfg)
let logits = m.forward(inputIds: [[1, 2, 3]], startPos: 0)
dump(logits, name: "logits")
```

Useful to confirm the forward chain runs end-to-end with deterministic
weights, decoupled from "did the weights load correctly".

### Compare to the Python reference

Tier 3 deferred but very effective: dump activations from
`Reference/inference/generate.py` on a fixed toy prompt, load the
JSON in a Swift test, run the Swift forward on the same input, assert
match within tolerance. See `docs/ROADMAP.md` T3.4 for the plan.

## 8. Common pitfalls (we hit these — you will too)

### 1. Mixed-dimensionality builtin attributes

```
error: Expecting input declarations with either all scalar types or all
vector types with the same number of elements
```

Cause: in one kernel,
```metal
uint2 tg     [[threadgroup_position_in_grid]],
uint  tid    [[thread_position_in_threadgroup]],   // ← uint vs uint2
uint  tgsize [[threads_per_threadgroup]]
```
Fix: promote all of them to the largest size used:
```metal
uint2 tg     [[threadgroup_position_in_grid]],
uint2 tidv   [[thread_position_in_threadgroup]],
uint2 tgsv   [[threads_per_threadgroup]]
```
Then use `tidv.x` and `tgsv.x` in the body. We hit this twice
(`act_quant.metal`, `softmax_axis.metal`).

### 2. Invalid hex literal with `P`

```
error: 'C' is not a valid digit in floating point exponent
```

Cause: `0xDEEPC0DE` — Swift parses `P` as the binary-exponent marker
(`0x1.0p10`) and then expects a decimal digit. `P` isn't a hex digit
either. Use only `0-9A-F`. We hit this with `0xDEEPC0DE` → fixed to
`0xDEADC0DE`.

### 3. `log1p` not in MSL

Metal Shading Language doesn't have `log1p`. Use
`log(1.0f + x)` directly — for the range we care about, it's accurate
enough. We hit this in `moe.metal:score_fn` for `sqrtsoftplus`.

### 4. `Bundle.module` requires resources or a build plugin

```
error: type 'Bundle' has no member 'module'
```

Cause: SwiftPM only synthesizes `Bundle.module` when the target has at
least one declared resource. Either:
- Add `resources: [.process("Kernels")]` to the target (works for
  Xcode-driven builds; `swift build` may not compile `.metal` files).
- Or, our chosen path: a `BuildToolPlugin` (`MetalLibPlugin`) that
  compiles the `.metal` files and emits `default.metallib` as a
  resource. See `Plugins/MetalLibPlugin/`.

### 5. Variable shadowing across function scope

```
error: Invalid redeclaration of 'bits'
```

Cause: two `let bits = ...` declarations in the same function scope,
even with different types. Swift forbids this. Rename one (we used
`outBits` for the second).

### 6. Function constants must be unique library-wide

The Metal Shading Language spec requires function-constant indices to
be globally unique. We had `index(0)` reused across `act_quant.metal`,
`moe.metal`, `hc_sinkhorn.metal` until the linker complained. Fix:
the reserved mapping at the top of [KERNELS.md](KERNELS.md#conventions).
When you add new function constants, claim 7+.

### 7. Forgetting `_ = input` in a test

```swift
let t = input.withUnsafeBytes { Tensor.from(bytes: $0, ...) }
let cmd = ...
Foo.apply(t, in: cmd)
cmd.commit(); cmd.waitUntilCompleted()
// ← without `_ = input`, ARC may have already freed `input` here
// even though Tensor.from copies bytes into a fresh buffer.
```

`Tensor.from(bytes:)` does copy, so this is technically safe — but if
you ever swap to a zero-copy variant the test would silently break.
Pin the source alive with `_ = input` at the end of the test
function.

### 8. Buffer sizes must be page-aligned for `bytesNoCopy`

`device.makeBuffer(bytesNoCopy:length:options:deallocator:)` requires
the length to be a multiple of the page size and the pointer to be
page-aligned. We round up to a `sysconf(_SC_PAGESIZE)` multiple in
`SafeTensors.swift`. If you ever wrap a non-mmap buffer with
`bytesNoCopy`, observe the same constraint.

### 9. CPU-GPU sync overhead

Each `cmd.commit(); cmd.waitUntilCompleted()` is a full sync — costs
tens of microseconds on top of the actual GPU work. Don't sprinkle
them inside hot loops. The decode loop in `Sources/deepseek/main.swift`
syncs once per token (intentionally, since we need the next logits
before sampling).

### 10. Resume detection requires same flags

The converter's resume scan looks for `model-NNNNN-of-MMMMM.safetensors`
files matching the *expected* total shard count. If you re-run with
different `--shard-size-gb` or `--target-dtype`, the totals don't
match and resume bails. Wipe the output directory before changing
flags.

## 9. Recipe: add a new `ModelEndpoint` backend

OpenRouter is the only remote backend today. The dispatch points
that need to grow when a second one lands (Anthropic direct, OpenAI
direct, a local Ollama-style server, …) are:

1. **`ModelEndpoint` case** in `Sources/DeepSeekUI/State/ModelEndpoint.swift`.
   Add a `case` for the new backend with whatever it needs to be
   addressed (`baseURL`, model id, etc. — `apiKey` does NOT live
   here, it lives in Keychain). Update `id`, `displayName`,
   `subtitle`, `iconName`, `isRemote`. Both switches that consume
   the enum will then refuse to compile until you handle the new
   case.

2. **`KeychainStore` account** in
   `Sources/DeepSeekUI/State/KeychainStore.swift`. Add a new
   `KeychainAccount.<backend>APIKey` constant. Decide whether to
   reuse a single account or namespace per-baseURL — for OpenRouter
   we used a single account because the slug carries the model
   identity.

3. **`ModelState.load(_:)` dispatch** in
   `Sources/DeepSeekUI/State/ModelState.swift`. Add a `case .new`
   branch that calls a new private `loadRemote<Backend>` mirroring
   `loadRemoteOpenRouter` — unload the current local model, mark
   `.loading`, validate the key with whatever the backend exposes,
   mark `.loaded` on success / `.error` on failure.

4. **`ChatStore.send` dispatch** in
   `Sources/DeepSeekUI/State/ChatStore.swift`. Add a `case .new`
   branch (right next to the existing `.openRouter` one) that calls
   a new private `sendVia<Backend>`. Most of the existing remote
   plumbing in `runRemoteLoop` — `buildOpenAIMessages`,
   `composeOpenAITools`, the per-iteration finalize logic — is
   reusable if the backend speaks the OpenAI Chat Completions
   shape. If not, write a parallel driver and keep the same
   `phases[id]` lifecycle so the chat view doesn't need to know
   which backend produced the events.

5. **HTTP client** (only if the new backend isn't OpenAI-compatible).
   Mirror `OpenRouterAPI.swift`'s shape: typed DTOs for the
   response payloads (Decodable structs are fine — the request body
   is built as `[String: Any]` so optional fields don't all become
   Optional in the type), one `Client` class with
   `validateKey(_:) async throws`, `fetchModels(apiKey:)` if
   applicable, and an `AsyncThrowingStream<…, Error>` factory for
   the streaming completion.

6. **UI surfaces.** `ContentView.ModelPicker` already shows recents
   by endpoint icon — add a new "Add … model" menu entry next to
   "Add OpenRouter model…", plus a sheet to pick the slug from the
   backend's catalog (mirror `AddOpenRouterModelSheet`).
   `APIKeysSettingsTab` grows a new SecureField + Save/Test/Delete
   row.

The chat view itself should need no changes — it consumes
`GenerationPhase` and `Conversation.cumulativeCostUSD`, both of
which are backend-agnostic. The cost banner reads
`GenerationMetrics.turnCostUSD` which any backend can populate.

## 10. Recipe: add a Settings tab

```swift
struct MyNewSettingsTab: View {
    var body: some View {
        Form { … }
            .formStyle(.grouped)
            .padding(20)
    }
}
```

Then in `Sources/DeepSeekUI/Views/Settings/SettingsScene.swift`:

```swift
MyNewSettingsTab()
    .tabItem { Label("My thing", systemImage: "sf.symbol.name") }
```

If the tab needs to read or write a library (`AgentLibrary`,
`ModelLibrary`, etc.), thread the `@ObservedObject` through
`SettingsScene` and the App scene that owns it — the existing
tabs are the template.

`@AppStorage`-backed settings live in
`Sources/DeepSeekUI/State/AppSettings.swift`; add the key to
`AppSettingsKey` so the storage namespace stays discoverable.

After adding new files, regenerate the Xcode project so the GUI
target compiles:

```bash
./Tools/generate-xcodeproj.sh
```

(`swift build` does not need regeneration.)

## 11. Recipe: add a new MCP transport

Today `MCPClient` is hard-wired to stdio: it spawns a child
`Process` and frames JSON-RPC over its stdin/stdout pipes.
Supporting HTTP/SSE or WebSocket requires extracting the framing
+ I/O behind a protocol.

Suggested shape:

```swift
protocol MCPTransport: AnyObject {
    func start() async throws
    func send(_ frame: Data) throws
    func setIncomingHandler(_ handler: @escaping (Data) -> Void)
    func stop()
}

final class StdioMCPTransport: MCPTransport { … existing stdio code … }
final class HTTPSSETransport: MCPTransport { … URLSession.bytes + POST … }
```

`MCPClient` then takes an `MCPTransport` in init instead of
constructing a `Process` directly. `MCPClientPool.sync(with:)`
inspects the `MCPServerConfig.command` (or a new
`MCPServerConfig.transport` enum) to decide which transport to
construct.

The JSON-RPC layer (request id allocation, pending continuation
map, frame routing) lives above the transport in
`MCPClient.sendRequest` / `handleFrame` and doesn't need to know
which transport carried the bytes.

## 12. Recipe: wire a new InferenceService method through the chat

When you add a method to `InferenceService` (e.g. a new sampling
hook, a new prompt-encoding shortcut, a new diagnostic), the wiring
into the chat looks like:

1. Add the method to `Sources/DeepSeekUI/State/InferenceService.swift`.
   Mark it `func` (not `nonisolated`) and dispatch the body onto
   the internal `q` serial queue — every other method does this so
   the cache + tokenizer aren't touched from two threads.

2. Surface it on `ChatStore` only if the chat surface needs to call
   it. The pattern: a thin wrapper that knows which conversation /
   which mode applies, then forwards to `service`.

3. If the method changes the `phases[id]` lifecycle (e.g.
   introduces a new `GenerationPhase` case), update both branches
   of `ChatStore.send` (local and remote) so the UI gets the same
   case from both backends. The chat view reads `phase(of:)` and
   shouldn't have to special-case backend.

4. If the method affects what's on disk (a new field on
   `Conversation`), add the field with `Optional` for BWC and an
   `init(... = nil)` so old `.json` files still decode.

5. Test by manually round-tripping a conversation: save to disk,
   restart the app, reopen the chat. Anything that doesn't
   gracefully degrade with `nil` is a migration bug.

## 13. Recipe: add a native tool to `DeepSeekTools`

The tool catalogue is enumerated in
`Sources/DeepSeekTools/DefaultTools.swift`. To add a new one:

1. **Pick a category.** `readOnly` / `planning` / `mutating` /
   `dangerous` / `network`. This drives both the plan-mode filter
   and the permission policy — get this right first.

2. **Drop the implementation under `Sources/DeepSeekTools/Tools/`.**
   Conform to `Tool`:

   ```swift
   struct MyNewTool: Tool {
       let schema: ToolSchema = .build {
           $0.name = "mytool"
           $0.summary = "What it does, one line."
           $0.category = .readOnly
           $0.parameters = SchemaBuilder.object {
               $0.required("path")
               $0.string("path", description: "Absolute or root-relative.")
               $0.boolean("verbose", defaultValue: false)
           }
       }

       func run(input: ToolInput, context: ToolContext)
           async throws -> ToolResult
       {
           let path = try input.string("path")
           // … do work, respect context.rootDirectory + context.sandbox …
           return ToolResult(output: "…",
                              metadata: ["bytesRead": .int(123)])
       }
   }
   ```

3. **Wire it into `DefaultTools.standard(...)`** so the registry
   sees it on construction. Tests should drive a registry through
   `register(MyNewTool())` directly so they don't depend on the
   default order.

4. **Add a smoke test** under `Tests/DeepSeekToolsTests/`. The
   bare minimum is "dispatching the tool by name returns a
   non-empty output for a valid input + throws `.invalidInput`
   for a missing required arg".

5. **Add per-tool persistence (if any)** under
   `Sources/DeepSeekUI/State/` and add a `PersistencePaths` entry.
   The plan/task/todo tools share `PlanStore` for this reason —
   long-lived state lives there, not on the tool struct.

6. **Don't forget the permission story.** Mutating / dangerous /
   network tools go through `PermissionDelegate.request(...)` the
   first time per session per `(tool, category)`. The default
   `AutoPermissionDelegate` auto-allows everything except
   `.dangerous`; the GUI's `NativeToolHost` bridges to
   `PermissionPromptView`. If the tool needs unusual handling
   (e.g. a long-running confirmation), extend the delegate
   protocol rather than baking it into the tool.

7. **Regenerate the Xcode project** if you added files:

   ```bash
   ./Tools/generate-xcodeproj.sh
   ```

   `swift build` picks the new files up automatically.

The existing tools are the template — `ReadTool` for read-only,
`WriteTool` for mutating, `ShellTool` for dangerous, `WebFetchTool`
for network. Each is < 200 LOC.

## 14. Sources

- Reference implementation pattern: `Sources/DeepSeekKit/Layers/Hadamard.swift`
- Test pattern: `Tests/DeepSeekKitTests/HadamardTests.swift`
- Build plugin: `Plugins/MetalLibPlugin/MetalLibPlugin.swift`
- Function-constant reservations: [KERNELS.md](KERNELS.md#conventions)
- Recipe-style examples (more concrete code): [EXAMPLES.md](EXAMPLES.md)
- Desktop-app module map: [MODULES.md](MODULES.md#sourcesdeepseekui-desktop-app)
- Desktop-app data flow: [ARCHITECTURE.md](ARCHITECTURE.md#desktop-app-architecture)

See also [TESTING.md](TESTING.md) for the test surface and tolerance
guidance.
