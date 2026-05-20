# Code recipes

Runnable snippets for the most common tasks. Each example shows the
short narrative, the code, and the command that exercises it. Most
can be pasted into a new Swift file under `Sources/deepseek/` or
adapted into a one-off `swift run` target for quick experiments.

For step-by-step CLI workflows (build, convert, run), see
[USAGE.md](USAGE.md). For the API surface they call, see
[MODULES.md](MODULES.md).

## At a glance

| Recipe | Section | Difficulty |
|---|---|---|
| **Engine** | | |
| Load a Tensor from safetensors | §1 | trivial |
| Dispatch a Metal kernel manually | §2 | trivial |
| Linear with f32 / BF16 / FP8 weight | §3 | easy |
| Add a new Layer (toy LayerNorm) | §4 | medium |
| Generate one token with random weights | §5 | easy |
| Generate with real V4-Flash weights | §6 | requires checkpoint |
| Encode a chat message with tool calls | §7 | easy |
| Inspect converted output | §8 | easy |
| FAQ: what if I want to … | §9 | — |
| **Desktop app** | | |
| Add an OpenRouter API key from code | §10 | trivial |
| Stream a one-shot completion via OpenRouter | §11 | easy |
| Register an MCP server programmatically | §12 | easy |
| Define an agent that delegates to another agent | §13 | medium |
| Invoke a native tool through the registry | §14 | easy |
| Define an agent locked to Plan mode | §15 | trivial |
| Add a custom slash command | §16 | easy |
| Read GGUF metadata + a pass-through tensor | §17 | easy |
| Render a chat with a Jinja2 template | §18 | easy |
| Hit the local OpenAI-compatible server with `curl` | §19 | trivial |
| Sources | §20 | — |

## 1. Load a Tensor from safetensors

```swift
import DeepSeekKit

let url = URL(fileURLWithPath: "/Volumes/DATA/V4-Flash-bf16/model-00001-of-00133.safetensors")
let file = try SafeTensorsFile(url: url)

// List what's inside:
print("Tensors in this shard:")
for (name, entry) in file.entries.prefix(5) {
    print("  \(name) \(entry.dtype) \(entry.shape)")
}

// Load one specific tensor (returns a Tensor referencing the mmap):
let embed = try file.load("embed.weight")
print("embed shape: \(embed.shape), dtype: \(embed.dtype)")

// Peek at the first few values:
let arr = embed.toFloatArray()
print("first 8 values: \(arr.prefix(8))")
```

For a directory of shards, use `WeightLoader` instead — it indexes
every shard and exposes `load(_:)` and `tryLoad(_:[fallbackNames])`:

```swift
let loader = try WeightLoader(directory: URL(fileURLWithPath: "/Volumes/DATA/V4-Flash-bf16"))
print("Indexed \(loader.totalKnownNames) tensors across \(loader.shardCount) shards")

let embed = try loader.load("embed.weight")
let lmHead = try loader.tryLoad(["head.weight", "lm_head.weight"])
```

## 2. Dispatch a Metal kernel manually

The lowest-level pattern: encode a kernel, dispatch, sync, read back.

```swift
import DeepSeekKit
import Metal

// 1. Build the input tensor (host array → MTLBuffer).
let input: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
let t = input.withUnsafeBytes {
    Tensor.from(bytes: $0, shape: [1, 8], dtype: .f32)
}

// 2. Run the Hadamard kernel (in place).
let cmd = Device.shared.queue.makeCommandBuffer()!
Hadamard.apply(t, in: cmd)
cmd.commit(); cmd.waitUntilCompleted()

// 3. Read back and verify against the CPU reference.
let out = t.toFloatArray()
var ref = input
Hadamard.referenceCPU(&ref)
print("GPU: \(out)")
print("CPU: \(ref)")
for i in 0..<out.count {
    assert(abs(out[i] - ref[i]) < 1e-5)
}
_ = input  // pin source alive
```

Save this as `Sources/deepseek/main.swift` body (replacing the
existing CLI) or in a new executable target; run with `swift run -c
release`.

## 3. Linear with different weight dtypes

`Linear` dispatches automatically based on `weight.dtype`. Same
construction, different speed/precision.

### F32 weight (slowest, exact)

```swift
let inFeat = 16, outFeat = 8
let wArr = (0..<outFeat*inFeat).map { Float($0) * 0.01 }
let wT = wArr.withUnsafeBytes {
    Tensor.from(bytes: $0, shape: [outFeat, inFeat], dtype: .f32)
}
let lin = Linear(inFeatures: inFeat, outFeatures: outFeat, weight: wT, scale: nil)

let xArr: [Float] = Array(repeating: 1.0, count: inFeat)
let xT = xArr.withUnsafeBytes {
    Tensor.from(bytes: $0, shape: [1, inFeat], dtype: .f32)
}

let cmd = Device.shared.queue.makeCommandBuffer()!
let y = lin(xT, in: cmd)
cmd.commit(); cmd.waitUntilCompleted()
print("y: \(y.toFloatArray())")
```

### BF16 weight (Metal-native, fast)

Same code, but build `wT` with `dtype: .bf16`. The buffer must contain
BF16-packed bytes — easiest is to construct it via the converter
output, or pack manually:

```swift
import Foundation

func packBF16(_ floats: [Float]) -> [UInt16] {
    return floats.map { f in
        let bits = f.bitPattern
        let rounded = bits &+ ((bits >> 16) & 1) &+ 0x7FFF
        return UInt16(truncatingIfNeeded: rounded >> 16)
    }
}

let wBF16 = packBF16(wArr)
let wT = wBF16.withUnsafeBytes {
    Tensor.from(bytes: $0, shape: [outFeat, inFeat], dtype: .bf16)
}
let lin = Linear(inFeatures: inFeat, outFeatures: outFeat, weight: wT, scale: nil)
// rest identical
```

### FP8 weight (smallest, slower at inference)

Needs both a `weight` and a `scale` tensor. After running the
converter with `--target-dtype keep`, FP8 tensors in the output are
ready to load via `WeightLoader.tryLoad(["foo.weight", "foo.scale"])`.

```swift
let w = try loader.load("layers.0.attn.wq_a.weight")
let s = try loader.load("layers.0.attn.wq_a.scale")
let lin = Linear(inFeatures: w.shape[1], outFeatures: w.shape[0],
                 weight: w, scale: s)
// Use the same way; Linear dispatches to gemm_fp8_to_f32 because w.dtype == .fp8E4M3.
```

See [DTYPES.md](DTYPES.md) for the bit layouts and tradeoffs.

## 4. Add a new Layer (toy LayerNorm)

Demonstrates the kernel + wrapper + test pattern end to end.

### Step 1 — Kernel

`Sources/DeepSeekKit/Kernels/layernorm.metal`:

```metal
#include <metal_stdlib>
using namespace metal;

// y = (x - mean) * rsqrt(var + eps) * gamma + beta
// One threadgroup per row, threads cooperate on the reductions.

kernel void layernorm_f32(
    device const float* x      [[buffer(0)]],
    device const float* gamma  [[buffer(1)]],
    device const float* beta   [[buffer(2)]],
    device float*       y      [[buffer(3)]],
    constant uint&      dim    [[buffer(4)]],
    constant float&     eps    [[buffer(5)]],
    threadgroup float*  shared_ [[threadgroup(0)]],
    uint  row    [[threadgroup_position_in_grid]],
    uint  tid    [[thread_position_in_threadgroup]],
    uint  tgsize [[threads_per_threadgroup]]
) {
    device const float* xr = x + row * dim;
    device float*       yr = y + row * dim;

    // sum + sum-of-squares
    float s = 0, ss = 0;
    for (uint i = tid; i < dim; i += tgsize) { float v = xr[i]; s += v; ss += v * v; }
    shared_[tid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgsize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared_[tid] += shared_[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = shared_[0] / float(dim);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    shared_[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgsize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared_[tid] += shared_[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float var = shared_[0] / float(dim) - mean * mean;
    float scale = rsqrt(var + eps);

    for (uint i = tid; i < dim; i += tgsize) {
        yr[i] = (xr[i] - mean) * scale * gamma[i] + beta[i];
    }
}
```

### Step 2 — Wrapper

`Sources/DeepSeekKit/Layers/LayerNorm.swift`:

```swift
import Foundation
import Metal

public final class LayerNorm {
    public let gamma: Tensor
    public let beta: Tensor
    public let eps: Float
    private let pipeline: MTLComputePipelineState

    public init(gamma: Tensor, beta: Tensor, eps: Float) {
        precondition(gamma.dtype == .f32 && beta.dtype == .f32)
        precondition(gamma.shape == beta.shape && gamma.shape.count == 1)
        self.gamma = gamma; self.beta = beta; self.eps = eps
        self.pipeline = Device.shared.makePipeline("layernorm_f32")
    }

    public func callAsFunction(_ x: Tensor, in cmd: MTLCommandBuffer) -> Tensor {
        precondition(x.dtype == .f32)
        let dim = x.shape.last!
        let rows = x.count / dim
        let y = Tensor.empty(shape: x.shape, dtype: .f32)

        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(x.buffer, offset: x.offset, index: 0)
        enc.setBuffer(gamma.buffer, offset: gamma.offset, index: 1)
        enc.setBuffer(beta.buffer, offset: beta.offset, index: 2)
        enc.setBuffer(y.buffer, offset: 0, index: 3)
        var d = UInt32(dim); var e = eps
        enc.setBytes(&d, length: 4, index: 4)
        enc.setBytes(&e, length: 4, index: 5)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        enc.setThreadgroupMemoryLength(256 * MemoryLayout<Float>.size, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: tg)
        enc.endEncoding()
        return y
    }

    public static func referenceCPU(x: [Float], gamma: [Float], beta: [Float],
                                     rows: Int, dim: Int, eps: Float) -> [Float] {
        var out = [Float](repeating: 0, count: rows * dim)
        for r in 0..<rows {
            var s: Float = 0, ss: Float = 0
            for d in 0..<dim { s += x[r * dim + d]; ss += x[r * dim + d] * x[r * dim + d] }
            let mean = s / Float(dim)
            let variance = ss / Float(dim) - mean * mean
            let scale = 1.0 / (variance + eps).squareRoot()
            for d in 0..<dim {
                out[r * dim + d] = (x[r * dim + d] - mean) * scale * gamma[d] + beta[d]
            }
        }
        return out
    }
}
```

### Step 3 — Test

`Tests/DeepSeekKitTests/LayerNormTests.swift`:

```swift
import XCTest
@testable import DeepSeekKit

final class LayerNormTests: XCTestCase {
    func testMatchesCPU() throws {
        let rows = 3, dim = 16
        let x = (0..<rows*dim).map { Float($0) * 0.01 }
        let gamma = [Float](repeating: 1.0, count: dim)
        let beta = [Float](repeating: 0.0, count: dim)

        let xT = x.withUnsafeBytes { Tensor.from(bytes: $0, shape: [rows, dim], dtype: .f32) }
        let gammaT = gamma.withUnsafeBytes { Tensor.from(bytes: $0, shape: [dim], dtype: .f32) }
        let betaT = beta.withUnsafeBytes { Tensor.from(bytes: $0, shape: [dim], dtype: .f32) }

        let ln = LayerNorm(gamma: gammaT, beta: betaT, eps: 1e-5)
        let cmd = Device.shared.queue.makeCommandBuffer()!
        let y = ln(xT, in: cmd)
        cmd.commit(); cmd.waitUntilCompleted()

        let gpu = y.toFloatArray()
        let cpu = LayerNorm.referenceCPU(x: x, gamma: gamma, beta: beta,
                                          rows: rows, dim: dim, eps: 1e-5)
        for i in 0..<gpu.count {
            XCTAssertEqual(gpu[i], cpu[i], accuracy: 1e-4)
        }
        _ = x; _ = gamma; _ = beta
    }
}
```

Run: `swift test -c release --filter LayerNormTests`.

## 5. Generate one token with random weights

End-to-end smoke without needing the V4 checkpoint:

```swift
import DeepSeekKit

var cfg = ModelConfig()
cfg.dim = 256              // shrink for speed
cfg.nLayers = 2
cfg.nRoutedExperts = 4
cfg.maxSeqLen = 128

let model = Transformer.randomInit(config: cfg)
let prompt: [[Int]] = [[1, 2, 3, 4, 5]]
let logits = model.forward(inputIds: prompt, startPos: 0)
print("Logits shape: \(logits.shape)")

let next = Sampler.argmax(logits)
print("Next token (greedy): \(next)")
```

No checkpoint, no tokenizer needed. Run with
`swift run -c release deepseek` after replacing
`Sources/deepseek/main.swift` with this snippet (or create a new
executable target).

## 6. Generate with real weights

Once the converter has produced
`/Volumes/DATA/V4-Flash-bf16/`, the inference CLI is the easiest
path:

```bash
.build/release/deepseek /Volumes/DATA/V4-Flash-bf16 \
    "Ciao, dimmi una poesia in tre versi" \
    --mode chat --max-tokens 100
```

Programmatic equivalent (drop into a fresh CLI target):

```swift
import DeepSeekKit

let modelDir = URL(fileURLWithPath: "/Volumes/DATA/V4-Flash-bf16")
let config = try ModelConfig.load(from: modelDir.appendingPathComponent("config.json"))
let tokenizer = try TokenizerLoader.load(from: modelDir.appendingPathComponent("tokenizer.json"))

print("Loading model...")
let model = try Transformer.load(config: config, from: modelDir)

let prompt = EncodingDSV4.encodeMessages([
    Message(role: .user, content: "Ciao!")
], mode: .chat)
var ids = tokenizer.encode(prompt)

var logits = model.forward(inputIds: [ids], startPos: 0)
var opts = SamplingOptions(temperature: 0.7, topP: 0.9)
var generated: [Int] = []

for step in 0..<100 {
    let next = Sampler.sample(logits, history: generated, options: &opts)
    if next == (tokenizer.eosId ?? -1) { break }
    generated.append(next)
    print(tokenizer.decode([next]), terminator: "")
    fflush(stdout)
    if step == 99 { break }
    logits = model.forward(inputIds: [[next]], startPos: ids.count + step)
}
print()
```

## 7. Encode a chat message with tool calls

Server-side prompt construction:

```swift
import DeepSeekKit

let toolSchemas = """
[
  {"name": "search", "description": "Web search",
   "parameters": {"query": {"type": "string"}}}
]
"""

let prompt = EncodingDSV4.encodeMessages([
    Message(role: .system, content: "You are a helpful assistant."),
    Message(role: .user, content: "What's the weather in Paris?"),
], mode: .max, toolSchemasJSON: toolSchemas)

print(prompt)
```

Parse a completion that may contain a tool call:

```swift
let modelOutput = """
<think>
The user is asking about weather. I should use the search tool.
</think>
<｜DSML｜tool_calls>
<｜DSML｜invoke name="search">
<｜DSML｜parameter name="query" string="true">Paris weather today</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>
"""

let msg = EncodingDSV4.parseCompletion(modelOutput, mode: .max)
print("Reasoning: \(msg.reasoningContent ?? "(none)")")
print("Content: \(msg.content)")
for tc in msg.toolCalls {
    print("Tool: \(tc.name), args: \(tc.args)")
}
```

## 8. Inspect converted output

After a conversion run, sanity-check the output:

```bash
# Count + size of shards:
ls /Volumes/DATA/V4-Flash-bf16/*.safetensors | wc -l
du -sh /Volumes/DATA/V4-Flash-bf16/

# Peek at the index:
python3 -c "
import json
d = json.load(open('/Volumes/DATA/V4-Flash-bf16/model.safetensors.index.json'))
print('total_size:', d['metadata']['total_size'])
print('weight_map entries:', len(d['weight_map']))
print('first 5:', list(d['weight_map'].items())[:5])
"

# Inspect a specific shard's tensor list (Python with safetensors):
python3 -c "
from safetensors import safe_open
with safe_open('/Volumes/DATA/V4-Flash-bf16/model-00001-of-00133.safetensors', 'pt') as f:
    for key in list(f.keys())[:10]:
        t = f.get_slice(key)
        print(f'{key}: {t.get_dtype()} {t.get_shape()}')
"
```

Pure-Swift version (no Python needed) inside a `swift run` target:

```swift
import DeepSeekKit

let shardURL = URL(fileURLWithPath: "/Volumes/DATA/V4-Flash-bf16/model-00001-of-00133.safetensors")
let f = try SafeTensorsFile(url: shardURL)
print("Shard contains \(f.entries.count) tensors. First 10:")
for (name, e) in f.entries.sorted(by: { $0.key < $1.key }).prefix(10) {
    print("  \(name) [\(e.dtype)] \(e.shape)")
}
```

## 9. FAQ — what if I want to …

### … use a different prompt format

`EncodingDSV4` is one strategy. For raw inference, pass `--mode raw` to
the CLI and provide the prompt exactly as the model expects. You can
also build your own encoder; it just needs to produce a string the
tokenizer accepts.

### … swap BF16 for F16 in the converter output

```bash
.build/release/converter --target-dtype f16 ...
```

Same size as BF16 (16 bits), smaller dynamic range. Use only if you
have a downstream tool that specifically needs F16; otherwise BF16
matches the reference better.

### … keep FP4 / FP8 on disk and let Metal dequant at runtime

```bash
.build/release/converter --target-dtype keep ...
```

Output is ~140 GB (vs ~600 GB for BF16 V4-Flash). Inference is slower
because every GEMM does shader-side dequant.

### … run on a partial checkpoint

The loader (`Transformer.load`) prints a summary of names it couldn't
find at the end of loading and fills them with random init. So you
can experiment with a subset of layers' weights converted. Expect
garbage outputs, but the pipeline runs end-to-end.

### … profile a specific kernel

See [PERFORMANCE.md](PERFORMANCE.md) §4 "How to profile". Short version:
use Xcode's GPU frame capture for a single decode call, drill into
the slow encoder.

### … add a new tokenizer

Implement the `Tokenizer` protocol in
`Sources/DeepSeekKit/Tokenizer.swift`. The minimal interface is
`encode/decode/bosId/eosId`. Hook it up by changing the
`TokenizerLoader.load` body to dispatch on a config flag or file
suffix.

### … run the model under a different `maxSeqLen`

Set `max_seq_len` in `config.json` before running. The KV cache and
RoPE freqs are sized from this. Setting it tightly to the actual
prompt length saves a lot of RAM.

### … check that my changes didn't regress

```bash
swift test -c release
```

15 test classes; should all pass. Add yours before submitting
significant kernel changes.

## 10. Add an OpenRouter API key from code

Normal flow is the Settings → API Keys tab, but the same Keychain
slot is reachable from `KeychainStore`. Useful for fixtures /
one-off scripts.

```swift
import DeepSeekKit
// (or whichever module-mapping you set up — KeychainStore lives in
// the app target, so you'd vendor it into a test bundle to use it
// outside of the app.)

try KeychainStore.set("sk-or-v1-…",
                       account: KeychainAccount.openRouterAPIKey)

// Read it back from anywhere on the same Mac (the macOS Keychain
// keeps it per-user, not per-process):
let key = KeychainStore.get(account: KeychainAccount.openRouterAPIKey)

// Forget it (Settings → API Keys → Delete equivalent):
try KeychainStore.delete(account: KeychainAccount.openRouterAPIKey)
```

## 11. Stream a one-shot completion via OpenRouter

Minimal driver: no chat history, no tools. Useful for a quick
"does this model do what I expect?" probe outside the chat
surface.

```swift
let client = OpenRouterClient()
let body: [String: Any] = [
    "model": "anthropic/claude-3.5-sonnet",
    "messages": [
        ["role": "user", "content": "Write a haiku about Metal shaders."]
    ],
    "temperature": 0.7,
    "max_tokens": 80,
    "usage": ["include": true]
]

let stream = client.streamChatCompletion(apiKey: key, body: body)

var buf = ""
do {
    for try await chunk in stream {
        if let delta = chunk.choices.first?.delta?.content {
            buf.append(delta)
            print(delta, terminator: "")
        }
        if let usage = chunk.usage {
            print("\n→ cost: $\(usage.totalCost ?? 0)")
        }
    }
} catch {
    print("openrouter failed: \(error.localizedDescription)")
}
```

To validate a key without spending any tokens:

```swift
try await client.validateKey(key)        // throws on 401/403
let models = try await client.fetchModels(apiKey: key)
print("OpenRouter knows \(models.count) models")
```

## 12. Register an MCP server programmatically

The MCP servers tab does this through the library; the same path
is reachable from code if you need to scaffold N servers from a
script (e.g. seeding a clean install).

```swift
let library = MCPServerLibrary()    // loads mcp.json from disk
let pool = MCPClientPool()
pool.attach(to: library)

library.add(MCPServerConfig(
    name: "filesystem",
    command: "npx",
    args: ["@modelcontextprotocol/server-filesystem",
           "/Users/me/Documents/scratch"],
    env: [:],
    enabled: true))

pool.librarySynced(library)

// Wait for the JSON-RPC handshake. In production this is observed
// reactively in the UI; in a script we poll the pool's per-server
// status:
while true {
    let client = pool.client(forServer: library.servers.first!.id)!
    if case .connected = client.status { break }
    try await Task.sleep(nanoseconds: 200_000_000)
}

let tools = pool.allTools().map(\.qualifiedName)
print("Exposed: \(tools)")
```

Importing an existing Claude Desktop config is one call:

```swift
let json = try Data(contentsOf:
    URL(fileURLWithPath: "/Users/me/Library/Application Support/Claude/claude_desktop_config.json"))
let added = try library.importClaudeDesktopJSON(json)
print("Imported \(added) servers")
pool.librarySynced(library)
```

## 13. Define an agent that delegates to another agent

Agents are persisted under `agents.json`. From the GUI this is
Settings → Agents → +. Programmatically:

```swift
let agents = AgentLibrary()

// Worker agent: code-search specialist, only filesystem tools.
let searcher = AgentConfig(
    name: "Code Searcher",
    summary: "Greps the user's project files",
    systemPrompt: """
        You are a code-search specialist. You can read files from
        the user's working directory. Always cite the path of any
        file you reference.
        """,
    allowedToolNames: ["filesystem__read_file",
                       "filesystem__list_directory"],
    defaultMode: "chat",
    temperature: 0.5,
    iconName: "magnifyingglass",
    tint: "purple")
agents.add(searcher)

// Orchestrator: no MCP tools (it delegates instead), high reasoning.
let orchestrator = AgentConfig(
    name: "Architect",
    summary: "Designs systems by delegating focused look-ups",
    systemPrompt: """
        You are an architect. You don't read files directly —
        delegate that to "Code Searcher" and synthesise the
        answer.
        """,
    allowedToolNames: [],         // explicit "no MCP tools"
    defaultMode: "high",
    temperature: 0.7,
    iconName: "compass.drawing",
    tint: "blue")
agents.add(orchestrator)
```

In the chat: attach **Architect** from the toolbar picker. When you
ask it a code question, the system block carries the synthetic
`__delegate_to_agent` schema with **Code Searcher** as a delegable
option. The Architect emits
`{ agent_name: "Code Searcher", task: "List + read every .swift in …" }`,
the worker runs in isolation through `runSubAgentToCompletion`, its
reply comes back as a tool output, and the Architect synthesises
the final answer.

The live delegation appears as a pinned card above the composer
with the worker's icon + the streaming buffer.

Nesting cap is 3 levels; the chain (`[hostID, agent.id, …]`) is
checked on every dispatch to refuse cycles.

## 14. Invoke a native tool through the registry

The chat invokes tools through `NativeToolHost.dispatch`; the
same `ToolRegistry` is reachable from any code that imports
`DeepSeekTools`. Useful for scripts / fixtures.

```swift
import DeepSeekTools

// Spin up a registry pre-populated with the built-in tools.
let registry = ToolRegistry()
await registry.register(DefaultTools.standard(planStore: PlanStore()))

// Auto-approve every consent prompt for this script (CLI default).
let delegate = AutoPermissionDelegate(autoAllowDangerous: false)

let ctx = ToolContext(
    rootDirectory: URL(fileURLWithPath: "/Users/me/code/project"),
    mode: .build,
    permissions: delegate,
    sandbox: .off)

let result = try await registry.dispatch(
    name: "read",
    input: ["path": "README.md", "limit": 50],
    context: ctx)

print(result.output)
```

`dispatch` throws `ToolError.denied` when:

- the registry filters the tool out by mode
  (`.plan` + `.mutating` → denied without prompting);
- `PermissionStore` carries an `alwaysDeny` for that
  `(tool, category)`;
- the delegate's prompt returns `.denied`.

To drive the GUI's modal flow from outside the SwiftUI tree
(tests, debug scripts), implement your own `PermissionDelegate`
that returns the canned decision.

## 15. Define an agent locked to Plan mode

A read-only-by-default reviewer that can only `read` / `glob` /
`grep` / `repo_overview`. Plan-mode keeps `.mutating` +
`.dangerous` tools out of the schema the model sees; the
`allowedToolNames` set further narrows the MCP catalogue.

```swift
let agents = AgentLibrary()

let reviewer = AgentConfig(
    name: "Code Reviewer",
    summary: "Reviews the project without touching files",
    systemPrompt: """
        You are a code reviewer. Inspect files, point out bugs
        and inconsistencies, propose changes in prose — never
        edit anything.
        """,
    allowedToolNames: [
        "filesystem__read_file",
        "filesystem__list_directory"
    ],
    defaultMode: "high",          // think hard
    agentMode: .plan,             // ← Plan mode locks mutating/dangerous
    allowedSkillIDs: [],          // no skill restriction
    temperature: 0.5,
    iconName: "magnifyingglass.circle",
    tint: "purple")
agents.add(reviewer)
```

Attach it to a chat from the toolbar Agent picker; the mode
picker beside the composer will lock to "Plan" with a 🔒 hint.

## 16. Add a custom slash command

Built-in commands live in `SlashCommandLibrary`'s catalogue.
Custom commands today are added at registry-construction time;
a user-facing Settings → Slash Commands tab is on the TODO list.

```swift
import DeepSeekTools

let standup = SlashCommand(
    name: "standup",
    summary: "Insert a daily standup template",
    expansion: """
        ## Yesterday
        - 

        ## Today
        - 

        ## Blockers
        - 
        """)

// SlashCommandLibrary is @MainActor + ObservableObject; the
// `register(_:)` API extends the live catalogue and persists
// custom entries to slash_commands.json.
slashCommands.register(standup)
```

In the composer: type `/standup`, pick it from the palette, and
the draft is replaced with the expansion text.

## 17. Read GGUF metadata + a pass-through tensor

`GGUFFile` is a read-only wrapper over a `.gguf` file. Useful
for inspecting a downloaded checkpoint without booting the chat.
For unsupported (quantised) dtypes the loader throws — fall
back to `info(name:)` to get the raw byte range.

```swift
import DeepSeekKit

let url = URL(fileURLWithPath: "/Volumes/DATA/Llama-3-8B.gguf")
let gguf = try GGUFFile(url: url)

print("GGUF v\(gguf.header.version) — \(gguf.header.tensorCount) tensors")
for (key, value) in gguf.header.metadata.prefix(8) {
    print("  \(key) = \(value)")
}

// Inspect one tensor.
let info = try gguf.info(name: "token_embd.weight")
print("token_embd.weight shape: \(info.shape), type: \(info.type), \(info.byteCount) B")

// Zero-copy view for pass-through dtypes.
do {
    let t = try gguf.load(name: "token_embd.weight")
    print("loaded as \(t.dtype)")
} catch GGUFError.unsupportedType(let type) {
    print("\(type) needs a dequant kernel — see GGUF.md")
}
```

## 18. Render a chat with a Jinja2 template

When the loaded model directory carries a `tokenizer_config.json`
with a `chat_template`, the dispatcher wraps it in
`JinjaChatTemplate`. The same render path is reachable directly
for diagnostic / preview use:

```swift
import DeepSeekKit

let templateSource = """
{%- for m in messages -%}
{{ '<|' + m.role + '|>\n' + m.content + '<|end|>\n' }}
{%- endfor -%}
{%- if add_generation_prompt -%}{{ '<|assistant|>\n' }}{%- endif -%}
"""

let template = try JinjaChatTemplate(source: templateSource)

let messages: [Message] = [
    Message(role: .system,    content: "You are a code reviewer."),
    Message(role: .user,      content: "Review this snippet."),
]

let opts = ChatTemplateOptions(
    mode: .chat,
    addGenerationPrompt: true,
    toolSchemasJSON: nil)

let prompt = try template.render(messages: messages, options: opts)
print(prompt)
```

For a real model, prefer:

```swift
let template = service.chatTemplate      // resolved at load time
let prompt = try template.render(messages: messages, options: opts)
```

That way the V4 path keeps using `DSV4Template` (fast, no Jinja
interpretation) and the Llama / Mistral / Qwen path picks up the
model's own format automatically.

## 19. Hit the local OpenAI-compatible server with `curl`

The desktop app can expose an OpenAI-shaped HTTP API on localhost so
external clients (VS Code, Zed, TUI, GitHub Actions, custom scripts)
can talk to the locally-loaded model with no extra plumbing. Wire it
up in **Settings → Server**: enable the toggle, pick a port (default
`8080`), optionally add a bearer token. Once `Listening on http://…`
turns green, hit the API:

```bash
# Catalog of loaded models (just the one currently loaded today).
curl -s http://127.0.0.1:8080/v1/models | jq

# Non-streaming chat completion. `model` is ignored — we always use
# whatever is loaded in the desktop app.
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local",
    "messages": [
      {"role": "system", "content": "You are helpful."},
      {"role": "user", "content": "What is 2+2?"}
    ],
    "max_tokens": 64
  }' | jq

# Streaming (SSE). `data: {...}` chunks per round, terminated by
# `data: [DONE]`.
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "Tell me a joke."}],
    "stream": true
  }'
```

If you configured a bearer token, every request needs an
`Authorization: Bearer <token>` header.

The `tools[]` field of the request is honored as a name-filter
against the server's MCP registry: pass an empty array to opt out of
tools entirely, an array of `{type:"function", function:{name:"…"}}`
entries to restrict the model to those names, or omit the field to
expose every MCP tool the desktop app currently has connected. The
schemas inside `tools[]` are ignored — the server uses the
authoritative schema from the MCP server itself. Tool dispatch runs
server-side (up to 21 round-trips per request); only the final
text-only round streams to the client.

Limitations of the first cut:

  - JSON-schema constrained output (`response_format`) is not
    plumbed yet (T3 in `TODO.md` §10).
  - Streaming clients see no SSE bytes while intermediate tool
    rounds are running.
  - One generation at a time across all HTTP clients (the
    `InferenceService` queue serializes).

## 20. Sources

- All recipes here exercise public API documented in
  [MODULES.md](MODULES.md) and kernels listed in [KERNELS.md](KERNELS.md).
- For the development workflow around these recipes (where to put
  files, how to commit, common errors), see [DEVELOPING.md](DEVELOPING.md).
- For deeper dives on the numerics, see [DTYPES.md](DTYPES.md) and
  [MEMORY.md](MEMORY.md).
