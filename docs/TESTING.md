# Testing

XCTest target structure, what each test verifies, and how to add new
ones. Open this when:

- You broke a test and need to know what it was checking.
- You added a kernel and need to write its test.
- You want to know what's *not* covered.

## At a glance

| Test file | Module | What it verifies |
|---|---|---|
| `HadamardTests` | `Layers/Hadamard.swift` | FWHT matches CPU reference on dims 4…512; H∘H = identity |
| `HCSinkhornTests` | `Layers/HCSinkhorn.swift` | Sinkhorn split (pre/post/comb) matches CPU; comb is doubly-stochastic |
| `ActQuantTests` | `Layers/ActQuant.swift` | FP8 + FP4 round-trip == CPU; scale match; relative bound on FP8 |
| `SoftmaxAxisTests` | `Layers/SoftmaxAxis.swift` | Softmax along last/middle/first axis, plus Compressor-shape `[B, NB, R, C]` |
| `TopKTests` | `Layers/TopK.swift` | Top-K values+indices match CPU; k=1 == argmax |
| `MoEDispatchTests` | `Layers/MoEDispatch.swift` | Gather→scatter identity with weight=1; top-K=2 weighted sum |
| `OverlapTransformTests` | `Layers/OverlapTransform.swift` | Shuffle matches CPU; first-row pad behavior |
| `EinsumTests` | `Layers/Einsum.swift` | `bshd,btd→bsht` and `bsgd,grd→bsgr` match CPU |
| `AttentionIndicesTests` | `Layers/AttentionIndices.swift` | Sliding window prefill + decode wrap; compressed prefill + decode |
| `LinearTests` | `Layers/Linear.swift` | f32 GEMM exact match; FP8 GEMM within FP8's precision bound |
| `SparseAttentionTests` | `Layers/SparseAttention.swift` | Match CPU on small randomized input; all-padding produces zeros |
| `HyperConnectionsTests` | `Layers/HyperConnections.swift` | HC.pre and HC.post separately match CPU |
| `CompressorTests` | `Layers/Compressor.swift` | Prefill no-overlap path matches CPU on toy config |
| `BPETokenizerTests` | `BPETokenizer.swift` | encode/decode round-trip on mini vocab + UTF-8 + special tokens |
| `MoEHashRoutingTests` | `Layers/MoE.swift` | Hash routing layer picks experts from tid2eid lookup |

15 files total, one or more `func test...` per file. All tests build
and run via `swift test -c release`.

## 1. Running tests

```bash
# All tests
swift test -c release 2>&1 | tail -20

# A single class
swift test -c release --filter HadamardTests

# A single function
swift test -c release --filter HadamardTests/testFWHTMatchesReferenceCPU
```

XCTest emits one `XCTAssert*` line per failure. A passing run ends with
`Test Suite ... passed`.

First test run is slower because `swift build -c release` rebuilds the
test target (links against DeepSeekKit). Subsequent runs are
incremental.

## 2. Test pattern

Every Layer with a Metal kernel follows the same template:

```swift
import XCTest
import Metal
@testable import DeepSeekKit

final class FooTests: XCTestCase {

    func testMatchesCPU() throws {
        // 1. Build small randomized input.
        let N = 4, D = 8
        let input = randomArray(N * D, seed: 1)

        // 2. Wrap as a Tensor (copies bytes into a fresh MTLBuffer).
        let t = input.withUnsafeBytes {
            Tensor.from(bytes: $0, shape: [N, D], dtype: .f32)
        }

        // 3. Run the Metal kernel via its wrapper.
        let cmd = Device.shared.queue.makeCommandBuffer()!
        Foo.apply(t, in: cmd)
        cmd.commit(); cmd.waitUntilCompleted()

        // 4. Read GPU output back to host.
        let gpu = t.toFloatArray()

        // 5. Compute the same thing on the CPU.
        let cpu = Foo.referenceCPU(input, ...)

        // 6. Compare element-wise within tolerance.
        for i in 0..<gpu.count {
            XCTAssertEqual(gpu[i], cpu[i], accuracy: 1e-5, "i=\(i)")
        }

        // 7. Keep input alive until past the GPU read.
        _ = input
    }

    private func randomArray(_ count: Int, seed: UInt64) -> [Float] {
        var state = seed | 1
        var out = [Float](repeating: 0, count: count)
        for i in 0..<count {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let frac = Float(Double(state >> 11) / Double(1 << 53))
            out[i] = (frac - 0.5) * 4
        }
        return out
    }
}
```

Key conventions:

- **Same RNG**: deterministic LCG (linear congruential generator) seeded
  per test. Pattern reused across files. Don't import a fancy RNG.
- **Small shapes**: `4 × 8`, `2 × 16`, etc. Enough to exercise the
  kernel without exploding test time.
- **Pin input alive**: the `_ = input` line at the end keeps Swift from
  freeing the source array before the GPU read. Important because
  `Tensor.from(bytes:)` makes its own copy but if you ever switch to a
  zero-copy `Tensor.fromMappedBytes` the test would silently break.
- **Tolerance** chosen by class of operation (see §3 below).

## 3. Tolerance choices

| Operation | Typical tolerance | Reason |
|---|---|---|
| Pure float math (softmax, hadamard, einsum, RMSNorm) | 1e-5 absolute | F32 round-off accumulation across small N |
| Multi-step composition (HC pre/post, Compressor prefill) | 1e-3 absolute | Same plus accumulated rounding through 4-5 kernels |
| Quantized round-trip (FP8 round-trip) | 1e-3 absolute on the *dequantized* value | Scale rounding noise + FP8 ULP |
| FP8 GEMM | relative error < 0.5 | FP8 weight precision (3-bit mantissa) is intrinsically ±12% |
| Index equality (TopK indices, Gate routing) | exact (no tolerance) | They're integers |

Choose the tightest tolerance that the operation reliably produces.
If your test fails intermittently at the tolerance bound, the kernel
probably has a real bug — don't loosen the tolerance to make it pass.

## 4. Anatomy of a representative test

`Tests/DeepSeekKitTests/HadamardTests.swift` is the simplest model:

```swift
func testFWHTMatchesReferenceCPU() throws {
    for dim in [4, 16, 64, 128, 512] {
        let rows = 3
        var input = randomRow(rows: rows, dim: dim, seed: UInt64(dim))

        // CPU reference (in-place, per row)
        var cpu = input
        for r in 0..<rows {
            var slice = Array(cpu[r * dim ..< (r + 1) * dim])
            Hadamard.referenceCPU(&slice)
            for i in 0..<dim { cpu[r * dim + i] = slice[i] }
        }

        // GPU
        let t = input.withUnsafeBytes { raw in
            Tensor.from(bytes: raw, shape: [rows, dim], dtype: .f32)
        }
        let cmd = Device.shared.queue.makeCommandBuffer()!
        Hadamard.apply(t, in: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        let gpu = t.toFloatArray()

        for i in 0..<gpu.count {
            XCTAssertEqual(gpu[i], cpu[i], accuracy: 1e-4,
                           "dim=\(dim) idx=\(i)")
        }
        _ = input
    }
}
```

This tests:
- Multiple dims (FWHT correctness depends on log₂(dim) butterfly steps)
- Multiple rows (kernel handles row-major batching)
- Bounded error vs a known-correct CPU implementation

The companion `testFWHTInvolution` then asserts `H∘H = I` (a property
that must hold regardless of the implementation), which catches
swapped-sign bugs that the comparison test might miss.

## 5. Tests by category

### Pure kernel correctness (Metal == CPU)

`Hadamard`, `HCSinkhorn`, `ActQuant`, `SoftmaxAxis`, `TopK`,
`OverlapTransform`, `Einsum`, `MoEDispatch`. Each compares the GPU
output to a pure-Swift implementation that lives in the same Layer
file as `static func referenceCPU(...)`.

### Property tests (algebraic identities)

- `testFWHTInvolution`: applying Hadamard twice returns identity
  (modulo float rounding).
- `testCombApproximatelyDoublyStochastic`: after 20 Sinkhorn iters, the
  `comb` matrix has row+col sums ≈ 1.
- `testAllPaddingProducesZero` (sparse_attn): all -1 indices → zero
  output (correctly absorbed by sink-only denominator).
- `testGatherScatterRoundTripIdentity` (MoE dispatch): permute + invert
  with weight=1 returns the original tensor.
- `testFP8RoundTripRelativeBound`: random inputs through FP8 quant +
  dequant stay within FP8's intrinsic precision bound.

### Indexing / table-driven

- `AttentionIndicesTests`: hand-computes the expected sliding-window
  indices for specific positions and asserts byte-for-byte equality.
  Helps catch off-by-one in the ring wrap.
- `MoEHashRoutingTests`: prepares a known `tid2eid` table, runs the
  gate, asserts the emitted indices match the lookup.

### Tokenizer

`BPETokenizerTests` exercises three paths:
1. Encode + decode round trip on a mini vocab with merges.
2. Special-token passthrough (added_tokens emit literal ids).
3. UTF-8 round trip on multi-byte strings (Italian, Japanese, emoji),
   using a synthetic per-byte vocab.

## 6. What's NOT tested

### Engine

- **End-to-end forward against Python reference** — Tier 3 deferred
  (`docs/ROADMAP.md` T3.4). Needs PyTorch + CUDA to dump activations
  from `Reference/inference/generate.py` on a toy config.
- **`Linear` with FP4 weight** — `LinearTests.swift` covers f32 and
  FP8 paths but not FP4. The FP4 GEMM kernel `gemm_fp8_fp4_to_f32`
  exists and is verified via the FP4 dequant path in `ActQuantTests`,
  but the full FP8 act × FP4 weight pipeline doesn't have a dedicated
  test.
- **MLA forward** — too large for a self-contained unit test (full
  attention with cache + compressor + sparse_attn). The constituent
  pieces are all individually verified.
- **MoEFFN forward** — same. Gate, dispatch, expert linears, scatter
  are individually tested; the assembled MoEFFN isn't.
- **Compressor decode path** — only the prefill path has a CPU
  reference + test (`CompressorTests.testPrefillNoOverlapMatchesCPU`).
  Decode-step state machine is exercised only via end-to-end runs.
- **HyperConnections in a full Block** — pre and post are tested
  individually, but a round-trip through one Block isn't.
- **Converter output structural correctness** — no automated test
  that the converter produces a checkpoint that
  `Transformer.load(from:)` can consume. Smoke-tested manually.
- **Sampler.sample** with non-trivial options — partly covered:
  `Tests/DeepSeekKitTests/SamplerTests.swift` exercises top-K /
  top-P / min-p / tfs / typical / freq / presence / Mirostat v2
  on canned distributions, but the full pipeline ordering test
  (every layer toggled together against a CPU reference) is
  still open.
- **`EncodingDSV4` on the golden corpus** under
  `Reference/encoding/`. Tool-call DSML emit + parse + the
  `__delegate_to_agent` synthetic schema have no token-by-token
  comparison vs the Python reference.
- **`Transformer.snapshotKVCache` / `restoreKVCache`** round-trip —
  the buffer copy + slot-shape match path isn't covered. Today the
  proof is "sub-agent delegation works in the chat", which is
  indirect.

The post-llama.cpp-gap merge added engine-side coverage that
*does* live in `Tests/DeepSeekKitTests/`:

- `ChatTemplateTests.swift` — dispatcher chooses
  `DSV4Template` vs `JinjaChatTemplate` from the loaded
  directory's metadata; both produce the expected prompt for a
  fixed conversation.
- `JinjaTemplateTests.swift` — the Jinja2 subset driver against
  small fixture templates (variable interpolation, for/if/elif,
  filters, raise_exception).
- `GGUFTests.swift` — magic + version check, KV metadata
  parsing, tensor info table, pass-through `load(name:)`,
  `unsupportedType` for quantised dtypes.
- `WordPieceTokenizerTests.swift` — encode/decode round-trip
  + `##` continuation prefix on a BERT vocab.
- `SamplerTests.swift` — see above.

### Desktop app

`DeepSeekUI` has no test bundle. The recently-added
`DeepSeekTools` target does — `Tests/DeepSeekToolsTests/`:

  - `ToolRegistryTests.swift` — dispatch happy path on a
    minimal in-memory tool, plan-mode filter (`.mutating` /
    `.dangerous` denied without prompting), session permission
    cache (`allowOnce` → next call doesn't re-prompt), durable
    `.alwaysAllow` short-circuit.
  - `SlashCommandTests.swift` — built-in command parser
    (`/mode plan`, `/skill <name>`, malformed inputs) and the
    custom command registry path.

What's still uncovered:

- **`OpenRouterClient`** with a mock URLSession:
  - `validateKey` returns 401/403 → throws `unauthorized`.
  - `fetchModels` deserialises a fixture payload into
    `[OpenRouterModel]`.
  - `streamChatCompletion` parses a fixture SSE stream into the
    right sequence of `OpenAIStreamChunk`s (incl. heartbeats /
    `[DONE]` / partial JSON skipping).
- **`MCPClient`** with a Swift-inline mock stdio server:
  - Handshake (`initialize` + `notifications/initialized` +
    `tools/list`) finalises `status == .connected(toolCount: N)`
    with the parsed tools.
  - `tools/call` round-trip flattens the typed `content` array
    into a string.
  - A bogus JSON-RPC error finishes the awaiting continuation
    with `.rpc(code:message:)`.
- **`ChatStore.runRemoteLoop`** with a stub backend:
  - Single-iteration happy path: tool_calls empty → marks idle,
    stamps `usage.completionTokens` / `usage.total_cost`.
  - Multi-iteration: emits tool_calls iter 1, MCP returns a
    canned output, iter 2 finalises. Verifies the placeholder
    swap + new placeholder append happen in order.
  - Cap hit: 9 iterations → phase becomes `.error` with the
    truncation message.
- **`KeychainStore`** round-trip — set/get/delete/exists on a
  unique account scoped to the test bundle.
- **`MCPServerLibrary.importClaudeDesktopJSON`** on a fixture
  payload that exercises both shapes (`{ mcpServers: { … } }` and
  the legacy bare-dict variant).
- **Each native tool's `run`** on a sandboxed temp directory:
  `ReadTool` / `WriteTool` / `EditTool` / `ApplyPatchTool` /
  `GlobTool` / `GrepTool` against a known fixture tree;
  `RepoOverviewTool` on a fixture with a `Package.swift` +
  `package.json`. `ShellTool` / `WebFetchTool` / `WebSearchTool`
  need an injected fake subprocess / `URLProtocol` mock.
- **`PermissionStore` durable round-trip** — set/get/delete a
  `<tool>:<category>` rule, ensure it survives an init.
- **`PlanStore` actor-isolation contract** — concurrent
  `plan` / `task` / `todo` calls don't corrupt state.

## 7. How to add a new test

1. **Pick the layer/module you want to cover.** It must have a Metal
   wrapper *and* a pure-Swift reference. If the reference doesn't
   exist, add it to the same file first (see existing examples like
   `Layers/Hadamard.swift:referenceCPU`).

2. **Create the file** at `Tests/DeepSeekKitTests/FooTests.swift`. The
   SwiftPM test target auto-discovers any `*Tests.swift` file in that
   directory.

3. **Use the template from §2.** Pick a small shape that exercises
   the kernel's tricky paths (multiple rows for per-row kernels;
   power-of-2 dims for FWHT; sizes around block boundaries for
   quant). Add property tests where you can.

4. **Choose tolerance from §3.** Run the test and check the typical
   error magnitude — if it's an order of magnitude tighter than your
   tolerance, tighten until it's a comfortable 3-5× margin.

5. **Run locally**: `swift test -c release --filter FooTests`.

6. **Add a row** to the table in §0 of this doc.

## 8. Sources

- `Tests/DeepSeekKitTests/` — all test files
- `Sources/DeepSeekKit/Layers/Hadamard.swift:60` — minimal example of
  a Layer + referenceCPU + Metal dispatch
- `Tests/DeepSeekKitTests/HadamardTests.swift` — minimal example of a
  test class

See also [DEVELOPING.md](DEVELOPING.md) for the full "how to add a
kernel + wrapper + test" recipe.
