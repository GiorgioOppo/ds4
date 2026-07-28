import Foundation
import Metal
import DS4Core

// First-cut Laguna S 2.1 resident engine, modeled on GLM52ResidentModel but
// deliberately simpler: Laguna requires full residency upstream, so every
// validated tensor is uploaded once into shared MTLBuffers and there is no
// SSD streaming, sidecar or expert cache. The per-token graph mirrors
// `laguna_graph_forward_token` in the reference `laguna-s2.1` branch,
// dispatching the shared GLM primitives where upstream shares them
// (rms_norm/matvec/add/argmax-free readback, `kernel_glm52_router_select`
// with 10 active experts, the K-quant MoE matvecs) plus the Laguna kernels
// for per-head norm/RoPE, the F16 ring KV store and gated GQA attention.
//
// Scope limits of this cut, all enforced at load with distinct errors:
// - Q8_0-signal layout only (official Q4_K routed recipe and the mixed
//   RoutedQ2_K/Last27Q3_K file, whose K-quant matvecs are wired); the
//   legacy F16/Q4_K recipe is refused until its Q6_K/F16 paths are wired;
// - prefill is token-by-token through the decode path (correct, not fast);
// - two command-buffer syncs per MoE layer (the router selection is read
//   back on the host to pick expert slabs, like the GLM chained decode).

public struct LagunaResidentModelOptions: Sendable {
    /// Number of leading transformer blocks to run (nil = all 48). Front
    /// truncation exists for bring-up tests against partial uploads; the
    /// output head then reads whatever hidden state the last kept layer
    /// produced.
    public var layerCount: Int?
    /// KV positions per full-attention layer (sliding-window layers stay at
    /// the 512-row ring regardless).
    public var cacheCapacity: Int = 4_096

    public init() {}
}

public enum LagunaResidentModelError: Error, Sendable, CustomStringConvertible {
    case unsupportedRecipe(String)
    case bufferAllocation(String)
    case invalidToken(Int32)
    case contextFull(capacity: Int)

    public var description: String {
        switch self {
        case .unsupportedRecipe(let reason):
            return "unsupported Laguna recipe for the resident engine: \(reason)"
        case .bufferAllocation(let what):
            return "Laguna resident engine could not allocate \(what)"
        case .invalidToken(let token):
            return "Laguna token id \(token) is outside the vocabulary"
        case .contextFull(let capacity):
            return "Laguna context is full (\(capacity) positions)"
        }
    }
}

public final class LagunaResidentModel {
    public let configuration: LagunaConfiguration
    public private(set) var position = 0
    public var loadedLayerCount: Int { layers.count }
    public var contextCapacity: Int { cacheCapacity }

    private let runtime: MetalRuntime
    private let model: GGUFModel
    private let shape: LagunaShape
    private let cacheCapacity: Int
    private let embeddingDescriptor: LagunaWeightDescriptor

    private static let simdgroupsPerThreadgroup = 4
    private static let q8Type: UInt32 = 8

    /// Routed-expert types with a wired matvec path (the K-quant dot helpers
    /// shared with GLM): Q2_K, Q3_K and Q4_K — the published official and
    /// mixed recipes. The tensor schema already guarantees the three routed
    /// projections of one layer share this type on the Q8_0-signal layout.
    private static let routedTypes: Set<UInt32> = [10, 11, 12]

    private enum FFN {
        case dense(gate: MTLBuffer, up: MTLBuffer, down: MTLBuffer)
        case moe(routedType: UInt32,
                 routerRows: MTLBuffer, routerBias: MTLBuffer,
                 routedGate: MTLBuffer, routedUp: MTLBuffer,
                 routedDown: MTLBuffer,
                 gateUpExpertBytes: Int, downExpertBytes: Int,
                 sharedGate: MTLBuffer, sharedUp: MTLBuffer,
                 sharedDown: MTLBuffer)
    }

    private struct Layer {
        let index: Int
        let spec: LagunaAttentionSpec
        let attnNorm: MTLBuffer
        let query: MTLBuffer
        let key: MTLBuffer
        let value: MTLBuffer
        let gate: MTLBuffer
        let queryNorm: MTLBuffer
        let keyNorm: MTLBuffer
        let attnOutput: MTLBuffer
        let ffnNorm: MTLBuffer
        let ffn: FFN
        let cache: LagunaMetalKVCache
    }

    private var layers: [Layer] = []
    private let outputNorm: MTLBuffer
    private let outputHead: MTLBuffer

    // Persistent activation scratch (F32 unless noted).
    private var hidden: MTLBuffer
    private var hiddenNext: MTLBuffer
    private let normed: MTLBuffer
    private let queryRows: MTLBuffer
    private let keyRows: MTLBuffer
    private let valueRows: MTLBuffer
    private let gateRows: MTLBuffer
    private let heads: MTLBuffer
    private let attnOut: MTLBuffer
    private let afterAttn: MTLBuffer
    private let ffnNormed: MTLBuffer
    private let ffnMid: MTLBuffer
    private let ffnOut: MTLBuffer
    private let routerLogits: MTLBuffer
    private let routerSelected: MTLBuffer   // Int32 × expertsUsed
    private let routerWeights: MTLBuffer    // Float × expertsUsed
    private let routerProbs: MTLBuffer      // Float × expertCount
    private let logits: MTLBuffer

    public init(runtime: MetalRuntime, path: String,
                options: LagunaResidentModelOptions = .init()) throws {
        let model = try GGUFModel(path: path, metalMapping: false,
                                  prefetchCPU: false)
        let map = try LagunaWeightMap(model: model)
        guard map.quantizationLayout == .signalQ8 else {
            throw LagunaResidentModelError.unsupportedRecipe(
                "the first cut runs the official Q8_0 signal-path file; the "
                + "legacy F16/Q4_K recipe needs its matvec kernels wired first"
            )
        }

        self.runtime = runtime
        self.model = model
        self.configuration = map.configuration
        self.shape = map.configuration.shape
        self.cacheCapacity = max(1, options.cacheCapacity)
        self.embeddingDescriptor = try map.global(.tokenEmbedding)

        func upload(_ descriptor: LagunaWeightDescriptor) throws -> MTLBuffer {
            guard let buffer = runtime.device.makeBuffer(
                bytes: model.mapBase + Int(descriptor.absOffset),
                length: Int(descriptor.bytes),
                options: .storageModeShared
            ) else {
                throw LagunaResidentModelError.bufferAllocation(descriptor.name)
            }
            return buffer
        }
        func scratch(_ count: Int, _ what: String) throws -> MTLBuffer {
            guard let buffer = runtime.device.makeBuffer(
                length: count * MemoryLayout<Float>.stride,
                options: .storageModeShared
            ) else {
                throw LagunaResidentModelError.bufferAllocation(what)
            }
            memset(buffer.contents(), 0, count * MemoryLayout<Float>.stride)
            return buffer
        }

        self.outputNorm = try upload(map.global(.outputNorm))
        self.outputHead = try upload(map.global(.output))

        let embd = Int(shape.nEmbd)
        let maxQueryWidth = Int(shape.nHead) * Int(shape.nHeadDim)
        self.hidden = try scratch(embd, "hidden state")
        self.hiddenNext = try scratch(embd, "next hidden state")
        self.normed = try scratch(embd, "attention norm")
        self.queryRows = try scratch(maxQueryWidth, "query projection")
        self.keyRows = try scratch(Int(shape.keyValueProjectionWidth), "key projection")
        self.valueRows = try scratch(Int(shape.keyValueProjectionWidth), "value projection")
        self.gateRows = try scratch(Int(shape.nHead), "gate projection")
        self.heads = try scratch(maxQueryWidth, "attention heads")
        self.attnOut = try scratch(embd, "attention output")
        self.afterAttn = try scratch(embd, "post-attention hidden")
        self.ffnNormed = try scratch(embd, "FFN norm")
        self.ffnMid = try scratch(Int(shape.nFFDense), "FFN intermediate")
        self.ffnOut = try scratch(embd, "FFN output")
        self.routerLogits = try scratch(LagunaRouterReference.expertCount, "router logits")
        self.routerSelected = try scratch(LagunaRouterReference.expertsUsed, "router selection")
        self.routerWeights = try scratch(LagunaRouterReference.expertsUsed, "router weights")
        self.routerProbs = try scratch(LagunaRouterReference.expertCount, "router probabilities")
        self.logits = try scratch(Int(shape.nVocab), "logits")

        let totalLayers = Int(shape.nLayer)
        let keptLayers = min(max(options.layerCount ?? totalLayers, 1), totalLayers)
        var layers: [Layer] = []
        layers.reserveCapacity(keptLayers)
        for index in 0..<keptLayers {
            let spec = LagunaAttentionSpec.spec(
                forLayer: index, shape: shape,
                contextSize: cacheCapacity
            )
            let cacheRows = shape.isSlidingWindowLayer(index)
                ? min(Int(shape.nSWA), cacheCapacity)
                : cacheCapacity
            let boundedSpec = LagunaAttentionSpec(
                headCount: spec.headCount, kvHeadCount: spec.kvHeadCount,
                headDim: spec.headDim, rotationDims: spec.rotationDims,
                cacheCapacity: cacheRows,
                ropeFrequencyBase: spec.ropeFrequencyBase,
                ropeFrequencyScale: spec.ropeFrequencyScale,
                extrapolationFactor: spec.extrapolationFactor,
                attentionFactor: spec.attentionFactor,
                betaFast: spec.betaFast, betaSlow: spec.betaSlow,
                ropeOriginalContext: spec.ropeOriginalContext,
                rmsEpsilon: spec.rmsEpsilon
            )

            let ffn: FFN
            if index < Int(shape.nLeadingDense) {
                ffn = .dense(
                    gate: try upload(map.layer(index, .denseGate)),
                    up: try upload(map.layer(index, .denseUp)),
                    down: try upload(map.layer(index, .denseDown))
                )
            } else {
                let routed = try map.routedExperts(layer: index)
                guard Self.routedTypes.contains(routed.gate.type),
                      routed.up.type == routed.gate.type,
                      routed.down.type == routed.gate.type else {
                    throw LagunaResidentModelError.unsupportedRecipe(
                        "layer \(index) routed experts are "
                        + "\(GGUF.typeName(routed.gate.type))/"
                        + "\(GGUF.typeName(routed.down.type)); this engine "
                        + "runs coherent Q2_K/Q3_K/Q4_K routed experts"
                    )
                }
                let gateUpExpertBytes = Int(routed.gate.bytes)
                    / Int(shape.nExpert)
                let downExpertBytes = Int(routed.down.bytes)
                    / Int(shape.nExpert)
                ffn = .moe(
                    routedType: routed.gate.type,
                    routerRows: try upload(map.layer(index, .router)),
                    routerBias: try upload(map.layer(index, .routerBias)),
                    routedGate: try upload(routed.gate),
                    routedUp: try upload(routed.up),
                    routedDown: try upload(routed.down),
                    gateUpExpertBytes: gateUpExpertBytes,
                    downExpertBytes: downExpertBytes,
                    sharedGate: try upload(map.layer(index, .sharedGate)),
                    sharedUp: try upload(map.layer(index, .sharedUp)),
                    sharedDown: try upload(map.layer(index, .sharedDown))
                )
            }

            layers.append(Layer(
                index: index,
                spec: boundedSpec,
                attnNorm: try upload(map.layer(index, .attentionNorm)),
                query: try upload(map.layer(index, .attentionQuery)),
                key: try upload(map.layer(index, .attentionKey)),
                value: try upload(map.layer(index, .attentionValue)),
                gate: try upload(map.layer(index, .attentionGate)),
                queryNorm: try upload(map.layer(index, .attentionQueryNorm)),
                keyNorm: try upload(map.layer(index, .attentionKeyNorm)),
                attnOutput: try upload(map.layer(index, .attentionOutput)),
                ffnNorm: try upload(map.layer(index, .feedForwardNorm)),
                ffn: ffn,
                cache: try runtime.lagunaKVCache(
                    capacity: cacheRows,
                    rowWidth: Int(shape.keyValueProjectionWidth)
                )
            ))
        }
        self.layers = layers
    }

    // MARK: Public API

    public func resetContext() {
        position = 0
    }

    /// Dequantize one Q8_0 embedding row on the CPU (rows are read per token,
    /// exactly like the reference engine; the table is never resident).
    public func embeddingRow(_ token: Int32) throws -> [Float] {
        try Self.dequantizeQ8Row(
            base: model.mapBase + Int(embeddingDescriptor.absOffset),
            row: Int(token),
            rowCount: Int(shape.nVocab),
            width: Int(shape.nEmbd)
        )
    }

    public func forwardNext(_ token: Int32) throws -> [Float] {
        guard token >= 0, Int(token) < Int(shape.nVocab) else {
            throw LagunaResidentModelError.invalidToken(token)
        }
        guard position < cacheCapacity else {
            throw LagunaResidentModelError.contextFull(capacity: cacheCapacity)
        }

        let embedding = try embeddingRow(token)
        embedding.withUnsafeBytes {
            _ = memcpy(hidden.contents(), $0.baseAddress!, $0.count)
        }

        for layer in layers {
            try forward(layer: layer, position: position)
        }

        try encodeOutputHead()
        position += 1

        let count = Int(shape.nVocab)
        let pointer = logits.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    public func forwardNextGreedy(_ token: Int32) throws -> Int32 {
        let logits = try forwardNext(token)
        var best = logits[0]
        var bestIndex: Int32 = 0
        for index in 1..<logits.count where logits[index] > best {
            best = logits[index]
            bestIndex = Int32(index)
        }
        return bestIndex
    }

    /// Token-by-token prefill through the decode path; returns the logits of
    /// the last prompt token.
    public func prefill(_ tokens: [Int32]) throws -> [Float] {
        var last: [Float] = []
        for token in tokens {
            last = try forwardNext(token)
        }
        return last
    }

    public func generateGreedy(prompt: [Int32], maxNewTokens: Int,
                               endTokens: Set<Int32> = []) throws -> [Int32] {
        guard !prompt.isEmpty else { return [] }
        var logits = try prefill(prompt)
        var output: [Int32] = []
        for _ in 0..<maxNewTokens {
            var best = logits[0]
            var bestIndex: Int32 = 0
            for index in 1..<logits.count where logits[index] > best {
                best = logits[index]
                bestIndex = Int32(index)
            }
            if endTokens.contains(bestIndex) { break }
            output.append(bestIndex)
            logits = try forwardNext(bestIndex)
        }
        return output
    }

    // MARK: Layer forward

    private func forward(layer: Layer, position: Int) throws {
        let embd = Int(shape.nEmbd)
        let spec = layer.spec
        let queryWidth = spec.queryWidth
        let kvWidth = spec.keyValueWidth

        var commands = try beginCommands()

        // Attention half: norm, projections, per-head norm/RoPE, KV store,
        // gated attention, output projection and residual.
        try encodeRMSNorm(commands, input: hidden, weight: layer.attnNorm,
                          output: normed, width: embd)
        try encodeMatvecPair(commands, x: normed,
                             typeA: Self.q8Type, rowsA: queryWidth, weightsA: layer.query,
                             typeB: Self.q8Type, rowsB: kvWidth, weightsB: layer.key,
                             inputWidth: embd, outA: queryRows, outB: keyRows)
        try encodeMatvecPair(commands, x: normed,
                             typeA: Self.q8Type, rowsA: kvWidth, weightsA: layer.value,
                             typeB: Self.q8Type, rowsB: spec.headCount, weightsB: layer.gate,
                             inputWidth: embd, outA: valueRows, outB: gateRows)
        try encodeQKNormRope(commands, spec: spec, position: position,
                             queryNorm: layer.queryNorm, keyNorm: layer.keyNorm)
        try encodeStoreKV(commands, cache: layer.cache, position: position)
        try encodeAttention(commands, spec: spec, cache: layer.cache,
                            position: position)
        try encodeQuantMatvec(commands, type: Self.q8Type, rows: embd,
                              inputWidth: queryWidth, weights: layer.attnOutput,
                              input: heads, output: attnOut, accumulate: false)
        try encodeAdd(commands, a: hidden, b: attnOut, out: afterAttn,
                      count: embd)
        try encodeRMSNorm(commands, input: afterAttn, weight: layer.ffnNorm,
                          output: ffnNormed, width: embd)

        switch layer.ffn {
        case .dense(let gate, let up, let down):
            let intermediate = Int(shape.nFFDense)
            try encodePairSwiGLU(commands, type: Self.q8Type, rows: intermediate,
                                 inputWidth: embd, gateRows: gate, upRows: up,
                                 input: ffnNormed, routeWeight: 1, mid: ffnMid)
            try encodeQuantMatvec(commands, type: Self.q8Type, rows: embd,
                                  inputWidth: intermediate, weights: down,
                                  input: ffnMid, output: ffnOut,
                                  accumulate: false)
            try encodeAdd(commands, a: afterAttn, b: ffnOut, out: hiddenNext,
                          count: embd)
            try endCommands(commands)

        case .moe(let routedType, let routerRows, let routerBias,
                  let routedGate, let routedUp, let routedDown,
                  let gateUpExpertBytes, let downExpertBytes,
                  let sharedGate, let sharedUp, let sharedDown):
            // Phase A ends with the router: selection is read back on the
            // host to address the expert slabs (GLM chained-decode pattern).
            try encodeF32Matvec(commands, rows: routerRows,
                                rowCount: LagunaRouterReference.expertCount,
                                inputWidth: embd, x: ffnNormed,
                                out: routerLogits)
            try encodeRouterSelect(commands, bias: routerBias)
            try endCommands(commands)

            let used = LagunaRouterReference.expertsUsed
            let selectedPointer = routerSelected.contents()
                .bindMemory(to: Int32.self, capacity: used)
            let weightPointer = routerWeights.contents()
                .bindMemory(to: Float.self, capacity: used)
            let selected = Array(UnsafeBufferPointer(start: selectedPointer,
                                                     count: used))
            let routeWeights = Array(UnsafeBufferPointer(start: weightPointer,
                                                         count: used))

            commands = try beginCommands()
            let expertWidth = Int(shape.nFFExpert)
            for (rank, expert) in selected.enumerated() {
                let e = Int(expert)
                try encodePairSwiGLU(
                    commands, type: routedType, rows: expertWidth,
                    inputWidth: embd,
                    gateRows: routedGate, gateOffset: e * gateUpExpertBytes,
                    upRows: routedUp, upOffset: e * gateUpExpertBytes,
                    input: ffnNormed, routeWeight: routeWeights[rank],
                    mid: ffnMid
                )
                try encodeQuantMatvec(
                    commands, type: routedType, rows: embd,
                    inputWidth: expertWidth,
                    weights: routedDown, weightsOffset: e * downExpertBytes,
                    input: ffnMid, output: ffnOut, accumulate: rank != 0
                )
            }
            let sharedWidth = Int(shape.nFFShared)
            try encodePairSwiGLU(commands, type: Self.q8Type, rows: sharedWidth,
                                 inputWidth: embd, gateRows: sharedGate,
                                 upRows: sharedUp, input: ffnNormed,
                                 routeWeight: 1, mid: ffnMid)
            try encodeQuantMatvec(commands, type: Self.q8Type, rows: embd,
                                  inputWidth: sharedWidth, weights: sharedDown,
                                  input: ffnMid, output: ffnOut,
                                  accumulate: true)
            try encodeAdd(commands, a: afterAttn, b: ffnOut, out: hiddenNext,
                          count: embd)
            try endCommands(commands)
        }

        swap(&hidden, &hiddenNext)
    }

    private func encodeOutputHead() throws {
        let commands = try beginCommands()
        try encodeRMSNorm(commands, input: hidden, weight: outputNorm,
                          output: normed, width: Int(shape.nEmbd))
        try encodeQuantMatvec(commands, type: Self.q8Type,
                              rows: Int(shape.nVocab),
                              inputWidth: Int(shape.nEmbd),
                              weights: outputHead, input: normed,
                              output: logits, accumulate: false)
        try endCommands(commands)
    }

    // MARK: Command encoding

    private struct Commands {
        let buffer: MTLCommandBuffer
        let encoder: MTLComputeCommandEncoder
    }

    private func beginCommands() throws -> Commands {
        guard let buffer = runtime.queue.makeCommandBuffer(),
              let encoder = buffer.makeComputeCommandEncoder() else {
            throw LagunaResidentModelError.bufferAllocation("command buffer")
        }
        return Commands(buffer: buffer, encoder: encoder)
    }

    private func endCommands(_ commands: Commands) throws {
        commands.encoder.endEncoding()
        commands.buffer.commit()
        commands.buffer.waitUntilCompleted()
        if let error = commands.buffer.error { throw error }
    }

    private func setArguments(_ encoder: MTLComputeCommandEncoder,
                              _ words: [UInt32]) {
        words.withUnsafeBytes {
            encoder.setBytes($0.baseAddress!, length: $0.count, index: 0)
        }
    }

    private func rowGroups(_ rows: Int) -> MTLSize {
        MTLSize(width: (rows + Self.simdgroupsPerThreadgroup - 1)
                    / Self.simdgroupsPerThreadgroup,
                height: 1, depth: 1)
    }

    private var simdgroupThreads: MTLSize {
        MTLSize(width: 32, height: Self.simdgroupsPerThreadgroup, depth: 1)
    }

    private func encodeRMSNorm(_ commands: Commands, input: MTLBuffer,
                               weight: MTLBuffer, output: MTLBuffer,
                               width: Int) throws {
        let pipeline = try runtime.pipeline("kernel_glm52_rms_norm_f32")
        let encoder = commands.encoder
        encoder.setComputePipelineState(pipeline)
        setArguments(encoder, [UInt32(width), shape.rmsEpsilon.bitPattern, 0, 0])
        encoder.setBuffer(input, offset: 0, index: 1)
        encoder.setBuffer(weight, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.setThreadgroupMemoryLength(256 * MemoryLayout<Float>.stride,
                                           index: 0)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
    }

    private func encodeMatvecPair(_ commands: Commands, x: MTLBuffer,
                                  typeA: UInt32, rowsA: Int, weightsA: MTLBuffer,
                                  typeB: UInt32, rowsB: Int, weightsB: MTLBuffer,
                                  inputWidth: Int,
                                  outA: MTLBuffer, outB: MTLBuffer) throws {
        let pipeline = try runtime.pipeline("kernel_glm52_matvec_pair_sg")
        let encoder = commands.encoder
        encoder.setComputePipelineState(pipeline)
        setArguments(encoder, [typeA, UInt32(rowsA), typeB, UInt32(rowsB),
                               UInt32(inputWidth), 0, 0, 0])
        encoder.setBuffer(x, offset: 0, index: 1)
        encoder.setBuffer(weightsA, offset: 0, index: 2)
        encoder.setBuffer(weightsB, offset: 0, index: 3)
        encoder.setBuffer(outA, offset: 0, index: 4)
        encoder.setBuffer(outB, offset: 0, index: 5)
        encoder.dispatchThreadgroups(rowGroups(rowsA + rowsB),
                                     threadsPerThreadgroup: simdgroupThreads)
    }

    private func encodePairSwiGLU(_ commands: Commands, type: UInt32, rows: Int,
                                  inputWidth: Int,
                                  gateRows: MTLBuffer, gateOffset: Int = 0,
                                  upRows: MTLBuffer, upOffset: Int = 0,
                                  input: MTLBuffer, routeWeight: Float,
                                  mid: MTLBuffer) throws {
        let pipeline = try runtime.pipeline("kernel_glm52_moe_pair_swiglu_sg")
        let encoder = commands.encoder
        encoder.setComputePipelineState(pipeline)
        setArguments(encoder, [type, UInt32(rows), UInt32(inputWidth),
                               routeWeight.bitPattern])
        encoder.setBuffer(input, offset: 0, index: 1)
        encoder.setBuffer(gateRows, offset: gateOffset, index: 2)
        encoder.setBuffer(upRows, offset: upOffset, index: 3)
        encoder.setBuffer(mid, offset: 0, index: 4)
        encoder.dispatchThreadgroups(rowGroups(rows),
                                     threadsPerThreadgroup: simdgroupThreads)
    }

    private func encodeQuantMatvec(_ commands: Commands, type: UInt32, rows: Int,
                                   inputWidth: Int,
                                   weights: MTLBuffer, weightsOffset: Int = 0,
                                   input: MTLBuffer, output: MTLBuffer,
                                   accumulate: Bool) throws {
        let name = accumulate
            ? "kernel_glm52_moe_down_acc_sg"
            : "kernel_glm52_moe_down_sg"
        let pipeline = try runtime.pipeline(name)
        let encoder = commands.encoder
        encoder.setComputePipelineState(pipeline)
        setArguments(encoder, [type, UInt32(rows), UInt32(inputWidth), 0])
        encoder.setBuffer(input, offset: 0, index: 1)
        encoder.setBuffer(weights, offset: weightsOffset, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.dispatchThreadgroups(rowGroups(rows),
                                     threadsPerThreadgroup: simdgroupThreads)
    }

    private func encodeF32Matvec(_ commands: Commands, rows: MTLBuffer,
                                 rowCount: Int, inputWidth: Int,
                                 x: MTLBuffer, out: MTLBuffer) throws {
        let pipeline = try runtime.pipeline("kernel_glm52_matvec_f32")
        let encoder = commands.encoder
        encoder.setComputePipelineState(pipeline)
        setArguments(encoder, [UInt32(rowCount), UInt32(inputWidth), 0, 0])
        encoder.setBuffer(rows, offset: 0, index: 1)
        encoder.setBuffer(x, offset: 0, index: 2)
        encoder.setBuffer(out, offset: 0, index: 3)
        let threads = 256
        encoder.dispatchThreadgroups(
            MTLSize(width: (rowCount + threads - 1) / threads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
    }

    private func encodeAdd(_ commands: Commands, a: MTLBuffer, b: MTLBuffer,
                           out: MTLBuffer, count: Int) throws {
        let pipeline = try runtime.pipeline("kernel_glm52_add_f32")
        let encoder = commands.encoder
        encoder.setComputePipelineState(pipeline)
        setArguments(encoder, [UInt32(count), 0, 0, 0])
        encoder.setBuffer(a, offset: 0, index: 1)
        encoder.setBuffer(b, offset: 0, index: 2)
        encoder.setBuffer(out, offset: 0, index: 3)
        let threads = 256
        encoder.dispatchThreadgroups(
            MTLSize(width: (count + threads - 1) / threads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
    }

    private func encodeRouterSelect(_ commands: Commands,
                                    bias: MTLBuffer) throws {
        let pipeline = try runtime.pipeline("kernel_glm52_router_select")
        let encoder = commands.encoder
        encoder.setComputePipelineState(pipeline)
        setArguments(encoder, [
            UInt32(LagunaRouterReference.expertCount),
            UInt32(LagunaRouterReference.expertsUsed),
            LagunaRouterReference.expertWeightScale.bitPattern,
            0,
        ])
        encoder.setBuffer(routerLogits, offset: 0, index: 1)
        encoder.setBuffer(bias, offset: 0, index: 2)
        encoder.setBuffer(routerSelected, offset: 0, index: 3)
        encoder.setBuffer(routerWeights, offset: 0, index: 4)
        encoder.setBuffer(routerProbs, offset: 0, index: 5)
        // 256 selection scores + 256 int32 indices.
        encoder.setThreadgroupMemoryLength(
            256 * (MemoryLayout<Float>.stride + MemoryLayout<Int32>.stride),
            index: 0
        )
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
    }

    private func encodeQKNormRope(_ commands: Commands,
                                  spec: LagunaAttentionSpec,
                                  position: Int,
                                  queryNorm: MTLBuffer,
                                  keyNorm: MTLBuffer) throws {
        let pipeline = try runtime.pipeline("kernel_laguna_qk_head_rms_norm_rope_neox")
        let encoder = commands.encoder
        encoder.setComputePipelineState(pipeline)
        let combinedHeads = spec.headCount + spec.kvHeadCount
        var arguments = [UInt32](repeating: 0, count: 14)
        arguments[0] = 1
        arguments[1] = UInt32(combinedHeads)
        arguments[2] = UInt32(spec.headDim)
        arguments[3] = UInt32(spec.rotationDims)
        arguments[4] = UInt32(position)
        arguments[5] = UInt32(spec.ropeOriginalContext)
        arguments[6] = spec.rmsEpsilon.bitPattern
        arguments[7] = spec.ropeFrequencyBase.bitPattern
        arguments[8] = spec.ropeFrequencyScale.bitPattern
        arguments[9] = spec.extrapolationFactor.bitPattern
        arguments[10] = spec.attentionFactor.bitPattern
        arguments[11] = spec.betaFast.bitPattern
        arguments[12] = spec.betaSlow.bitPattern
        setArguments(encoder, arguments)
        encoder.setBuffer(queryRows, offset: 0, index: 1)
        encoder.setBuffer(keyRows, offset: 0, index: 2)
        encoder.setBuffer(queryNorm, offset: 0, index: 3)
        encoder.setBuffer(keyNorm, offset: 0, index: 4)
        var queryHeads = UInt32(spec.headCount)
        encoder.setBytes(&queryHeads, length: MemoryLayout<UInt32>.stride,
                         index: 5)
        encoder.setThreadgroupMemoryLength(128 * MemoryLayout<Float>.stride,
                                           index: 0)
        encoder.dispatchThreadgroups(
            MTLSize(width: combinedHeads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
        )
    }

    private func encodeStoreKV(_ commands: Commands,
                               cache: LagunaMetalKVCache,
                               position: Int) throws {
        let pipeline = try runtime.pipeline("kernel_laguna_store_kv_f16")
        let encoder = commands.encoder
        encoder.setComputePipelineState(pipeline)
        setArguments(encoder, [
            UInt32(cache.capacity),
            UInt32(position % cache.capacity),
            1,
            UInt32(cache.rowWidth),
        ])
        encoder.setBuffer(keyRows, offset: 0, index: 1)
        encoder.setBuffer(valueRows, offset: 0, index: 2)
        encoder.setBuffer(cache.keys, offset: 0, index: 3)
        encoder.setBuffer(cache.values, offset: 0, index: 4)
        let threads = 256
        encoder.dispatchThreadgroups(
            MTLSize(width: (cache.rowWidth + threads - 1) / threads,
                    height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
    }

    private func encodeAttention(_ commands: Commands,
                                 spec: LagunaAttentionSpec,
                                 cache: LagunaMetalKVCache,
                                 position: Int) throws {
        let pipeline = try runtime.pipeline("kernel_laguna_attention_decode_gqa_f16")
        let encoder = commands.encoder
        encoder.setComputePipelineState(pipeline)
        let keyCount = min(position + 1, cache.capacity)
        let keyStart = position + 1 - keyCount
        var arguments = [UInt32](repeating: 0, count: 8)
        arguments[0] = UInt32(spec.headCount)
        arguments[1] = UInt32(spec.kvHeadCount)
        arguments[2] = UInt32(spec.headDim)
        arguments[3] = UInt32(cache.capacity)
        arguments[4] = UInt32(keyStart)
        arguments[5] = UInt32(keyCount)
        arguments[6] = (1 / Float(spec.headDim).squareRoot()).bitPattern
        setArguments(encoder, arguments)
        encoder.setBuffer(queryRows, offset: 0, index: 1)
        encoder.setBuffer(gateRows, offset: 0, index: 2)
        encoder.setBuffer(cache.keys, offset: 0, index: 3)
        encoder.setBuffer(cache.values, offset: 0, index: 4)
        encoder.setBuffer(heads, offset: 0, index: 5)
        encoder.setThreadgroupMemoryLength(
            (16 + 8 * spec.headDim) * MemoryLayout<Float>.stride, index: 0
        )
        encoder.dispatchThreadgroups(
            MTLSize(width: spec.headCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
    }

    // MARK: CPU helpers

    /// Q8_0 row dequantization (32-element blocks: F16 scale + 32 int8).
    static func dequantizeQ8Row(base: UnsafeRawPointer, row: Int,
                                rowCount: Int, width: Int) throws -> [Float] {
        guard row >= 0, row < rowCount, width % 32 == 0 else {
            throw LagunaResidentModelError.invalidToken(Int32(clamping: row))
        }
        let blockBytes = 34
        let blocks = width / 32
        let rowBytes = blocks * blockBytes
        let rowBase = base + row * rowBytes
        var out = [Float](repeating: 0, count: width)
        for block in 0..<blocks {
            let blockBase = rowBase + block * blockBytes
            let scaleBits = blockBase.loadUnaligned(as: UInt16.self)
            let scale = Half.float(scaleBits)
            let quants = (blockBase + 2).assumingMemoryBound(to: Int8.self)
            for i in 0..<32 {
                out[block * 32 + i] = scale * Float(quants[i])
            }
        }
        return out
    }
}
