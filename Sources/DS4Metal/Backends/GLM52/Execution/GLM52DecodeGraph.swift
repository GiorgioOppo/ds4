import DS4Core
import Foundation
import Metal

/// Cooperative-matvec dispatch policy: one SIMD group per row with lanes
/// striding the row's quant groups — the memory-bandwidth-friendly path the
/// telemetry demanded once I/O stalls stopped dominating. DS4_GLM_SG=0
/// reverts to the per-thread reference kernels; DS4_GLM_NSG picks the rows
/// (simdgroups) per threadgroup, default 4 = 128 threads.
enum GLM52MatvecDispatch {
    static let cooperative =
        ProcessInfo.processInfo.environment["DS4_GLM_SG"] != "0"
    static let rowsPerThreadgroup = max(1, min(8,
        ProcessInfo.processInfo.environment["DS4_GLM_NSG"]
            .flatMap(Int.init) ?? 4))
}

/// Telemetry of the synchronous graph commits — splits REAL GPU execution
/// from host overhead (encode, sync round-trips, router, readbacks) in the
/// streaming report. Accumulated on the decode thread that owns the graph;
/// nonisolated(unsafe) is acceptable for counters read only by the report.
enum GLM52GraphTelemetry {
    nonisolated(unsafe) static var gpuSeconds = 0.0
    nonisolated(unsafe) static var commits = 0

    static func reset() {
        gpuSeconds = 0
        commits = 0
    }
}

// Resident decode graph — the persistent form of glm52DecodeAttention. The
// per-dispatch executor re-uploads every weight array on every call; here the
// quantized weights are uploaded ONCE into MTLBuffers, the compact and
// indexer caches live on the GPU and are appended in place, and the whole
// attention step is encoded on chained buffers: in the fill-range path the
// entire step is ONE command buffer; the top-k path splits only around the
// score readback that feeds the host-orchestrated multi-block top-k. The
// CPU keeps exactly what the per-dispatch executor kept host-side minus the
// norms (now kernel_glm52_rms_norm_f32): the residual adds, the router and
// the 32-row F32 indexer-proj matvec. Correctness anchor: parity with
// glm52DecodeAttention, which is itself judged by GLM52DecodeCPUReference.

/// Quantization types of the big streamed tensors consumed by the chained
/// decode's type-parametric matvec kernels. Q8_0 everywhere on the plain
/// GGUF path; the Q4_K layer sidecar overrides the requantized ones.
/// keyB/valueB have no entry by contract: their kernels are Q8-hardwired
/// and the sidecar carries them verbatim.
public struct GLM52StreamedWeightTypes: Sendable {
    public var qA = GLM52TensorSchema.q8_0
    public var qB = GLM52TensorSchema.q8_0
    public var kvA = GLM52TensorSchema.q8_0
    public var attnOutput = GLM52TensorSchema.q8_0
    public var indexerKey = GLM52TensorSchema.q8_0
    public var indexerQueryB = GLM52TensorSchema.q8_0
    public var sharedGateUp = GLM52TensorSchema.q8_0
    public var sharedDown = GLM52TensorSchema.q8_0
    public init() {}
}

/// One layer's decode weights resident on the GPU. Upload happens once at
/// construction; the buffers are shared-storage so fixtures stay comparable.
public final class GLM52ResidentDecodeWeights {
    public let geometry: GLM52DecodeGeometry
    /// Per-tensor quantization the chained decode passes to the kernels.
    let types: GLM52StreamedWeightTypes
    let attnNorm: MTLBuffer
    let qA: MTLBuffer
    let qANorm: MTLBuffer
    let qB: MTLBuffer
    let kvA: MTLBuffer
    let kvANorm: MTLBuffer
    let keyB: MTLBuffer
    let valueB: MTLBuffer
    let attnOutput: MTLBuffer

    struct ResidentIndexer {
        let key: MTLBuffer
        let keyNorm: MTLBuffer
        let keyNormBias: MTLBuffer
        let queryB: MTLBuffer
        /// The 32-row F32 matvec stays on CPU beside the router.
        let proj: [Float]
    }
    let indexer: ResidentIndexer?
    public var isFullIndexer: Bool { indexer != nil }

    /// Compose a weights view over EXISTING buffers — the streaming path
    /// builds one per layer step from per-layer norm buffers plus reused
    /// staging slots. No validation and no copies: the caller (the layer
    /// streamer) owns both.
    init(geometry: GLM52DecodeGeometry,
         attnNorm: MTLBuffer, qA: MTLBuffer, qANorm: MTLBuffer,
         qB: MTLBuffer, kvA: MTLBuffer, kvANorm: MTLBuffer,
         keyB: MTLBuffer, valueB: MTLBuffer, attnOutput: MTLBuffer,
         indexer: ResidentIndexer?,
         types: GLM52StreamedWeightTypes = GLM52StreamedWeightTypes()) {
        self.geometry = geometry
        self.types = types
        self.attnNorm = attnNorm
        self.qA = qA
        self.qANorm = qANorm
        self.qB = qB
        self.kvA = kvA
        self.kvANorm = kvANorm
        self.keyB = keyB
        self.valueB = valueB
        self.attnOutput = attnOutput
        self.indexer = indexer
    }

    public init(runtime: MetalRuntime,
                geometry: GLM52DecodeGeometry,
                attention: GLM52QuantizedDecodeAttention,
                indexer quantizedIndexer: GLM52QuantizedDecodeIndexer?) throws {
        try runtime.glm52ValidateDecodeWeights(
            geometry: geometry, attention: attention,
            indexer: quantizedIndexer)
        self.geometry = geometry
        types = GLM52StreamedWeightTypes()
        attnNorm = try runtime.glm52GraphBuffer(attention.attnNorm)
        qA = try runtime.glm52GraphBuffer(attention.qA)
        qANorm = try runtime.glm52GraphBuffer(attention.qANorm)
        qB = try runtime.glm52GraphBuffer(attention.qB)
        kvA = try runtime.glm52GraphBuffer(attention.kvA)
        kvANorm = try runtime.glm52GraphBuffer(attention.kvANorm)
        keyB = try runtime.glm52GraphBuffer(attention.keyB)
        valueB = try runtime.glm52GraphBuffer(attention.valueB)
        attnOutput = try runtime.glm52GraphBuffer(attention.attnOutput)
        if let quantizedIndexer {
            indexer = ResidentIndexer(
                key: try runtime.glm52GraphBuffer(quantizedIndexer.key),
                keyNorm: try runtime.glm52GraphBuffer(quantizedIndexer.keyNorm),
                keyNormBias: try runtime.glm52GraphBuffer(
                    quantizedIndexer.keyNormBias),
                queryB: try runtime.glm52GraphBuffer(quantizedIndexer.queryB),
                proj: quantizedIndexer.proj)
        } else {
            indexer = nil
        }
    }
}

/// One layer's decode caches resident on the GPU: interleaved compact rows
/// (`[capacity][576]` F16) and — on full-indexer layers — the indexer key
/// plane (`[capacity][128]` F16), appended in place by the graph.
public final class GLM52ResidentDecodeCaches {
    public let capacity: Int
    public private(set) var rows: Int = 0
    let compact: MTLBuffer
    let indexerKeys: MTLBuffer?
    private let geometry: GLM52DecodeGeometry

    public init(runtime: MetalRuntime,
                geometry: GLM52DecodeGeometry,
                capacity: Int,
                fullIndexer: Bool) throws {
        guard capacity > 0, capacity <= Int(UInt32.max) else {
            throw MetalError.unsupported(
                "GLM 5.2 resident cache capacity \(capacity) is invalid")
        }
        self.capacity = capacity
        self.geometry = geometry
        let rowWidth = geometry.layer.kvLoraRank + geometry.layer.ropeDimension
        guard let compactBuffer = runtime.device.makeBuffer(
            length: capacity * rowWidth * MemoryLayout<UInt16>.stride,
            options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        compact = compactBuffer
        if fullIndexer {
            guard let keyBuffer = runtime.device.makeBuffer(
                length: capacity * geometry.indexerHeadDimension
                    * MemoryLayout<UInt16>.stride,
                options: .storageModeShared) else {
                throw MetalError.bufferAlloc
            }
            indexerKeys = keyBuffer
        } else {
            indexerKeys = nil
        }
    }

    func appendedRow() { rows += 1 }

    /// Forget every live row (fresh conversation). Buffer contents beyond
    /// the live count are never read, so no clearing is needed.
    public func reset() { rows = 0 }

    /// The live compact rows as F16 bits — for tests and checkpoints.
    public func compactSnapshot() -> [UInt16] {
        let count = rows * (geometry.layer.kvLoraRank
                                + geometry.layer.ropeDimension)
        let pointer = compact.contents().bindMemory(
            to: UInt16.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    /// The live indexer key rows as F16 bits (empty on IndexShare layers).
    public func indexerKeySnapshot() -> [UInt16] {
        guard let indexerKeys else { return [] }
        let count = rows * geometry.indexerHeadDimension
        let pointer = indexerKeys.contents().bindMemory(
            to: UInt16.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }
}

/// One layer's FFN resident on the GPU: the norm plus dense or shared
/// weights uploaded once. Routed experts stay a per-token stream (the
/// provider yields the selected experts' bytes); the F32 router rows stay
/// host-side beside the CPU router matvec.
/// One routed selection staged for the GPU: every selected record resident
/// in ONE shared arena buffer at its own offset (packed gate|up|down),
/// ready to bind — no per-expert allocations, no copies. Offsets are
/// per-record because the arena keeps records wherever its LRU placed
/// them (a hit reuses a slot filled tokens ago).
public struct GLM52StagedExpertSelection {
    public let buffer: MTLBuffer
    /// Byte offset of each rank's record in `buffer` (rank order of the
    /// staged ids).
    public let recordOffsets: [Int]
    public let gateBytes: Int
    public let upBytes: Int
    public let downBytes: Int
    public let gateUpType: UInt32
    public let downType: UInt32

    public init(buffer: MTLBuffer, recordOffsets: [Int], gateBytes: Int,
                upBytes: Int, downBytes: Int,
                gateUpType: UInt32, downType: UInt32) {
        self.buffer = buffer
        self.recordOffsets = recordOffsets
        self.gateBytes = gateBytes
        self.upBytes = upBytes
        self.downBytes = downBytes
        self.gateUpType = gateUpType
        self.downType = downType
    }
}

public final class GLM52ResidentFFN {
    enum Kind {
        case dense(gate: MTLBuffer, up: MTLBuffer, down: MTLBuffer)
        case sparse(routerRows: [Float], routerBias: [Float],
                    sharedGate: MTLBuffer, sharedUp: MTLBuffer,
                    sharedDown: MTLBuffer,
                    expertProvider: (UInt32) throws -> GLM52QuantizedExpert)
    }
    let ffnNorm: MTLBuffer
    let kind: Kind
    /// Optional batched zero-copy expert path: when set, the chained decode
    /// stages the WHOLE selection through this closure (one concurrent read
    /// into one reusable buffer, bound by offsets) instead of walking
    /// `expertProvider` serially with per-expert copies. The provider stays
    /// as constructed — it remains the reference and fallback path.
    var stagedSelection: (([UInt32]) throws -> GLM52StagedExpertSelection)?
    /// Shared-expert quantization (Q8_0 unless the Q4_K layer sidecar
    /// requantized this layer's shared tensors).
    var sharedWeightTypes: (gateUp: UInt32, down: UInt32) =
        (GLM52TensorSchema.q8_0, GLM52TensorSchema.q8_0)

    /// Compose an FFN view over EXISTING buffers (streaming path — see the
    /// weights counterpart). No validation and no copies.
    init(ffnNorm: MTLBuffer, kind: Kind) {
        self.ffnNorm = ffnNorm
        self.kind = kind
    }

    public init(runtime: MetalRuntime,
                geometry: GLM52DecodeGeometry,
                ffnNorm: [Float],
                ffn: GLM52QuantizedLayerFFN) throws {
        let layer = geometry.layer
        let embedBytes = MetalRuntime.glm52Q8RowBytes(layer.embeddingWidth)
        func require(_ got: Int, _ expected: Int,
                     _ component: String) throws {
            guard got == expected else {
                throw MetalError.unsupported(
                    "GLM 5.2 resident FFN \(component) has \(got) elements, "
                    + "expected \(expected)")
            }
        }
        try require(ffnNorm.count, layer.embeddingWidth, "ffnNorm")
        self.ffnNorm = try runtime.glm52GraphBuffer(ffnNorm)
        switch ffn {
        case .dense(let gate, let up, let down):
            let hiddenBytes = MetalRuntime.glm52Q8RowBytes(
                layer.denseHiddenWidth)
            try require(gate.count, layer.denseHiddenWidth * embedBytes,
                        "dense gate")
            try require(up.count, layer.denseHiddenWidth * embedBytes,
                        "dense up")
            try require(down.count, layer.embeddingWidth * hiddenBytes,
                        "dense down")
            kind = .dense(gate: try runtime.glm52GraphBuffer(gate),
                          up: try runtime.glm52GraphBuffer(up),
                          down: try runtime.glm52GraphBuffer(down))
        case .sparse(let routerRows, let routerBias, let sharedGate,
                     let sharedUp, let sharedDown, let expertProvider):
            let hiddenBytes = MetalRuntime.glm52Q8RowBytes(
                layer.expertHiddenWidth)
            try require(routerRows.count,
                        GLM52RouterReference.expertCount
                            * layer.embeddingWidth, "router rows")
            try require(routerBias.count,
                        GLM52RouterReference.expertCount, "router bias")
            try require(sharedGate.count,
                        layer.expertHiddenWidth * embedBytes, "shared gate")
            try require(sharedUp.count,
                        layer.expertHiddenWidth * embedBytes, "shared up")
            try require(sharedDown.count,
                        layer.embeddingWidth * hiddenBytes, "shared down")
            kind = .sparse(routerRows: routerRows, routerBias: routerBias,
                           sharedGate: try runtime.glm52GraphBuffer(sharedGate),
                           sharedUp: try runtime.glm52GraphBuffer(sharedUp),
                           sharedDown: try runtime.glm52GraphBuffer(sharedDown),
                           expertProvider: expertProvider)
        }
    }
}

/// The output head resident on the GPU: final RMSNorm weight and the Q8_0
/// vocabulary matvec rows, uploaded once.
public final class GLM52ResidentOutputHead {
    let norm: MTLBuffer
    let head: MTLBuffer
    public let vocabularySize: Int
    let embeddingWidth: Int

    public init(runtime: MetalRuntime,
                geometry: GLM52DecodeGeometry,
                outputNorm: [Float],
                outputHead: [UInt8],
                vocabularySize: Int) throws {
        let embed = geometry.layer.embeddingWidth
        guard outputNorm.count == embed, vocabularySize > 0,
              outputHead.count == vocabularySize
                  * MetalRuntime.glm52Q8RowBytes(embed) else {
            throw MetalError.unsupported(
                "GLM 5.2 resident output head expects [vocab] Q8_0 rows of "
                + "\(embed) and a matching norm")
        }
        norm = try runtime.glm52GraphBuffer(outputNorm)
        head = try runtime.glm52GraphBuffer(outputHead)
        self.vocabularySize = vocabularySize
        embeddingWidth = embed
    }
}

/// One resident decode layer of the stack: the ABSOLUTE layer index drives
/// the IndexShare role through `GLM52IndexSharePolicy`.
public struct GLM52ResidentStackLayer {
    public let index: Int
    public let weights: GLM52ResidentDecodeWeights
    public let ffn: GLM52ResidentFFN
    public let caches: GLM52ResidentDecodeCaches

    public init(index: Int, weights: GLM52ResidentDecodeWeights,
                ffn: GLM52ResidentFFN, caches: GLM52ResidentDecodeCaches) {
        self.index = index
        self.weights = weights
        self.ffn = ffn
        self.caches = caches
    }
}

extension MetalRuntime {
    // MARK: - Buffer helpers

    func glm52GraphBuffer(_ values: [Float]) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(
            bytes: values,
            length: values.count * MemoryLayout<Float>.stride,
            options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        return buffer
    }

    func glm52GraphBuffer(_ bytes: [UInt8]) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(
            bytes: bytes, length: bytes.count,
            options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        return buffer
    }

    func glm52GraphOutputBuffer(floats count: Int) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(
            length: count * MemoryLayout<Float>.stride,
            options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        return buffer
    }

    func glm52GraphReadback(_ buffer: MTLBuffer,
                                    count: Int) -> [Float] {
        let pointer = buffer.contents().bindMemory(
            to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    /// Encode one kernel dispatch into an open command buffer. Sequential
    /// encoders in one command buffer execute in order with automatic hazard
    /// tracking on the shared buffers.
    func glm52GraphEncode(
        into commandBuffer: MTLCommandBuffer,
        pipelineName: String,
        arguments: [UInt32],
        buffers: [MTLBuffer],
        offsets: [Int]? = nil,
        threadgroups: MTLSize,
        threadsPerThreadgroup: MTLSize,
        threadgroupMemoryLength: Int? = nil) throws {
        let pipeline = try pipeline(pipelineName)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        encoder.setComputePipelineState(pipeline)
        arguments.withUnsafeBytes {
            encoder.setBytes($0.baseAddress!, length: $0.count, index: 0)
        }
        for (index, buffer) in buffers.enumerated() {
            encoder.setBuffer(buffer, offset: offsets?[index] ?? 0,
                              index: index + 1)
        }
        if let length = threadgroupMemoryLength {
            // Metal API validation aborts on lengths that are not multiples
            // of 16 bytes (Xcode-run builds have it on; CLI runs do not —
            // which is why the test suite never tripped this).
            encoder.setThreadgroupMemoryLength((length + 15) & ~15, index: 0)
        }
        encoder.dispatchThreadgroups(
            threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }

    func glm52GraphCommit(_ commandBuffer: MTLCommandBuffer) throws {
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        GLM52GraphTelemetry.gpuSeconds += max(
            0, commandBuffer.gpuEndTime - commandBuffer.gpuStartTime)
        GLM52GraphTelemetry.commits += 1
        if let error = commandBuffer.error { throw error }
    }

    // MARK: - Encoded stages

    func glm52EncodeRMSNorm(into commandBuffer: MTLCommandBuffer,
                                    input: MTLBuffer, weight: MTLBuffer,
                                    output: MTLBuffer, width: Int,
                                    epsilon: Float = 1e-5) throws {
        try glm52GraphEncode(
            into: commandBuffer, pipelineName: "kernel_glm52_rms_norm_f32",
            arguments: [UInt32(width), epsilon.bitPattern, 0, 0],
            buffers: [input, weight, output],
            threadgroups: MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1),
            threadgroupMemoryLength: 256 * MemoryLayout<Float>.stride)
    }

    func glm52EncodeMatvecQ8(
        into commandBuffer: MTLCommandBuffer,
        input: MTLBuffer, weights: MTLBuffer,
        output: MTLBuffer, rowCount: Int, inputWidth: Int,
        weightType: UInt32 = GLM52TensorSchema.q8_0,
        weightsOffset: Int = 0) throws {
        if GLM52MatvecDispatch.cooperative {
            let rows = GLM52MatvecDispatch.rowsPerThreadgroup
            try glm52GraphEncode(
                into: commandBuffer,
                pipelineName: "kernel_glm52_moe_down_sg",
                arguments: [weightType, UInt32(rowCount),
                            UInt32(inputWidth), Float(1).bitPattern],
                buffers: [input, weights, output],
                offsets: [0, weightsOffset, 0],
                threadgroups: MTLSize(width: (rowCount + rows - 1) / rows,
                                      height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: rows,
                                               depth: 1))
            return
        }
        let width = 256
        try glm52GraphEncode(
            into: commandBuffer, pipelineName: "kernel_glm52_moe_down",
            arguments: [weightType, UInt32(rowCount),
                        UInt32(inputWidth), Float(1).bitPattern],
            buffers: [input, weights, output],
            offsets: [0, weightsOffset, 0],
            threadgroups: MTLSize(width: (rowCount + width - 1) / width,
                                  height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }

    func glm52EncodePairSwiGLU(
        into commandBuffer: MTLCommandBuffer,
        input: MTLBuffer, gate: MTLBuffer, up: MTLBuffer, mid: MTLBuffer,
        hiddenWidth: Int, inputWidth: Int, routeWeight: Float,
        weightType: UInt32 = GLM52TensorSchema.q8_0,
        gateOffset: Int = 0, upOffset: Int = 0) throws {
        if GLM52MatvecDispatch.cooperative {
            let rows = GLM52MatvecDispatch.rowsPerThreadgroup
            try glm52GraphEncode(
                into: commandBuffer,
                pipelineName: "kernel_glm52_moe_pair_swiglu_sg",
                arguments: [weightType, UInt32(hiddenWidth),
                            UInt32(inputWidth), routeWeight.bitPattern],
                buffers: [input, gate, up, mid],
                offsets: [0, gateOffset, upOffset, 0],
                threadgroups: MTLSize(
                    width: (hiddenWidth + rows - 1) / rows,
                    height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: rows,
                                               depth: 1))
            return
        }
        let width = 256
        try glm52GraphEncode(
            into: commandBuffer,
            pipelineName: "kernel_glm52_moe_pair_swiglu",
            arguments: [weightType, UInt32(hiddenWidth),
                        UInt32(inputWidth), routeWeight.bitPattern],
            buffers: [input, gate, up, mid],
            offsets: [0, gateOffset, upOffset, 0],
            threadgroups: MTLSize(width: (hiddenWidth + width - 1) / width,
                                  height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }

    func glm52EncodeAdd(into commandBuffer: MTLCommandBuffer,
                                a: MTLBuffer, b: MTLBuffer,
                                output: MTLBuffer, count: Int) throws {
        let width = 256
        try glm52GraphEncode(
            into: commandBuffer, pipelineName: "kernel_glm52_add_f32",
            arguments: [UInt32(count), 0, 0, 0],
            buffers: [a, b, output],
            threadgroups: MTLSize(width: (count + width - 1) / width,
                                  height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }

    func glm52EncodeRope(into commandBuffer: MTLCommandBuffer,
                                 pipelineName: String, values: MTLBuffer,
                                 headCount: Int, headDimension: Int,
                                 rotationDimension: Int,
                                 position: Int) throws {
        let pairs = headCount * (rotationDimension / 2)
        let width = 256
        try glm52GraphEncode(
            into: commandBuffer, pipelineName: pipelineName,
            arguments: [UInt32(headCount), UInt32(headDimension),
                        UInt32(rotationDimension), UInt32(position)],
            buffers: [values],
            threadgroups: MTLSize(width: (pairs + width - 1) / width,
                                  height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }

    // MARK: - Resident decode attention

    /// One decode attention step on resident weights and caches. Encoded on
    /// chained GPU buffers: the fill-range path is a single command buffer;
    /// the top-k path splits around the score readback feeding the
    /// host-orchestrated multi-block top-k. Appends this token's cache rows
    /// in place (before selection and attention, upstream's order). On error
    /// the caches are unspecified.
    public func glm52ResidentDecodeAttention(
        weights: GLM52ResidentDecodeWeights,
        caches: GLM52ResidentDecodeCaches,
        input: [Float],
        reusedSelection: [UInt32]?,
        position: Int) throws -> (output: [Float], selection: [UInt32]) {
        let g = weights.geometry
        let layer = g.layer
        let headsWidth = layer.headCount * layer.valueDimension
        let visible = position + 1

        guard input.count == layer.embeddingWidth else {
            throw MetalError.unsupported(
                "GLM 5.2 resident decode input must be "
                + "\(layer.embeddingWidth) wide")
        }
        guard position == caches.rows, visible <= caches.capacity else {
            throw MetalError.unsupported(
                "GLM 5.2 resident decode position \(position) does not "
                + "match \(caches.rows) live rows / capacity "
                + "\(caches.capacity)")
        }
        if weights.indexer != nil {
            guard reusedSelection == nil, caches.indexerKeys != nil else {
                throw MetalError.unsupported(
                    "full-indexer resident layer needs indexer caches and "
                    + "no reused selection")
            }
        } else {
            guard reusedSelection != nil, caches.indexerKeys == nil else {
                throw MetalError.unsupported(
                    "IndexShare resident layer requires the preceding "
                    + "full-indexer selection and owns no indexer keys")
            }
        }

        // Activation buffers for this step.
        let x = try glm52GraphBuffer(input)
        let normed = try glm52GraphOutputBuffer(floats: layer.embeddingWidth)
        let qRank = try glm52GraphOutputBuffer(floats: g.qLoraRank)
        let qRankNorm = try glm52GraphOutputBuffer(floats: g.qLoraRank)
        let kvRaw = try glm52GraphOutputBuffer(floats: layer.kvRawWidth)
        let cacheReady = try glm52GraphOutputBuffer(floats: layer.kvRawWidth)
        let query = try glm52GraphOutputBuffer(floats: g.queryWidth)
        let qLow = try glm52GraphOutputBuffer(
            floats: layer.headCount * layer.kvLoraRank)
        let attnLora = try glm52GraphOutputBuffer(
            floats: layer.headCount * layer.kvLoraRank)
        let heads = try glm52GraphOutputBuffer(floats: headsWidth)
        let output = try glm52GraphOutputBuffer(floats: layer.embeddingWidth)

        guard let stepBuffer = queue.makeCommandBuffer() else {
            throw MetalError.bufferAlloc
        }

        // Shared trunk: norms, LoRA projections, cache stores, query RoPE.
        try glm52EncodeRMSNorm(into: stepBuffer, input: x,
                               weight: weights.attnNorm, output: normed,
                               width: layer.embeddingWidth)
        try glm52EncodeMatvecQ8(into: stepBuffer, input: normed,
                                weights: weights.qA, output: qRank,
                                rowCount: g.qLoraRank,
                                inputWidth: layer.embeddingWidth)
        try glm52EncodeMatvecQ8(into: stepBuffer, input: normed,
                                weights: weights.kvA, output: kvRaw,
                                rowCount: layer.kvRawWidth,
                                inputWidth: layer.embeddingWidth)
        try glm52EncodeRMSNorm(into: stepBuffer, input: qRank,
                               weight: weights.qANorm, output: qRankNorm,
                               width: g.qLoraRank)
        try glm52GraphEncode(
            into: stepBuffer,
            pipelineName: "kernel_glm52_kv_lora_norm_cache_ready_f32",
            arguments: [1, 0, 0, 0],
            buffers: [kvRaw, weights.kvANorm, cacheReady],
            threadgroups: MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1),
            threadgroupMemoryLength: 128 * MemoryLayout<Float>.stride)
        try glm52GraphEncode(
            into: stepBuffer,
            pipelineName: "kernel_glm52_store_compact_row_f16",
            arguments: [UInt32(position), 1, UInt32(caches.capacity), 0],
            buffers: [cacheReady, caches.compact],
            threadgroups: MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        try glm52EncodeMatvecQ8(into: stepBuffer, input: qRankNorm,
                                weights: weights.qB, output: query,
                                rowCount: g.queryWidth,
                                inputWidth: g.qLoraRank)
        try glm52EncodeRope(into: stepBuffer,
                            pipelineName: "kernel_glm52_rope_tail_f32",
                            values: query, headCount: layer.headCount,
                            headDimension: g.qkDimension,
                            rotationDimension: layer.ropeDimension,
                            position: position)

        // Indexer store, then the selection: encoded score path only when
        // the visible range exceeds top-k.
        var selection: [UInt32]
        var scores: MTLBuffer?
        if let indexer = weights.indexer, let keyCache = caches.indexerKeys {
            let idxRaw = try glm52GraphOutputBuffer(
                floats: g.indexerHeadDimension)
            try glm52EncodeMatvecQ8(into: stepBuffer, input: x,
                                    weights: indexer.key, output: idxRaw,
                                    rowCount: g.indexerHeadDimension,
                                    inputWidth: layer.embeddingWidth)
            try glm52GraphEncode(
                into: stepBuffer,
                pipelineName: "kernel_glm52_store_indexer_k_f16",
                arguments: [UInt32(position), 1, UInt32(caches.capacity), 0],
                buffers: [idxRaw, indexer.keyNorm, indexer.keyNormBias,
                          keyCache],
                threadgroups: MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1),
                threadgroupMemoryLength: 32 * MemoryLayout<Float>.stride)
            if visible <= g.indexerTopK {
                selection = (0..<visible).map(UInt32.init)
            } else {
                let indexerQuery = try glm52GraphOutputBuffer(
                    floats: g.indexerQueryWidth)
                try glm52EncodeMatvecQ8(into: stepBuffer, input: qRankNorm,
                                        weights: indexer.queryB,
                                        output: indexerQuery,
                                        rowCount: g.indexerQueryWidth,
                                        inputWidth: g.qLoraRank)
                try glm52EncodeRope(
                    into: stepBuffer,
                    pipelineName: "kernel_glm52_rope_prefix_f32",
                    values: indexerQuery, headCount: g.indexerHeadCount,
                    headDimension: g.indexerHeadDimension,
                    rotationDimension: g.indexerRotationDimension,
                    position: position)
                var headWeights = [Float](
                    repeating: 0, count: g.indexerHeadCount)
                for head in 0..<g.indexerHeadCount {
                    var dot: Float = 0
                    let base = head * layer.embeddingWidth
                    for i in 0..<layer.embeddingWidth {
                        dot += indexer.proj[base + i] * input[i]
                    }
                    headWeights[head] = dot
                }
                let scoreBuffer = try glm52GraphOutputBuffer(floats: visible)
                try glm52GraphEncode(
                    into: stepBuffer,
                    pipelineName: "kernel_glm52_indexer_scores_f16",
                    arguments: [UInt32(visible), 1, UInt32(position),
                                g.indexerScale.bitPattern],
                    buffers: [indexerQuery,
                              try glm52GraphBuffer(headWeights),
                              keyCache, scoreBuffer],
                    threadgroups: MTLSize(width: visible, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 4,
                                                   depth: 1),
                    threadgroupMemoryLength:
                        (128 + 4) * MemoryLayout<Float>.stride)
                scores = scoreBuffer
                selection = []
            }
        } else {
            // Validated above: IndexShare layers carry the reused selection.
            selection = reusedSelection ?? []
        }

        // The top-k path must materialize the scores before selecting; the
        // fill-range path continues in the same command buffer.
        if let scores {
            try glm52GraphCommit(stepBuffer)
            let hostScores = glm52GraphReadback(scores, count: visible)
            selection = try glm52IndexerTopK(
                scores: hostScores, rowCount: visible, tokenCount: 1,
                topK: g.indexerTopK)
        }
        guard !selection.isEmpty,
              selection.count <= visible,
              Set(selection).count == selection.count,
              selection.allSatisfy({ Int($0) < visible }) else {
            throw MetalError.unsupported(
                "GLM 5.2 resident decode selection must be unique rows "
                + "inside 0..<\(visible)")
        }
        guard let selectionBuffer = device.makeBuffer(
            bytes: selection,
            length: selection.count * MemoryLayout<UInt32>.stride,
            options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }

        let attentionBuffer: MTLCommandBuffer
        if scores != nil {
            guard let fresh = queue.makeCommandBuffer() else {
                throw MetalError.bufferAlloc
            }
            attentionBuffer = fresh
        } else {
            attentionBuffer = stepBuffer
        }

        // Attention tail: absorb, indexed softmax with per-row tail
        // rotation over the resident cache, value projection, output.
        let attentionGeometry = GLM52AttentionGeometry.v5_2
        try glm52GraphEncode(
            into: attentionBuffer,
            pipelineName: "kernel_glm52_qk_lowrank_q8_0",
            arguments: [0, 0, 0, 0],
            buffers: [query, weights.keyB, qLow],
            threadgroups: MTLSize(width: layer.kvLoraRank / 4,
                                  height: layer.headCount, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
        try glm52GraphEncode(
            into: attentionBuffer,
            pipelineName: "kernel_glm52_attention_indexed_f16",
            arguments: [UInt32(visible), UInt32(selection.count),
                        attentionGeometry.scale.bitPattern, 1],
            buffers: [qLow, query, caches.compact, selectionBuffer, attnLora],
            threadgroups: MTLSize(width: layer.headCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1),
            threadgroupMemoryLength:
                (selection.count + 5) * MemoryLayout<Float>.stride)
        try glm52GraphEncode(
            into: attentionBuffer,
            pipelineName: "kernel_glm52_value_project_q8_0",
            arguments: [0, 0, 0, 0],
            buffers: [attnLora, weights.valueB, heads],
            threadgroups: MTLSize(width: layer.valueDimension / 4,
                                  height: layer.headCount, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
        try glm52EncodeMatvecQ8(into: attentionBuffer, input: heads,
                                weights: weights.attnOutput, output: output,
                                rowCount: layer.embeddingWidth,
                                inputWidth: headsWidth)
        try glm52GraphCommit(attentionBuffer)

        caches.appendedRow()
        return (glm52GraphReadback(output, count: layer.embeddingWidth),
                selection)
    }

    /// One full decode layer on resident state: attention residual, then the
    /// resident FFN stage. Dense layers and the shared expert run entirely
    /// on resident buffers; sparse layers tap `ffnIn` back to the host once
    /// for the F32 router and stream the selected experts' bytes per token
    /// (uploads that are inherent to streaming, not residency gaps).
    /// `activeExperts` caps the routed experts actually executed (rank
    /// order, weights untouched): less expert I/O, lower quality — the GLM
    /// analog of the DeepSeek DS4_ACTIVE_EXPERTS knob.
    public func glm52ResidentDecodeLayer(
        weights: GLM52ResidentDecodeWeights,
        ffn: GLM52ResidentFFN,
        caches: GLM52ResidentDecodeCaches,
        input: [Float],
        reusedSelection: [UInt32]?,
        position: Int,
        activeExperts: Int? = nil) throws
        -> (output: [Float], routing: GLM52RouterOutput?,
            selection: [UInt32]) {
        let geometry = weights.geometry
        let embed = geometry.layer.embeddingWidth
        let attn = try glm52ResidentDecodeAttention(
            weights: weights, caches: caches, input: input,
            reusedSelection: reusedSelection, position: position)
        let afterAttn = (0..<embed).map { input[$0] + attn.output[$0] }

        let afterAttnBuffer = try glm52GraphBuffer(afterAttn)
        let ffnIn = try glm52GraphOutputBuffer(floats: embed)
        let output = try glm52GraphOutputBuffer(floats: embed)

        switch ffn.kind {
        case .dense(let gate, let up, let down):
            let mid = try glm52GraphOutputBuffer(
                floats: geometry.layer.denseHiddenWidth)
            let ffnOut = try glm52GraphOutputBuffer(floats: embed)
            guard let commandBuffer = queue.makeCommandBuffer() else {
                throw MetalError.bufferAlloc
            }
            try glm52EncodeRMSNorm(into: commandBuffer, input: afterAttnBuffer,
                                   weight: ffn.ffnNorm, output: ffnIn,
                                   width: embed)
            try glm52EncodePairSwiGLU(
                into: commandBuffer, input: ffnIn, gate: gate, up: up,
                mid: mid, hiddenWidth: geometry.layer.denseHiddenWidth,
                inputWidth: embed, routeWeight: 1)
            try glm52EncodeMatvecQ8(
                into: commandBuffer, input: mid, weights: down,
                output: ffnOut, rowCount: embed,
                inputWidth: geometry.layer.denseHiddenWidth)
            try glm52EncodeAdd(into: commandBuffer, a: afterAttnBuffer,
                               b: ffnOut, output: output, count: embed)
            try glm52GraphCommit(commandBuffer)
            return (glm52GraphReadback(output, count: embed), nil,
                    attn.selection)

        case .sparse(let routerRows, let routerBias, let sharedGate,
                     let sharedUp, let sharedDown, let expertProvider):
            // Stage 1: the FFN norm, read back once for the F32 router.
            guard let normBuffer = queue.makeCommandBuffer() else {
                throw MetalError.bufferAlloc
            }
            try glm52EncodeRMSNorm(into: normBuffer, input: afterAttnBuffer,
                                   weight: ffn.ffnNorm, output: ffnIn,
                                   width: embed)
            try glm52GraphCommit(normBuffer)
            let ffnInHost = glm52GraphReadback(ffnIn, count: embed)
            let logits = try GLM52FFNCPUReference.matvec(
                rows: routerRows, input: ffnInHost,
                rowCount: GLM52RouterReference.expertCount)
            let routed = try glm52Route(logits: logits, bias: routerBias)

            // Stage 2: shared expert plus the streamed routed experts, all
            // accumulated on GPU (out = afterAttn + shared + sum experts).
            let hidden = geometry.layer.expertHiddenWidth
            let mid = try glm52GraphOutputBuffer(floats: hidden)
            let contribution = try glm52GraphOutputBuffer(floats: embed)
            guard let ffnBuffer = queue.makeCommandBuffer() else {
                throw MetalError.bufferAlloc
            }
            try glm52EncodePairSwiGLU(
                into: ffnBuffer, input: ffnIn, gate: sharedGate,
                up: sharedUp, mid: mid, hiddenWidth: hidden,
                inputWidth: embed, routeWeight: 1)
            try glm52EncodeMatvecQ8(
                into: ffnBuffer, input: mid, weights: sharedDown,
                output: contribution, rowCount: embed, inputWidth: hidden)
            try glm52EncodeAdd(into: ffnBuffer, a: afterAttnBuffer,
                               b: contribution, output: output, count: embed)
            let usedExperts = min(routed.selected.count,
                                  max(1, activeExperts ?? routed.selected.count))
            for rank in 0..<usedExperts {
                let expert = routed.selected[rank]
                let record = try expertProvider(UInt32(bitPattern: expert))
                let gate = try glm52GraphBuffer(record.gate)
                let up = try glm52GraphBuffer(record.up)
                let down = try glm52GraphBuffer(record.down)
                try glm52EncodePairSwiGLU(
                    into: ffnBuffer, input: ffnIn, gate: gate, up: up,
                    mid: mid, hiddenWidth: hidden, inputWidth: embed,
                    routeWeight: routed.weights[rank],
                    weightType: record.gateUpType)
                try glm52EncodeMatvecQ8(
                    into: ffnBuffer, input: mid, weights: down,
                    output: contribution, rowCount: embed,
                    inputWidth: hidden, weightType: record.downType)
                try glm52EncodeAdd(into: ffnBuffer, a: output,
                                   b: contribution, output: output,
                                   count: embed)
            }
            try glm52GraphCommit(ffnBuffer)
            return (glm52GraphReadback(output, count: embed), routed,
                    attn.selection)
        }
    }

    /// One decode token through a stack of resident layers in order, with
    /// the REAL IndexShare threading: full-indexer layers (per
    /// `GLM52IndexSharePolicy` on the absolute layer index) compute and
    /// publish the selection; the layers in between must present the
    /// expected source layer and reuse it verbatim. Returns the logits from
    /// the resident output head plus per-layer selections and routings.
    public func glm52ResidentDecodeForward(
        layers: [GLM52ResidentStackLayer],
        outputHead: GLM52ResidentOutputHead,
        embeddedToken: [Float],
        position: Int) throws
        -> (logits: [Float], selections: [Int: [UInt32]],
            routings: [Int: GLM52RouterOutput]) {
        guard !layers.isEmpty else {
            throw MetalError.unsupported(
                "GLM 5.2 resident forward requires at least one layer")
        }
        var hidden = embeddedToken
        var lastSelection: (source: Int, rows: [UInt32])?
        var selections: [Int: [UInt32]] = [:]
        var routings: [Int: GLM52RouterOutput] = [:]

        for layer in layers {
            let isFull = GLM52IndexSharePolicy.isFullIndexerLayer(layer.index)
            guard isFull == layer.weights.isFullIndexer else {
                throw MetalError.unsupported(
                    "GLM 5.2 layer \(layer.index) role mismatch: policy says "
                    + (isFull ? "full-indexer" : "IndexShare"))
            }
            let reused: [UInt32]?
            if isFull {
                reused = nil
            } else {
                guard let source = GLM52IndexSharePolicy
                          .selectionSourceLayer(for: layer.index),
                      let last = lastSelection, last.source == source else {
                    throw MetalError.unsupported(
                        "GLM 5.2 IndexShare layer \(layer.index) has no "
                        + "selection from its source layer")
                }
                reused = last.rows
            }
            let result = try glm52ResidentDecodeLayer(
                weights: layer.weights, ffn: layer.ffn,
                caches: layer.caches, input: hidden,
                reusedSelection: reused, position: position)
            hidden = result.output
            selections[layer.index] = result.selection
            if let routing = result.routing {
                routings[layer.index] = routing
            }
            if isFull {
                lastSelection = (layer.index, result.selection)
            }
        }

        return (try glm52ResidentLogits(outputHead: outputHead,
                                        hidden: hidden),
                selections, routings)
    }

    /// Resident output head on its own: final RMSNorm plus the vocabulary
    /// matvec over the resident Q8_0 rows. Shared by the stack forward and
    /// the streaming engine's manual layer loop.
    public func glm52ResidentLogits(outputHead: GLM52ResidentOutputHead,
                                    hidden: [Float]) throws -> [Float] {
        guard hidden.count == outputHead.embeddingWidth else {
            throw MetalError.unsupported(
                "GLM 5.2 resident head expects a "
                + "\(outputHead.embeddingWidth)-wide hidden state")
        }
        let hiddenBuffer = try glm52GraphBuffer(hidden)
        let normalized = try glm52GraphOutputBuffer(floats: hidden.count)
        let logits = try glm52GraphOutputBuffer(
            floats: outputHead.vocabularySize)
        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw MetalError.bufferAlloc
        }
        try glm52EncodeRMSNorm(into: commandBuffer, input: hiddenBuffer,
                               weight: outputHead.norm, output: normalized,
                               width: hidden.count)
        try glm52EncodeMatvecQ8(into: commandBuffer, input: normalized,
                                weights: outputHead.head, output: logits,
                                rowCount: outputHead.vocabularySize,
                                inputWidth: hidden.count)
        try glm52GraphCommit(commandBuffer)
        return glm52GraphReadback(logits, count: outputHead.vocabularySize)
    }

    /// Validation wrapper for the generic-width RMSNorm kernel.
    public func glm52RMSNorm(values: [Float], weight: [Float],
                             epsilon: Float = 1e-5) throws -> [Float] {
        guard !values.isEmpty, values.count == weight.count,
              values.count <= Int(UInt32.max), epsilon > 0 else {
            throw MetalError.unsupported(
                "GLM 5.2 RMSNorm expects matching non-empty values/weight")
        }
        let input = try glm52GraphBuffer(values)
        let weights = try glm52GraphBuffer(weight)
        let output = try glm52GraphOutputBuffer(floats: values.count)
        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw MetalError.bufferAlloc
        }
        try glm52EncodeRMSNorm(into: commandBuffer, input: input,
                               weight: weights, output: output,
                               width: values.count, epsilon: epsilon)
        try glm52GraphCommit(commandBuffer)
        return glm52GraphReadback(output, count: values.count)
    }
}
