import Foundation
import Metal
import MetalPerformanceShaders
import DS4Core

/// Native Vision-Exp encoder. The BF16 sidecar stays memory-mapped; matrix
/// products use MPS with Float32 accumulation and explicit BF16 boundaries.
/// Use from the inference service's serial worker, like StreamingDecoder.
public final class DeepSeekV4VisionEncoder {
    public static let sourceRevision = "e46e16bf6035c6f317eb2ac7458eb0362926d402"
    public let visualRouterBias: [[Float]]
    public let hashRouterBias: [[Float]]
    private let model: GGUFModel
    private let runtime: MetalRuntime
    private let library: MTLLibrary
    private var pipelines: [String: MTLComputePipelineState] = [:]
    private var tensors: [String: GPUTensor] = [:]
    private let sentinels: [DeepSeekV4VisionToken: [Float]]

    public static func isVisionLanguageModel(_ model: GGUFModel) -> Bool {
        model.string("general.architecture") == "deepseek4"
            && model.string("deepseek4.checkpoint_variant") == "vision-exp"
            && model.bool("deepseek4.vision.sidecar_required") == true
            && model.string("general.source.revision") == sourceRevision
            && model.u32("deepseek4.embedding_length") == 4096
            && model.u32("deepseek4.block_count") == 43
    }

    /// Validate the entire tensor directory without creating a Metal device or
    /// reading the 933 MB payload. Used when choosing the encoder in Settings.
    public static func validateModel(path: String) throws {
        try validate(GGUFModel(path: path, metalMapping: false))
    }

    public init(modelPath: String, runtime: MetalRuntime) throws {
        let model = try GGUFModel(path: modelPath)
        try Self.validate(model)
        self.model = model
        self.runtime = runtime
        let options = MTLCompileOptions()
        if #available(macOS 15.0, *) { options.mathMode = .safe }
        else { options.fastMathEnabled = false }
        self.library = try runtime.device.makeLibrary(source: deepSeekV4VisionMetalSource, options: options)
        self.visualRouterBias = try (0..<43).map { try Self.vector(model, "layers.\($0).ffn.gate.bias_vl") }
        self.hashRouterBias = try (0..<3).map { try Self.vector(model, "layers.\($0).ffn.gate.bias") }
        self.sentinels = try [
            .start: Self.vector(model, "image_start"), .pad: Self.vector(model, "image_pad"),
            .newline: Self.vector(model, "image_newline"), .end: Self.vector(model, "image_end"),
        ]
    }

    public func encode(imageData: Data) throws -> DeepSeekV4VisionEmbedding {
        let patches = try DeepSeekV4ImagePreprocessor.preprocess(data: imageData)
        return try encode(patches: patches)
    }

    public func promptBlock(embedding: DeepSeekV4VisionEmbedding, startPosition: Int,
                            vocabularySize: Int) throws -> DeepSeekV4VisionPromptBlock {
        guard vocabularySize > 0, vocabularySize <= Int.max - 4,
              (1...384).contains(embedding.gridWidth), (1...384).contains(embedding.gridHeight),
              embedding.values.count == embedding.gridHeight * embedding.gridWidth * 4096 else {
            throw DeepSeekV4VisionError.invalidLayout("embedding o vocabolario incoerente")
        }
        let layout = try DeepSeekV4VisionLayout(gridHeight: embedding.gridHeight,
            gridWidth: embedding.gridWidth, startPosition: startPosition)
        var rows: [[Float]] = [], imageIndex = 0
        rows.reserveCapacity(layout.types.count)
        for type in layout.types {
            if type == .image {
                let source = layout.imagePermutation[imageIndex] * 4096
                rows.append(Array(embedding.values[source..<(source + 4096)]))
                imageIndex += 1
            } else {
                guard let row = sentinels[type] else {
                    throw DeepSeekV4VisionError.invalidLayout("sentinella mancante")
                }
                rows.append(row)
            }
        }
        return .init(startPosition: startPosition,
            tokens: layout.types.map { vocabularySize + $0.rawValue }, embeddings: rows)
    }

    private static func validate(_ model: GGUFModel) throws {
        guard model.string("general.architecture") == "deepseek4-vision",
              model.string("deepseek4-vision.checkpoint_variant") == "vision-exp",
              model.string("general.source.revision") == sourceRevision else {
            throw DeepSeekV4VisionError.invalidModel("serve il sidecar DeepSeek V4 Flash Vision-Exp corrispondente")
        }
        let expected: [String: UInt32] = [
            "block_count": 32, "embedding_length": 1024, "feed_forward_length": 2816,
            "attention.head_count": 16, "projection_length": 4096, "patch_size": 14,
            "downsample_ratio": 3, "image.max_tokens": 384, "image.min_pixels": 147456,
            "image.max_width_height_ratio": 8, "language.block_count": 43, "language.expert_count": 256,
        ]
        for (key, value) in expected where model.u32("deepseek4-vision.\(key)") != value {
            throw DeepSeekV4VisionError.invalidModel("metadata \(key) non supportata")
        }
        guard model.f32Compat("deepseek4-vision.attention.layer_norm_rms_epsilon") == Float(1e-6),
              model.f32Compat("deepseek4-vision.rope.freq_base") == 10000 else {
            throw DeepSeekV4VisionError.invalidModel("parametri RMS/RoPE non supportati")
        }
        var specs: [String: (UInt32, [UInt64])] = [
            "vision.patch_embed.proj.weight": (30, [588, 1024]),
            "vision.patch_embed.proj.bias": (30, [1024]), "vision.norm.weight": (30, [1024]),
            "aligner.w1.weight": (30, [9216, 4096]), "aligner.w1.bias": (30, [4096]),
            "aligner.w2.weight": (30, [4096, 4096]), "aligner.w2.bias": (30, [4096]),
        ]
        for name in ["image_start", "image_pad", "image_newline", "image_end"] { specs[name] = (30, [4096]) }
        for layer in 0..<32 {
            let base = "vision.blocks.\(layer)."
            for suffix in ["norm1.weight", "norm2.weight", "attn.wo.bias"] { specs[base + suffix] = (30, [1024]) }
            specs[base + "attn.wqkv.weight"] = (30, [1024, 3072])
            specs[base + "attn.wqkv.bias"] = (30, [3072])
            specs[base + "attn.wo.weight"] = (30, [1024, 1024])
            specs[base + "mlp.w1.weight"] = (30, [1024, 5632])
            specs[base + "mlp.w2.weight"] = (30, [2816, 1024])
        }
        for layer in 0..<43 { specs["layers.\(layer).ffn.gate.bias_vl"] = (0, [256]) }
        for layer in 0..<3 {
            specs["mtp.\(layer).ffn.gate.bias_vl"] = (0, [256])
            specs["layers.\(layer).ffn.gate.bias"] = (0, [256])
        }
        guard model.tensors.count == specs.count else {
            throw DeepSeekV4VisionError.invalidModel("attesi \(specs.count) tensori, trovati \(model.tensors.count)")
        }
        for (name, (type, dims)) in specs {
            guard let tensor = model.findTensor(name), tensor.type == type, tensor.dims == dims else {
                throw DeepSeekV4VisionError.invalidModel("tensore \(name) mancante o con forma/tipo errati")
            }
        }
    }

    private static func vector(_ model: GGUFModel, _ name: String) throws -> [Float] {
        guard let tensor = model.findTensor(name) else {
            throw DeepSeekV4VisionError.invalidModel("tensore \(name) mancante")
        }
        let source = model.mapBase.advanced(by: Int(tensor.absOffset))
        if tensor.type == 0 {
            return Array(UnsafeBufferPointer(start: source.assumingMemoryBound(to: Float.self), count: Int(tensor.elements)))
        }
        let words = source.assumingMemoryBound(to: UInt16.self)
        return (0..<Int(tensor.elements)).map { Float(bitPattern: UInt32(words[$0]) << 16) }
    }

    private func tensor(_ name: String) throws -> GPUTensor {
        if let tensor = tensors[name] { return tensor }
        guard let info = model.findTensor(name) else {
            throw DeepSeekV4VisionError.invalidModel("tensore \(name) mancante")
        }
        let tensor = try GPUTensor.mappedNoCopy(runtime,
            ptr: model.mapBase.advanced(by: Int(info.absOffset)),
            byteLength: Int(info.bytes), elementCount: Int(info.elements))
        tensors[name] = tensor
        return tensor
    }

    private func pipeline(_ name: String) throws -> MTLComputePipelineState {
        if let pipeline = pipelines[name] { return pipeline }
        guard let function = library.makeFunction(name: name) else { throw MetalError.missingKernel(name) }
        let pipeline = try runtime.device.makeComputePipelineState(function: function)
        pipelines[name] = pipeline
        return pipeline
    }

    private func dispatch(_ cb: MTLCommandBuffer, _ name: String, args: [UInt32],
                          buffers: [GPUTensor], width: Int, height: Int = 1,
                          groups: Bool = false, threads: Int = 256, sharedBytes: Int = 0) throws {
        guard let enc = cb.makeComputeCommandEncoder() else { throw MetalError.bufferAlloc }
        defer { enc.endEncoding() }
        enc.setComputePipelineState(try pipeline(name))
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: 0) }
        for (index, buffer) in buffers.enumerated() {
            enc.setBuffer(buffer.buffer, offset: buffer.byteOffset, index: index + 1)
        }
        if sharedBytes > 0 { enc.setThreadgroupMemoryLength(sharedBytes, index: 0) }
        let size = MTLSize(width: width, height: height, depth: 1)
        let threadSize = MTLSize(width: threads, height: 1, depth: 1)
        if groups { enc.dispatchThreadgroups(size, threadsPerThreadgroup: threadSize) }
        else { enc.dispatchThreads(size, threadsPerThreadgroup: threadSize) }
    }

    private func matmul(_ cb: MTLCommandBuffer, weight: String, x: GPUTensor, out: GPUTensor,
                        inDim: Int, outDim: Int, rows: Int, scratch: GPUTensor) throws {
        try dispatch(cb, "kernel_vision_convert_bf16", args: [UInt32(inDim * outDim)],
            buffers: [try tensor(weight), scratch], width: inDim * outDim)
        let left = MPSMatrix(buffer: x.buffer, offset: x.byteOffset,
            descriptor: MPSMatrixDescriptor(rows: rows, columns: inDim, rowBytes: inDim * 4, dataType: .float32))
        let right = MPSMatrix(buffer: scratch.buffer, offset: scratch.byteOffset,
            descriptor: MPSMatrixDescriptor(rows: outDim, columns: inDim, rowBytes: inDim * 4, dataType: .float32))
        let result = MPSMatrix(buffer: out.buffer, offset: out.byteOffset,
            descriptor: MPSMatrixDescriptor(rows: rows, columns: outDim, rowBytes: outDim * 4, dataType: .float32))
        let multiply = MPSMatrixMultiplication(device: runtime.device, transposeLeft: false,
            transposeRight: true, resultRows: rows, resultColumns: outDim, interiorColumns: inDim,
            alpha: 1, beta: 0)
        multiply.encode(commandBuffer: cb, leftMatrix: left, rightMatrix: right, resultMatrix: result)
    }

    private func encode(patches: DeepSeekV4ImagePatches) throws -> DeepSeekV4VisionEmbedding {
        let rows = patches.gridHeight * patches.gridWidth
        let alignedHeight = (patches.gridHeight + 2) / 3, alignedWidth = (patches.gridWidth + 2) / 3
        let alignedRows = alignedHeight * alignedWidth
        func allocate(_ n: Int) throws -> GPUTensor { try .uninitializedBytes(runtime, byteLength: n * 4, elementCount: n) }
        let patch = try GPUTensor.floats(runtime, patches.values)
        let a = try allocate(rows * 1024), b = try allocate(rows * 1024)
        let qkv = try allocate(rows * 3072), q = try allocate(rows * 1024)
        let k = try allocate(rows * 1024), v = try allocate(rows * 1024), attn = try allocate(rows * 1024)
        let mlp = try allocate(rows * 5632), mid = try allocate(rows * 2816)
        let alignInput = try allocate(alignedRows * 9216)
        let alignA = try allocate(alignedRows * 4096), alignB = try allocate(alignedRows * 4096)
        // Reused conversion scratch bounds resident memory independently of
        // the number of encoder layers. MPS never accumulates in Float16.
        let matrixScratch = try allocate(9216 * 4096)

        func command(_ body: (MTLCommandBuffer) throws -> Void) throws {
            try Task.checkCancellation()
            guard let cb = runtime.queue.makeCommandBuffer() else { throw MetalError.bufferAlloc }
            try body(cb)
            cb.commit(); cb.waitUntilCompleted()
            if let error = cb.error { throw error }
            try Task.checkCancellation()
        }
        func round(_ cb: MTLCommandBuffer, _ x: GPUTensor, _ width: Int, _ n: Int) throws {
            try dispatch(cb, "kernel_deepseek4_vision_round_bf16", args: [UInt32(width), UInt32(n)],
                buffers: [x], width: width, height: n)
        }
        func norm(_ cb: MTLCommandBuffer, x: GPUTensor, out: GPUTensor, weight: String) throws {
            try dispatch(cb, "kernel_glm53_vision_rms_bf16", args: [1024, UInt32(rows), Float(1e-6).bitPattern],
                buffers: [x, try tensor(weight), out], width: rows, groups: true, sharedBytes: 32)
            try round(cb, out, 1024, rows)
        }
        func multiply(_ cb: MTLCommandBuffer, _ weight: String, _ x: GPUTensor, _ out: GPUTensor,
                      _ inDim: Int, _ outDim: Int, _ n: Int) throws {
            try matmul(cb, weight: weight, x: x, out: out, inDim: inDim, outDim: outDim, rows: n, scratch: matrixScratch)
        }
        func bias(_ cb: MTLCommandBuffer, _ x: GPUTensor, _ name: String, _ width: Int, _ n: Int,
                  residual: GPUTensor? = nil) throws {
            var buffers = [x, try tensor(name)]
            if let residual { buffers.append(residual) }
            try dispatch(cb, residual == nil ? "kernel_glm53_vision_add_bias" : "kernel_glm53_vision_bias_residual",
                args: [UInt32(width), UInt32(n), 0], buffers: buffers, width: width, height: n)
            try round(cb, x, width, n)
        }
        try command { cb in
            try multiply(cb, "vision.patch_embed.proj.weight", patch, a, 588, 1024, rows)
            try bias(cb, a, "vision.patch_embed.proj.bias", 1024, rows)
        }
        // Each layer ends with cur=a, tmp=b, preserving the upstream two-buffer
        // residual ordering and BF16 round-to-nearest-even boundaries.
        for layer in 0..<32 {
            try autoreleasepool {
                try command { cb in
                    let base = "vision.blocks.\(layer)."
                    try norm(cb, x: a, out: b, weight: base + "norm1.weight")
                    try multiply(cb, base + "attn.wqkv.weight", b, qkv, 1024, 3072, rows)
                    try dispatch(cb, "kernel_deepseek4_vision_qkv_rope", args: [UInt32(rows), UInt32(patches.gridWidth)],
                        buffers: [qkv, try tensor(base + "attn.wqkv.bias"), q, k, v], width: rows, height: 16, groups: true, threads: 32)
                    try dispatch(cb, "kernel_glm53_vision_attention", args: [UInt32(rows), Float(0.125).bitPattern],
                        buffers: [q, k, v, attn], width: rows, height: 16, groups: true, threads: 32)
                    try round(cb, attn, 1024, rows)
                    try multiply(cb, base + "attn.wo.weight", attn, b, 1024, 1024, rows)
                    try bias(cb, b, base + "attn.wo.bias", 1024, rows, residual: a)
                    try norm(cb, x: b, out: a, weight: base + "norm2.weight")
                    try multiply(cb, base + "mlp.w1.weight", a, mlp, 1024, 5632, rows)
                    try round(cb, mlp, 5632, rows)
                    try dispatch(cb, "kernel_deepseek4_vision_swiglu_split", args: [2816, UInt32(rows)],
                        buffers: [mlp, mid], width: 2816, height: rows)
                    try multiply(cb, base + "mlp.w2.weight", mid, a, 2816, 1024, rows)
                    try round(cb, a, 1024, rows)
                    try dispatch(cb, "kernel_deepseek4_vision_add_residual", args: [1024, UInt32(rows)],
                        buffers: [a, b], width: 1024, height: rows)
                }
            }
        }
        try command { cb in
            try norm(cb, x: a, out: b, weight: "vision.norm.weight")
            try dispatch(cb, "kernel_deepseek4_vision_aligner_reorder",
                args: [UInt32(patches.gridHeight), UInt32(patches.gridWidth), UInt32(alignedRows)],
                buffers: [b, alignInput], width: 9216, height: alignedRows)
            try multiply(cb, "aligner.w1.weight", alignInput, alignA, 9216, 4096, alignedRows)
            try dispatch(cb, "kernel_deepseek4_vision_gelu_bias", args: [4096, UInt32(alignedRows)],
                buffers: [alignA, try tensor("aligner.w1.bias"), alignB], width: 4096, height: alignedRows)
            try multiply(cb, "aligner.w2.weight", alignB, alignA, 4096, 4096, alignedRows)
            try bias(cb, alignA, "aligner.w2.bias", 4096, alignedRows)
        }
        let values = alignA.floatArray(alignedRows * 4096)
        guard values.allSatisfy(\.isFinite) else {
            throw DeepSeekV4VisionError.invalidModel("l’encoder ha prodotto embedding non finiti")
        }
        return .init(gridHeight: alignedHeight, gridWidth: alignedWidth, values: values)
    }
}
