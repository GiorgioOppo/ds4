import Foundation
import Metal
import DS4Core

/// ABI shared with the standalone upstream shader (13 uints, two floats).
struct BonsaiArgs {
    var n: UInt32 = 0, rows: UInt32 = 0, cols: UInt32 = 0, type: UInt32 = 0
    var rowBytes: UInt32 = 0, pos: UInt32 = 0, heads: UInt32 = 0, kvheads: UInt32 = 0
    var dim: UInt32 = 0, rot: UInt32 = 0, width: UInt32 = 0, mode: UInt32 = 0, groups: UInt32 = 0
    var eps: Float = 0, base: Float = 0
    init(n: Int = 0, rows: Int = 0, cols: Int = 0, type: UInt32 = 0,
         rowBytes: Int = 0, pos: Int = 0, heads: Int = 0, kvheads: Int = 0,
         dim: Int = 0, rot: Int = 0, width: Int = 0, mode: UInt32 = 0,
         groups: Int = 0, eps: Float = 0, base: Float = 0) {
        self.n = UInt32(n); self.rows = UInt32(rows); self.cols = UInt32(cols); self.type = type
        self.rowBytes = UInt32(rowBytes); self.pos = UInt32(pos); self.heads = UInt32(heads); self.kvheads = UInt32(kvheads)
        self.dim = UInt32(dim); self.rot = UInt32(rot); self.width = UInt32(width); self.mode = mode
        self.groups = UInt32(groups); self.eps = eps; self.base = base
    }
}

struct BonsaiBuffer {
    let buffer: MTLBuffer
    var offset = 0
    func row(_ row: Int, _ width: Int) -> Self { Self(buffer: buffer, offset: offset + row * width * 4) }
}

struct BonsaiWeight {
    let view: BonsaiBuffer
    let spec: BonsaiConfiguration.Weight
    let signs: BonsaiBuffer?
    var columns: Int { spec.columns }
    var rows: Int { spec.rows }
    var type: UInt32 { spec.tensor.type }
}

/// Counts independent resident sessions conservatively, including their mapped
/// weights, before constructing any large Metal allocation.
final class BonsaiMemoryReservation: @unchecked Sendable {
    private static let ledger = Ledger()
    private final class Ledger: @unchecked Sendable {
        let lock = NSLock()
        var bytes: UInt64 = 0
    }
    let bytes: UInt64
    init(bytes: UInt64, device: MTLDevice) throws {
        let ledger = Self.ledger
        ledger.lock.lock(); defer { ledger.lock.unlock() }
        let used = max(ledger.bytes, UInt64(device.currentAllocatedSize))
        let budget = device.recommendedMaxWorkingSetSize
        guard used <= budget, bytes <= budget - used else {
            throw SwiftModelDecoderError.gpu("Bonsai richiede \(bytes / 1_048_576) MiB residenti; memoria Metal disponibile \((budget > used ? budget - used : 0) / 1_048_576) MiB.")
        }
        ledger.bytes += bytes
        self.bytes = bytes
    }
    deinit {
        Self.ledger.lock.lock(); Self.ledger.bytes -= bytes; Self.ledger.lock.unlock()
    }
}

final class BonsaiGPU {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let library: MTLLibrary
    let pipelines: [String: MTLComputePipelineState]

    init(device: MTLDevice? = MTLCreateSystemDefaultDevice()) throws {
        guard let device, device.hasUnifiedMemory, let queue = device.makeCommandQueue() else {
            throw SwiftModelDecoderError.gpu("Bonsai richiede Metal con memoria unificata.")
        }
        self.device = device; self.queue = queue
        guard MemoryLayout<BonsaiArgs>.stride == 60 else { throw SwiftModelDecoderError.gpu("Bonsai ABI non valida") }
        let options = MTLCompileOptions()
        if #available(macOS 15.0, *) { options.mathMode = .safe } else { options.fastMathEnabled = false }
        let library = try device.makeLibrary(source: BonsaiMetalSource.source, options: options)
        self.library = library
        let names = ["embed", "mv", "pq2_mv", "pq2_mv_full", "pq2_gate_up", "pq2_gate_up_full", "bf16_pair",
                     "mm", "mm_pq2_tiled", "hadamard", "norm", "element", "conv", "gdn", "gdn_128", "conv_batch",
                     "l2_batch", "gdn_batch", "gdn_batch_128", "mm_pq2_gate_up_tiled", "rope", "mrope", "cache",
                     "scores", "softmax", "attention", "scores_batch", "softmax_batch", "attention_batch",
                     "bf16_pair_batch", "ptq_mm", "ptq_gate_up", "ptq_gate_up_batch", "ptq_mm_8", "ptq_gate_up_batch_8"]
        var pipelines: [String: MTLComputePipelineState] = [:]
        for suffix in names {
            let name = "bonsai_" + suffix
            guard let function = library.makeFunction(name: name) else { throw SwiftModelDecoderError.gpu("Kernel \(name) mancante") }
            let pipeline = try device.makeComputePipelineState(function: function)
            let small = ["gdn_128", "gdn_batch_128", "mm_pq2_gate_up_tiled", "bf16_pair_batch"].contains(suffix) || suffix.hasPrefix("ptq_")
            guard pipeline.threadExecutionWidth == 32, pipeline.maxTotalThreadsPerThreadgroup >= (small ? 128 : 256),
                  pipeline.staticThreadgroupMemoryLength <= device.maxThreadgroupMemoryLength else {
                throw SwiftModelDecoderError.gpu("Dispositivo incompatibile con \(name)")
            }
            pipelines[name] = pipeline
        }
        self.pipelines = pipelines
    }

    func allocate(floats: Int, label: String) throws -> BonsaiBuffer {
        guard floats > 0, floats <= device.maxBufferLength / 4,
              let buffer = device.makeBuffer(length: floats * 4, options: .storageModeShared) else {
            throw SwiftModelDecoderError.gpu("Allocazione Bonsai fallita: \(label)")
        }
        buffer.label = label
        return BonsaiBuffer(buffer: buffer)
    }

    func mapWeight(_ spec: BonsaiConfiguration.Weight, model: GGUFModel, signs: BonsaiBuffer?) throws -> BonsaiWeight {
        let bytes = Int(spec.tensor.bytes)
        let pointer = model.mapBase.advanced(by: Int(spec.tensor.absOffset))
        let buffer: MTLBuffer?, offset: Int
        if bytes < 65536 {
            buffer = device.makeBuffer(bytes: pointer, length: bytes, options: .storageModeShared); offset = 0
        } else {
            let page = Int(getpagesize()), address = Int(bitPattern: pointer)
            let base = address & ~(page - 1)
            offset = address - base
            let length = (bytes + offset + page - 1) & ~(page - 1)
            guard length <= device.maxBufferLength, let aligned = UnsafeMutableRawPointer(bitPattern: base) else {
                throw SwiftModelDecoderError.gpu("Tensor Bonsai supera il limite Metal: \(spec.tensor.name)")
            }
            buffer = device.makeBuffer(bytesNoCopy: aligned, length: length, options: .storageModeShared, deallocator: nil)
        }
        guard let buffer else { throw SwiftModelDecoderError.gpu("Mappatura tensor Bonsai fallita: \(spec.tensor.name)") }
        buffer.label = spec.tensor.name
        return BonsaiWeight(view: BonsaiBuffer(buffer: buffer, offset: offset), spec: spec, signs: signs)
    }

    func dispatch(_ encoder: MTLComputeCommandEncoder, _ name: String, _ args: BonsaiArgs,
                  _ buffers: [BonsaiBuffer], _ x: Int, _ y: Int = 1, _ z: Int = 1, threads: Int = 256) throws {
        guard let pipeline = pipelines["bonsai_" + name], x > 0, y > 0, z > 0,
              threads <= pipeline.maxTotalThreadsPerThreadgroup else { throw SwiftModelDecoderError.gpu("Dispatch Bonsai non valido: \(name)") }
        encoder.setComputePipelineState(pipeline)
        var args = args
        encoder.setBytes(&args, length: MemoryLayout<BonsaiArgs>.stride, index: 0)
        for (index, view) in buffers.enumerated() {
            guard view.offset >= 0, view.offset < view.buffer.length,
                  view.buffer.device.registryID == device.registryID else { throw SwiftModelDecoderError.gpu("Buffer Bonsai non valido: \(name)") }
            encoder.setBuffer(view.buffer, offset: view.offset, index: index + 1)
        }
        encoder.dispatchThreadgroups(MTLSize(width: x, height: y, depth: z), threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    }
}
