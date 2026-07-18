import DS4Core
import Foundation
import Metal

// SSD layer streaming — the GLM analog of the DeepSeek StreamingDecoder's
// load/compute/evict: the BIG per-layer Q8_0 tensors are pread DIRECTLY into
// two reusable staging MTLBuffer sets (no intermediate host arrays), and the
// next layer's fill runs on a background queue while the GPU computes the
// current one. Per-layer small state (norms, router, proj, decode caches)
// stays resident; only streamed SPARSE layers are supported — the three
// leading dense layers must remain resident (different FFN widths, and they
// are the natural resident prefix anyway).

/// The streamable big-tensor descriptors of one sparse layer.
public struct GLM52StreamedLayerTensors: Sendable {
    public let index: Int
    /// True when the descriptors point into a Q4_K layer sidecar file (the
    /// streamer then reads through that layer's sidecar reader).
    public let fromSidecar: Bool
    let qA: GLM52WeightDescriptor
    let qB: GLM52WeightDescriptor
    let kvA: GLM52WeightDescriptor
    let keyB: GLM52WeightDescriptor
    let valueB: GLM52WeightDescriptor
    let attnOutput: GLM52WeightDescriptor
    let indexerKey: GLM52WeightDescriptor?
    let indexerQueryB: GLM52WeightDescriptor?
    let sharedGate: GLM52WeightDescriptor
    let sharedUp: GLM52WeightDescriptor
    let sharedDown: GLM52WeightDescriptor

    init(index: Int, map: GLM52WeightMap, fullIndexer: Bool) throws {
        self.index = index
        fromSidecar = false
        qA = try map.layer(index, .attentionQueryA)
        qB = try map.layer(index, .attentionQueryB)
        kvA = try map.layer(index, .attentionKVA)
        keyB = try map.layer(index, .attentionKeyB)
        valueB = try map.layer(index, .attentionValueB)
        attnOutput = try map.layer(index, .attentionOutput)
        indexerKey = fullIndexer ? try map.layer(index, .indexerKey) : nil
        indexerQueryB = fullIndexer ? try map.layer(index, .indexerQueryB) : nil
        sharedGate = try map.layer(index, .sharedGate)
        sharedUp = try map.layer(index, .sharedUp)
        sharedDown = try map.layer(index, .sharedDown)
    }

    /// Synthetic view built by the layer sidecar (descriptors carry the
    /// requantized types and sidecar-file offsets).
    init(index: Int, fromSidecar: Bool,
         qA: GLM52WeightDescriptor, qB: GLM52WeightDescriptor,
         kvA: GLM52WeightDescriptor, keyB: GLM52WeightDescriptor,
         valueB: GLM52WeightDescriptor, attnOutput: GLM52WeightDescriptor,
         indexerKey: GLM52WeightDescriptor?,
         indexerQueryB: GLM52WeightDescriptor?,
         sharedGate: GLM52WeightDescriptor,
         sharedUp: GLM52WeightDescriptor,
         sharedDown: GLM52WeightDescriptor) {
        self.index = index
        self.fromSidecar = fromSidecar
        self.qA = qA
        self.qB = qB
        self.kvA = kvA
        self.keyB = keyB
        self.valueB = valueB
        self.attnOutput = attnOutput
        self.indexerKey = indexerKey
        self.indexerQueryB = indexerQueryB
        self.sharedGate = sharedGate
        self.sharedUp = sharedUp
        self.sharedDown = sharedDown
    }

    var all: [GLM52WeightDescriptor] {
        [qA, qB, kvA, keyB, valueB, attnOutput,
         sharedGate, sharedUp, sharedDown]
            + [indexerKey, indexerQueryB].compactMap { $0 }
    }
}

/// One staging set of big-tensor buffers, reused across layers.
struct GLM52StreamedBigTensors {
    let qA: MTLBuffer
    let qB: MTLBuffer
    let kvA: MTLBuffer
    let keyB: MTLBuffer
    let valueB: MTLBuffer
    let attnOutput: MTLBuffer
    let indexerKey: MTLBuffer
    let indexerQueryB: MTLBuffer
    let sharedGate: MTLBuffer
    let sharedUp: MTLBuffer
    let sharedDown: MTLBuffer
}

public final class GLM52LayerStreamer {
    private final class Slot {
        let buffers: GLM52StreamedBigTensors
        let ready = DispatchSemaphore(value: 0)
        var layer = -1
        var fillError: Error?
        init(buffers: GLM52StreamedBigTensors) { self.buffers = buffers }
    }

    private let reader: GLM52PayloadReader
    /// Per-layer sidecar readers (Q4_K layer files); layers absent here
    /// stream from the main GGUF.
    private let sidecarReaders: [Int: GLM52PayloadReader]
    private let slots: [Slot]
    private var nextSlot = 0
    private var pending: [Slot] = []
    /// CONCURRENT on purpose: with 3+ slots two layers' fills overlap,
    /// doubling the SSD queue depth of the dominant stream. Ordering is
    /// preserved by the pending FIFO plus per-slot semaphores.
    private let fillQueue = DispatchQueue(
        label: "glm52.layer.streamer", qos: .userInitiated,
        attributes: .concurrent)
    private let stateLock = NSLock()
    /// Optional Metal fast-resource-loading path (DS4_GLM_MTLIO=1): SSD →
    /// MTLBuffer without the CPU pread copy, mirroring the DeepSeek
    /// ExpertBundle backend. Any failure permanently falls back to pread.
    /// Guarded by `stateLock` (fills are concurrent); serves the MAIN GGUF
    /// only — sidecar layers read via pread.
    private var metalIO: (queue: MTLIOCommandQueue,
                          handle: MTLIOFileHandle)?

    private func metalIOSnapshot() -> (queue: MTLIOCommandQueue,
                                       handle: MTLIOFileHandle)? {
        stateLock.lock()
        defer { stateLock.unlock() }
        return metalIO
    }

    private func disableMetalIO() {
        stateLock.lock()
        metalIO = nil
        stateLock.unlock()
    }

    private static func makeMetalIO(runtime: MetalRuntime, path: String)
        -> (queue: MTLIOCommandQueue, handle: MTLIOFileHandle)? {
        // Default ON (DS4_GLM_MTLIO=0 opts out): the fallback to pread on
        // any anomaly is automatic and permanent, so the fast path is safe
        // to prefer.
        guard ProcessInfo.processInfo.environment["DS4_GLM_MTLIO"] != "0"
        else { return nil }
        do {
            let descriptor = MTLIOCommandQueueDescriptor()
            descriptor.type = .concurrent
            descriptor.priority = .high
            // Sized for the concurrent fill pipeline: with 3+ staging slots
            // two layers' IO command buffers can be in flight together.
            descriptor.maxCommandBufferCount = 4
            descriptor.maxCommandsInFlight = 32
            let queue = try runtime.device.makeIOCommandQueue(
                descriptor: descriptor)
            let handle = try runtime.device.makeIOFileHandle(
                url: URL(fileURLWithPath: path))
            // Warm-up: pay first-use cost at load and prove the handle reads.
            guard let scratch = runtime.device.makeBuffer(
                length: 4_096, options: .storageModeShared) else { return nil }
            let warm = queue.makeCommandBuffer()
            warm.load(scratch, offset: 0, size: 4_096,
                      sourceHandle: handle, sourceHandleOffset: 0)
            warm.commit()
            warm.waitUntilCompleted()
            guard warm.status == .complete else { return nil }
            return (queue, handle)
        } catch {
            return nil
        }
    }

    /// Prefetches the caller can keep in flight without racing the slot
    /// being consumed (one slot is always "in use").
    var prefetchDepth: Int { slots.count - 1 }

    /// Allocates `slotCount` staging sets from a template layer's descriptor
    /// sizes (uniform across sparse layers — the schema fixes every type;
    /// sidecar tensors are smaller and fit the same slots).
    init(runtime: MetalRuntime,
         reader: GLM52PayloadReader,
         template: GLM52StreamedLayerTensors,
         slotCount: Int = 3,
         sidecarReaders: [Int: GLM52PayloadReader] = [:]) throws {
        self.reader = reader
        self.sidecarReaders = sidecarReaders
        func buffer(_ descriptor: GLM52WeightDescriptor?) throws -> MTLBuffer {
            let length = Int(descriptor?.bytes ?? 0)
            guard let buffer = runtime.device.makeBuffer(
                length: max(length, 16), options: .storageModeShared) else {
                throw MetalError.bufferAlloc
            }
            return buffer
        }
        var built: [Slot] = []
        for _ in 0..<max(2, slotCount) {
            built.append(Slot(buffers: GLM52StreamedBigTensors(
                qA: try buffer(template.qA), qB: try buffer(template.qB),
                kvA: try buffer(template.kvA),
                keyB: try buffer(template.keyB),
                valueB: try buffer(template.valueB),
                attnOutput: try buffer(template.attnOutput),
                indexerKey: try buffer(template.indexerKey),
                indexerQueryB: try buffer(template.indexerQueryB),
                sharedGate: try buffer(template.sharedGate),
                sharedUp: try buffer(template.sharedUp),
                sharedDown: try buffer(template.sharedDown))))
        }
        slots = built
        metalIO = Self.makeMetalIO(runtime: runtime, path: reader.path)
    }

    /// Async fill of the next staging slot with one layer's big tensors.
    /// Callers may keep up to `prefetchDepth` prefetches outstanding (one
    /// slot is always the one being consumed); fills run CONCURRENTLY and
    /// the caller consumes them strictly in prefetch order.
    func prefetch(_ tensors: GLM52StreamedLayerTensors) {
        stateLock.lock()
        let slot = slots[nextSlot]
        nextSlot = (nextSlot + 1) % slots.count
        slot.layer = tensors.index
        slot.fillError = nil
        pending.append(slot)
        stateLock.unlock()
        // Sidecar layers read from their own Q4_K file via pread; the main
        // GGUF keeps the MetalIO fast path.
        let fillReader = tensors.fromSidecar
            ? sidecarReaders[tensors.index] : reader
        guard let fillReader else {
            slot.fillError = MetalError.unsupported(
                "GLM 5.2 streamer: sidecar reader mancante per blk"
                + "\(tensors.index)")
            slot.ready.signal()
            return
        }
        fillQueue.async { [weak self] in
            let reader = fillReader
            // Fast path: MetalIO SSD→MTLBuffer loads, one IO command buffer
            // per slot. Any anomaly disables it permanently and the pread
            // path below repeats the fill from scratch.
            if let self, !tensors.fromSidecar,
               let io = self.metalIOSnapshot() {
                let commandBuffer = io.queue.makeCommandBuffer()
                var sized = true
                func load(_ descriptor: GLM52WeightDescriptor?,
                          _ buffer: MTLBuffer) {
                    guard let descriptor else { return }
                    guard Int(descriptor.bytes) <= buffer.length else {
                        sized = false
                        return
                    }
                    commandBuffer.load(
                        buffer, offset: 0, size: Int(descriptor.bytes),
                        sourceHandle: io.handle,
                        sourceHandleOffset: Int(descriptor.absOffset))
                }
                load(tensors.qA, slot.buffers.qA)
                load(tensors.qB, slot.buffers.qB)
                load(tensors.kvA, slot.buffers.kvA)
                load(tensors.keyB, slot.buffers.keyB)
                load(tensors.valueB, slot.buffers.valueB)
                load(tensors.attnOutput, slot.buffers.attnOutput)
                load(tensors.indexerKey, slot.buffers.indexerKey)
                load(tensors.indexerQueryB, slot.buffers.indexerQueryB)
                load(tensors.sharedGate, slot.buffers.sharedGate)
                load(tensors.sharedUp, slot.buffers.sharedUp)
                load(tensors.sharedDown, slot.buffers.sharedDown)
                if sized {
                    commandBuffer.commit()
                    commandBuffer.waitUntilCompleted()
                    if commandBuffer.status == .complete {
                        slot.ready.signal()
                        return
                    }
                }
                self.disableMetalIO()
            }
            do {
                func fill(_ descriptor: GLM52WeightDescriptor?,
                          _ buffer: MTLBuffer) throws {
                    guard let descriptor else { return }
                    guard Int(descriptor.bytes) <= buffer.length else {
                        throw MetalError.unsupported(
                            "streamed tensor \(descriptor.name) exceeds its "
                            + "staging slot")
                    }
                    try reader.read(descriptor, into:
                        UnsafeMutableRawBufferPointer(
                            start: buffer.contents(),
                            count: Int(descriptor.bytes)))
                }
                try fill(tensors.qA, slot.buffers.qA)
                try fill(tensors.qB, slot.buffers.qB)
                try fill(tensors.kvA, slot.buffers.kvA)
                try fill(tensors.keyB, slot.buffers.keyB)
                try fill(tensors.valueB, slot.buffers.valueB)
                try fill(tensors.attnOutput, slot.buffers.attnOutput)
                try fill(tensors.indexerKey, slot.buffers.indexerKey)
                try fill(tensors.indexerQueryB, slot.buffers.indexerQueryB)
                try fill(tensors.sharedGate, slot.buffers.sharedGate)
                try fill(tensors.sharedUp, slot.buffers.sharedUp)
                try fill(tensors.sharedDown, slot.buffers.sharedDown)
            } catch {
                slot.fillError = error
            }
            slot.ready.signal()
        }
    }

    /// Drop every outstanding prefetch — the heal after a pass abortito a
    /// metà (un errore tra prefetch e wait lascerebbe slot stantii che il
    /// passo successivo popperebbe al posto dei suoi). Waits for in-flight
    /// fills to land so no background write outlives the reset.
    func reset() {
        stateLock.lock()
        let outstanding = pending
        pending.removeAll()
        stateLock.unlock()
        for slot in outstanding { slot.ready.wait() }
    }

    /// Block until the OLDEST prefetched slot is filled, verify it carries
    /// the expected layer, and hand its buffers to the caller. The slot is
    /// reusable again as soon as the caller's synchronous GPU work on it
    /// completes (the decode functions wait on their command buffers).
    func wait(for layer: Int) throws -> GLM52StreamedBigTensors {
        stateLock.lock()
        guard !pending.isEmpty else {
            stateLock.unlock()
            throw MetalError.unsupported(
                "GLM 5.2 streamer: no prefetch pending for layer \(layer)")
        }
        let slot = pending.removeFirst()
        stateLock.unlock()
        slot.ready.wait()
        if let error = slot.fillError { throw error }
        guard slot.layer == layer else {
            throw MetalError.unsupported(
                "GLM 5.2 streamer: slot carries layer \(slot.layer), "
                + "expected \(layer)")
        }
        return slot.buffers
    }
}
