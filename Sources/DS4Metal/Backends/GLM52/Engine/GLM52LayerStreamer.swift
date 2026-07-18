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
    private let slots: [Slot]
    private var nextSlot = 0
    private var pending: [Slot] = []
    private let fillQueue = DispatchQueue(
        label: "glm52.layer.streamer", qos: .userInitiated)
    private let stateLock = NSLock()
    /// Optional Metal fast-resource-loading path (DS4_GLM_MTLIO=1): SSD →
    /// MTLBuffer without the CPU pread copy, mirroring the DeepSeek
    /// ExpertBundle backend. Any failure permanently falls back to pread —
    /// only touched from the serial fill queue after init.
    private var metalIO: (queue: MTLIOCommandQueue,
                          handle: MTLIOFileHandle)?

    private static func makeMetalIO(runtime: MetalRuntime, path: String)
        -> (queue: MTLIOCommandQueue, handle: MTLIOFileHandle)? {
        guard ProcessInfo.processInfo.environment["DS4_GLM_MTLIO"] == "1"
        else { return nil }
        do {
            let descriptor = MTLIOCommandQueueDescriptor()
            descriptor.type = .concurrent
            descriptor.priority = .high
            descriptor.maxCommandBufferCount = 2
            descriptor.maxCommandsInFlight = 16
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

    /// Allocates the two staging sets from a template layer's descriptor
    /// sizes (uniform across sparse layers — the schema fixes every type).
    init(runtime: MetalRuntime,
         reader: GLM52PayloadReader,
         template: GLM52StreamedLayerTensors) throws {
        self.reader = reader
        func buffer(_ descriptor: GLM52WeightDescriptor?) throws -> MTLBuffer {
            let length = Int(descriptor?.bytes ?? 0)
            guard let buffer = runtime.device.makeBuffer(
                length: max(length, 16), options: .storageModeShared) else {
                throw MetalError.bufferAlloc
            }
            return buffer
        }
        var built: [Slot] = []
        for _ in 0..<2 {
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
    /// Safe to call while the OTHER slot's layer computes: fills alternate
    /// and the caller consumes them strictly in prefetch order.
    func prefetch(_ tensors: GLM52StreamedLayerTensors) {
        stateLock.lock()
        let slot = slots[nextSlot]
        nextSlot = (nextSlot + 1) % slots.count
        slot.layer = tensors.index
        slot.fillError = nil
        pending.append(slot)
        stateLock.unlock()
        fillQueue.async { [weak self, reader] in
            // Fast path: MetalIO SSD→MTLBuffer loads, one IO command buffer
            // per slot. Any anomaly disables it permanently and the pread
            // path below repeats the fill from scratch.
            if let self, let io = self.metalIO {
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
                self.metalIO = nil
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
