import Foundation
import Metal
import DS4Core

extension ExpertBundle {
    /// Enable Apple Metal fast resource loading for this already-validated
    /// bundle. The queue is concurrent so demand fills and next-layer
    /// look-ahead can remain in flight together. False means callers simply
    /// keep using the existing F_NOCACHE pread path.
    @discardableResult
    public func enableMetalIO(device: MTLDevice) -> Bool {
        do {
            let desc = MTLIOCommandQueueDescriptor()
            desc.type = .concurrent
            desc.priority = .high
            // A deep MetalIO queue can starve the demand gather on M1 Pro.
            // Two command buffers preserve demand + look-ahead concurrency;
            // 16 commands are enough to saturate its SSD without flooding it.
            desc.maxCommandBufferCount = 2
            desc.maxCommandsInFlight = 16
            let queue = try device.makeIOCommandQueue(descriptor: desc)
            let handle = try device.makeIOFileHandle(url: URL(fileURLWithPath: path))
            // Pay queue/file-handle first-use cost once at model load, not in
            // the first token's cache miss. Also proves this handle can read.
            guard let scratch = device.makeBuffer(length: 4096, options: .storageModeShared) else {
                Self.log("MetalIO non disponibile (scratch buffer) — uso pread")
                return false
            }
            let warm = queue.makeCommandBuffer()
            warm.load(scratch, offset: 0, size: 4096, sourceHandle: handle, sourceHandleOffset: 0)
            warm.commit(); warm.waitUntilCompleted()
            guard warm.status == .complete else {
                Self.log("MetalIO warm-up fallito (\(warm.error?.localizedDescription ?? "stato \(warm.status.rawValue)")) — uso pread")
                return false
            }
            metalIOLock.lock()
            metalIO = MetalIOBackend(queue: queue, handle: handle)
            metalIOFailureLogged = false
            metalIOSubmissions = 0
            let minGBs = ProcessInfo.processInfo.environment["DS4_MTLIO_MIN_GBS"].flatMap(Double.init) ?? 1.5
            metalIOBreaker = MetalIOCircuitBreaker(minimumGBs: minGBs)
            metalIOLock.unlock()
            Self.log("MetalIO attivo: SSD → MTLBuffer diretto (fallback pread disponibile)")
            return true
        } catch {
            Self.log("MetalIO non disponibile (\(error.localizedDescription)) — uso pread")
            return false
        }
    }

    func metalIOSnapshot() -> MetalIOBackend? {
        metalIOLock.lock(); defer { metalIOLock.unlock() }
        return metalIO
    }

    func disableMetalIO(_ message: String) {
        metalIOLock.lock()
        metalIO = nil
        let shouldLog = !metalIOFailureLogged
        metalIOFailureLogged = true
        metalIOLock.unlock()
        if shouldLog { Self.log("MetalIO fallito (\(message)) — fallback permanente a pread") }
    }

    func noteMetalIOSuccess(experts: Int, gbs: Double) {
        metalIOLock.lock()
        metalIOSubmissions += 1
        let first = metalIOSubmissions == 1
        metalIOLock.unlock()
        if first {
            Self.log(String(format: "MetalIO in uso: primo batch diretto di %d esperti → MTLBuffer (%.2f GB/s)",
                            experts, gbs))
        }
    }

    func evaluateMetalIO(bytes: Int, seconds: Double) -> MetalIOCircuitBreaker.Decision {
        metalIOLock.lock(); defer { metalIOLock.unlock() }
        return metalIOBreaker.record(bytes: bytes, seconds: seconds)
    }
}
