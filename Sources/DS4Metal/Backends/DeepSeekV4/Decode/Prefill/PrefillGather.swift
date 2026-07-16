import Foundation
import Metal
import DS4Core

/// Background expert-gather for the batched prefill pipeline: runs one union's
/// gather on a background queue so the SSD I/O of group g+1 overlaps the GPU
/// FFNs of group g. @unchecked Sendable: the gather closure only reads the
/// read-only GGUF mmap and creates FRESH Metal buffers (MTLDevice is
/// thread-safe); the result is handed back through the semaphore, which gives
/// the consumer a happens-before edge on everything the worker wrote.
final class PrefillGather: @unchecked Sendable {
    typealias Tensors = (GPUTensor, GPUTensor, GPUTensor)
    private let layer: Int
    private let gather: (Int, [Int32]) throws -> Tensors

    init(layer: Int, gather: @escaping (Int, [Int32]) throws -> Tensors) {
        self.layer = layer
        self.gather = gather
    }

    final class Pending: @unchecked Sendable {
        fileprivate var result: Result<Tensors, Error>?
        fileprivate let sem = DispatchSemaphore(value: 0)
        /// Block until the gather completes; returns the tensors or rethrows.
        /// Consume ONCE: call either wait() or join(), never both.
        func wait() throws -> Tensors {
            sem.wait()
            return try result!.get()
        }
        /// Block until the gather completes, discarding the outcome (the
        /// error/cancellation path — the pipeline must never leave a worker
        /// touching the mmap after the caller unwinds).
        func join() { sem.wait() }
    }

    func start(_ union: [Int32]) -> Pending {
        let p = Pending()
        DispatchQueue.global(qos: .userInitiated).async {
            p.result = Result { try self.gather(self.layer, union) }
            p.sem.signal()
        }
        return p
    }
}
