import Foundation

/// Cancellation-aware FIFO mutex for the single mutable inference engine.
///
/// This deliberately uses a small lock instead of an actor so `release()` can
/// run synchronously from a request's `defer`.  Server shutdown can therefore
/// await the request task knowing that it has already relinquished the gate;
/// there is no untracked cleanup task left behind.
final class RequestGate: @unchecked Sendable {
    private struct Waiter {
        let id: UUID
        let continuation: CheckedContinuation<Void, Error>
    }

    private let lock = NSLock()
    private var busy = false
    private var waiters: [Waiter] = []

    func acquire() async throws {
        try Task.checkCancellation()
        let id = UUID()

        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                var acquireImmediately = false
                var rejectAsCancelled = false

                lock.lock()
                // The cancellation handler can race registration. Checking the
                // current task while holding the same lock makes both orders
                // safe: either it removes an existing waiter, or we never add it.
                let cancelled = withUnsafeCurrentTask { $0?.isCancelled ?? true }
                if cancelled {
                    rejectAsCancelled = true
                } else if !busy {
                    busy = true
                    acquireImmediately = true
                } else {
                    waiters.append(Waiter(id: id, continuation: continuation))
                }
                lock.unlock()

                if acquireImmediately {
                    continuation.resume()
                } else if rejectAsCancelled {
                    continuation.resume(throwing: CancellationError())
                }
            }
        } onCancel: {
            self.cancelWaiter(id)
        }

        // Cancellation may win after a waiter was dequeued (and therefore can
        // no longer be found by `cancelWaiter`) but before its continuation
        // resumes. In that case ownership was transferred to us: relinquish it
        // synchronously before surfacing cancellation to the request.
        if Task.isCancelled {
            release()
            throw CancellationError()
        }
    }

    func release() {
        let next: CheckedContinuation<Void, Error>?
        lock.lock()
        if waiters.isEmpty {
            busy = false
            next = nil
        } else {
            next = waiters.removeFirst().continuation
        }
        lock.unlock()
        next?.resume()
    }

    private func cancelWaiter(_ id: UUID) {
        let continuation: CheckedContinuation<Void, Error>?
        lock.lock()
        if let index = waiters.firstIndex(where: { $0.id == id }) {
            continuation = waiters.remove(at: index).continuation
        } else {
            continuation = nil
        }
        lock.unlock()
        continuation?.resume(throwing: CancellationError())
    }
}
