import Foundation

/// Serializes async closures (the shard runs one step at a time).
/// Executor su coda GCD seriale (SE-0392): i body fanno fan-out con
/// DispatchQueue.concurrentPerform (gather esperti, slice del decoder), che
/// da un thread del pool cooperativo può degradare al quasi-seriale — vedi
/// InferenceService.engineQueue. Semantica identica (coda seriale), cambia
/// solo il thread di esecuzione.
actor DistGate {
    private nonisolated let queue = DispatchSerialQueue(label: "ds4.dist.gate", qos: .userInitiated)
    nonisolated var unownedExecutor: UnownedSerialExecutor { queue.asUnownedSerialExecutor() }
    func run<T: Sendable>(_ body: @Sendable () throws -> T) async rethrows -> T { try body() }
}
