import Foundation

/// DEMO: plugin minimale che riceve gli envelope spediti dal
/// `PluginRegistry` e li annota in un contatore + lista interna.
///
/// Espone deliberatamente `observedCount` / `observedKinds` come
/// accessor read-only sotto un `NSLock` per i test — questo è
/// l'eccezione documentata all'invariante "solo `let`" della
/// `@unchecked Sendable`: lo stato mutabile è dietro lock,
/// non property pubblica `var`.
public final class DemoLoggerPlugin: PluginBase, @unchecked Sendable {
    private let lock = NSLock()
    private var _observedCount: Int = 0
    private var _observedKinds: [String] = []
    private var _bootstrapCount: Int = 0

    public init() {
        super.init(name: "demo.logger", version: "0.1.0")
    }

    // `NSLock.lock/unlock` non sono disponibili da contesti async
    // in Swift 6. Stesso pattern di
    // `Sources/DeepSeekTraining/FineTuneProgress.swift:115` —
    // metodi sync che il caller async invoca.

    private func recordBootstrap() {
        lock.lock(); defer { lock.unlock() }
        _bootstrapCount += 1
    }

    private func recordObservation(kind: String) {
        lock.lock(); defer { lock.unlock() }
        _observedCount += 1
        _observedKinds.append(kind)
    }

    public override func bootstrap(host: PluginHost) async throws {
        recordBootstrap()
    }

    public override func observe(envelope: any MessageEnvelope) async {
        recordObservation(kind: envelope.kind)
    }

    public var observedCount: Int {
        lock.lock(); defer { lock.unlock() }
        return _observedCount
    }

    public var observedKinds: [String] {
        lock.lock(); defer { lock.unlock() }
        return _observedKinds
    }

    public var bootstrapCount: Int {
        lock.lock(); defer { lock.unlock() }
        return _bootstrapCount
    }
}

/// DEMO: host minimale per test — broadcast `emit` su un
/// `PluginRegistry` opzionale e ritorna `nil` per ogni secret.
public final class DemoPluginHost: PluginHost, @unchecked Sendable {
    public let registry: PluginRegistry?

    public init(registry: PluginRegistry? = nil) {
        self.registry = registry
    }

    public func emit(_ envelope: any MessageEnvelope) async {
        await registry?.broadcast(envelope)
    }

    public func secret(_ key: String) async -> String? { nil }
}
