import Foundation

/// Classe astratta base per i plugin host. Le subclass concrete
/// implementano `bootstrap(host:)` e — se interessate — possono
/// override `observe(envelope:)` per ricevere eventi e
/// `shutdown()` per cleanup.
///
/// **Invariante per `@unchecked Sendable`**: le subclass DEVONO
/// esporre soltanto stored properties `let` di tipo `Sendable`.
/// Nessuno stato mutabile. Una `var` aggiunta in futuro rompe
/// silenziosamente la thread-safety — questo è il prezzo della
/// scelta abstract-class vs protocol struct (vedi piano).
open class PluginBase: Plugin, @unchecked Sendable {
    public let name: String
    public let version: String

    public init(name: String, version: String = "0.1.0") {
        self.name = name
        self.version = version
    }

    /// ASTRATTO — subclass DEVE override. fatalError a runtime
    /// perché Swift non esprime astrattezza a compile-time.
    open func bootstrap(host: PluginHost) async throws {
        fatalError(
            "Subclass \(type(of: self)) must override bootstrap(host:)")
    }

    /// Default sovrascrivibile: nessuna azione di cleanup.
    open func shutdown() async { /* noop */ }

    /// Default sovrascrivibile: ignora ogni envelope ricevuto.
    open func observe(envelope: any MessageEnvelope) async { /* noop */ }
}
