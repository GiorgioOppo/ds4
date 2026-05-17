import Foundation

/// Protocollo per un'estensione host. Definisce il contratto
/// minimo per la type-erasure: lo storage di `PluginRegistry`
/// usa `any Plugin` per tenere insieme istanze eterogenee. Le
/// classi astratte concrete derivano da `PluginBase` per
/// condividere lifecycle e default sovrascrivibili.
public protocol Plugin: Sendable {
    /// Nome univoco del plugin (usato come chiave nel registry).
    var name: String { get }

    /// Versione del plugin in forma libera (semver consigliato).
    var version: String { get }

    /// Inizializzazione asincrona. Chiamata una sola volta dal
    /// registry al momento della registrazione, dopo aver
    /// risolto eventuali secret/configurazioni via `host`.
    func bootstrap(host: PluginHost) async throws

    /// Cleanup asincrono. Chiamato dal registry su
    /// `shutdownAll()` o `unregister(_:)`.
    func shutdown() async

    /// Callback opzionale per envelope spediti dall'host. Il
    /// plugin sceglie quali kind interessano e ignora il resto.
    func observe(envelope: any MessageEnvelope) async
}
