import Foundation

/// Sorgente di `Tool` registrabili. Generalizza il pattern di
/// `DefaultTools.standard(_:)` (lista statica hardcoded) per
/// permettere scoperta dinamica: bundle plugin, eseguibili
/// esterni, API remote.
///
/// Fonte: `docs/GAP-ANALYSIS-OPENCODE.md:284`,
/// `docs/MODULES.md:366`. La migrazione di `DefaultTools` a
/// usare `ToolProvider` è Phase B.
public protocol ToolProvider: Sendable {
    /// Nome univoco del provider (per logging/diagnostica).
    var providerName: String { get }

    /// Restituisce i tool offerti dal provider. Può essere
    /// chiamato più volte (rescan); deve essere idempotente
    /// entro la stessa istanza.
    func discover() async throws -> [any Tool]
}

/// Classe astratta base. Le subclass concrete implementano solo
/// `discover()`.
open class ToolProviderBase: ToolProvider, @unchecked Sendable {
    public let providerName: String

    public init(providerName: String) {
        self.providerName = providerName
    }

    /// ASTRATTO — la subclass implementa la scoperta.
    open func discover() async throws -> [any Tool] {
        fatalError(
            "Subclass \(type(of: self)) must override discover()")
    }
}
