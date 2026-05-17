import Foundation

/// Interfaccia che l'host del plugin (CLI, UI, server) espone ai
/// plugin per richiedere servizi: emissione di envelope verso
/// altri componenti, lookup di secret di configurazione.
///
/// Implementazioni reali (in `DeepSeekUI` o nel CLI) wrappano
/// `PluginRegistry` per il broadcast e una secret store (env vars,
/// Keychain, file) per la lookup.
public protocol PluginHost: Sendable {
    /// Il plugin emette un envelope verso l'host. L'host decide
    /// cosa farne (loggare, inoltrare ad altri plugin, ignorare).
    func emit(_ envelope: any MessageEnvelope) async

    /// Risoluzione di un secret per chiave stringa, es.
    /// `"slack.webhook_url"`, `"anthropic.api_key"`. `nil` se la
    /// chiave non è configurata.
    func secret(_ key: String) async -> String?
}
