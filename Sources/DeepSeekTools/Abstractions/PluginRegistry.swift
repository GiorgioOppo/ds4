import Foundation

/// Registry centrale dei plugin. Mantiene la mappa
/// `name -> any Plugin`, orchestra il bootstrap, propaga gli
/// envelope a tutti i plugin registrati e gestisce lo shutdown.
///
/// `actor` per la stessa ragione di `ToolRegistry` in
/// `DeepSeekTools/ToolRegistry.swift:15`: lo stato mutabile può
/// essere acceduto da qualunque task, e gli envelope possono
/// essere broadcast da actor diversi.
public actor PluginRegistry {
    private var plugins: [String: any Plugin] = [:]

    public init() {}

    /// Registra un plugin chiamando il suo `bootstrap(host:)`.
    /// Se il bootstrap fallisce, il plugin NON viene memorizzato
    /// e l'errore propaga al chiamante.
    public func register(_ plugin: any Plugin,
                         host: PluginHost) async throws {
        try await plugin.bootstrap(host: host)
        plugins[plugin.name] = plugin
    }

    /// Rimuove un plugin dal registry chiamando il suo
    /// `shutdown()`. No-op se il nome non esiste.
    public func unregister(_ name: String) async {
        guard let p = plugins.removeValue(forKey: name) else { return }
        await p.shutdown()
    }

    /// Inoltra un envelope a tutti i plugin registrati. Il
    /// broadcast è sequenziale (ordine non garantito); ogni
    /// plugin esegue `observe` in isolamento.
    ///
    /// Snapshot della lista prima di iterare: gli `await`
    /// interni possono sospendere l'actor e altri caller
    /// possono modificare `plugins`. Senza snapshot l'iterazione
    /// finirebbe in UB.
    public func broadcast(_ envelope: any MessageEnvelope) async {
        let snapshot = Array(plugins.values)
        for p in snapshot {
            await p.observe(envelope: envelope)
        }
    }

    /// Chiude tutti i plugin in batch. Tipicamente chiamata
    /// dall'host su quit/teardown.
    public func shutdownAll() async {
        let snapshot = Array(plugins.values)
        plugins.removeAll()
        for p in snapshot {
            await p.shutdown()
        }
    }

    /// Nomi dei plugin attualmente registrati. Ordinato per
    /// stabilità nei test e in UI.
    public func names() -> [String] {
        Array(plugins.keys).sorted()
    }

    /// Lookup di un plugin per nome.
    public func plugin(named name: String) -> (any Plugin)? {
        plugins[name]
    }
}
