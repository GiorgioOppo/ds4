import Foundation

/// Contratto minimo per ogni messaggio scambiato fra componenti
/// (agent, plugin, chat, training, conversion, UI). Esiste per
/// uniformare identità, ordinamento, kind discriminator e
/// versioning: un consumer generico (logger, event hub, plugin
/// observer) può fare routing senza conoscere il tipo concreto.
public protocol MessageEnvelope: Sendable {
    /// Identificatore univoco del messaggio. Per debugging/tracing.
    var envelopeID: UUID { get }

    /// Quando è stato emesso il messaggio.
    var timestamp: Date { get }

    /// Discriminatore stabile in forma `"<dominio>.<tipo>"`.
    /// Esempi: `"chat.question"`, `"chat.answer"`, `"agent.token"`,
    /// `"agent.done"`, `"model.delta"`, `"plugin.lifecycle"`.
    var kind: String { get }

    /// Versione dello schema concreto. I consumer usano questo
    /// campo per retro-compatibilità quando il payload evolve.
    var schemaVersion: Int { get }
}

extension MessageEnvelope {
    public var schemaVersion: Int { 1 }
}

/// Helper non astratto per generare envelope ID/timestamp coerenti
/// quando si costruisce un nuovo messaggio. Le subclass/struct
/// concrete possono usarlo come stored property o ricomporne i
/// campi inline.
public struct EnvelopeMetadata: Sendable {
    public let envelopeID: UUID
    public let timestamp: Date

    public init(envelopeID: UUID = UUID(),
                timestamp: Date = Date()) {
        self.envelopeID = envelopeID
        self.timestamp = timestamp
    }
}
