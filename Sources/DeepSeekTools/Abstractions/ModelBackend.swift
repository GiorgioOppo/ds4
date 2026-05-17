import Foundation

/// Richiesta di generazione: contratto di input al backend.
/// Tenuto minimale per Phase A — in Phase B i dispatch site
/// reali aggiungeranno sampler options, tool schemas, cache key.
public struct GenerationRequest: Sendable {
    public let messages: [AgentChatMessage]
    public let thinkingMode: AgentThinkingMode
    public let maxTokens: Int
    /// Extras specifici per backend (es. `"tools_json"` per
    /// backend che accettano una lista di tool dichiarati).
    public let extras: [String: String]

    public init(messages: [AgentChatMessage],
                thinkingMode: AgentThinkingMode = .chat,
                maxTokens: Int = 2048,
                extras: [String: String] = [:]) {
        self.messages = messages
        self.thinkingMode = thinkingMode
        self.maxTokens = maxTokens
        self.extras = extras
    }
}

/// Evento incrementale emesso dal backend durante la generazione.
/// Conforme a `MessageEnvelope` per uniformità di routing.
public struct GenerationDelta: MessageEnvelope, Sendable {
    public let envelopeID: UUID
    public let timestamp: Date
    public let kind: String = "model.delta"
    public let schemaVersion: Int = 1

    /// Testo del token (può essere stringa vuota nel `isFinal`
    /// terminale, oppure un singolo carattere/grapheme).
    public let token: String

    /// True quando questo è l'ultimo delta della generazione.
    public let isFinal: Bool

    public init(token: String,
                isFinal: Bool = false,
                envelopeID: UUID = UUID(),
                timestamp: Date = Date()) {
        self.token = token
        self.isFinal = isFinal
        self.envelopeID = envelopeID
        self.timestamp = timestamp
    }
}

/// Backend di inferenza astratto: locale (Transformer su Metal),
/// OpenRouter, Anthropic, OpenAI, finto echo per i test.
/// Generalizza `ModelEndpoint` enum esistente (vedi
/// `docs/DEVELOPING.md:468`).
public protocol ModelBackend: Sendable {
    var backendName: String { get }
    func generate(_ request: GenerationRequest)
        -> AsyncThrowingStream<GenerationDelta, Error>
}

open class ModelBackendBase: ModelBackend, @unchecked Sendable {
    public let backendName: String

    public init(backendName: String) {
        self.backendName = backendName
    }

    /// ASTRATTO — la subclass produce lo stream di delta.
    open func generate(_ request: GenerationRequest)
        -> AsyncThrowingStream<GenerationDelta, Error> {
        fatalError(
            "Subclass \(type(of: self)) must override generate(_:)")
    }
}
