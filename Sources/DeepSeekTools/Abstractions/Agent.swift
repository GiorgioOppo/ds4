import Foundation

/// Input di uno step di agent: la storia dei messaggi finora e
/// la modalità di reasoning richiesta. Tenuto deliberatamente
/// minimale per Phase A — `DSV4Agent` di Phase B aggiungerà
/// campi specifici (tool schemas, sampler options, cache id).
public struct AgentInput: Sendable {
    public var messages: [AgentChatMessage]
    public var thinkingMode: AgentThinkingMode

    public init(messages: [AgentChatMessage],
                thinkingMode: AgentThinkingMode = .chat) {
        self.messages = messages
        self.thinkingMode = thinkingMode
    }
}

/// Stream type-erased di envelope che l'agent emette durante uno
/// step (token incrementali, tool call decisi, progresso, done).
/// La subclass è responsabile di creare e chiudere lo stream.
public typealias AgentEventStream =
    AsyncThrowingStream<any MessageEnvelope, Error>
