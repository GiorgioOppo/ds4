import Foundation

/// Classe astratta base per un agent runtime. L'identità è
/// stabile per la durata dell'istanza; la modalità è una scelta
/// di policy (build vs plan) ortogonale alla logica concreta.
///
/// La sottoclasse reale (Phase B: `DSV4Agent`) wrappa
/// `InferenceService` di DeepSeekUI per produrre token via il
/// Transformer; le demo (vedi `DemoEchoAgent`) restituiscono
/// envelope sintetici per validare il contratto.
///
/// **Invariante per `@unchecked Sendable`**: solo stored
/// properties `let` di tipo `Sendable`. Eventuale stato mutabile
/// (es. cancellation flag) va dietro un actor o
/// `ManagedCriticalState`, non come `var` qui.
open class AgentBase: @unchecked Sendable {
    public let id: UUID
    public let mode: AgentMode

    public init(id: UUID = UUID(), mode: AgentMode = .build) {
        self.id = id
        self.mode = mode
    }

    /// ASTRATTO — la subclass implementa il loop di inferenza e
    /// restituisce uno stream di envelope. Il chiamante itera
    /// sullo stream finché non incontra l'envelope `agent.done`
    /// oppure lo stream finisce.
    open func step(input: AgentInput) -> AgentEventStream {
        fatalError(
            "Subclass \(type(of: self)) must override step(input:)")
    }

    /// Default sovrascrivibile: reset di stato locale (KV cache,
    /// contatori). Le subclass stateless lo lasciano vuoto.
    open func reset() async { /* noop */ }

    /// Default sovrascrivibile: richiesta di interruzione
    /// cooperativa. La subclass è responsabile di leggere il
    /// flag dentro il proprio loop.
    open func cancel() async { /* noop */ }
}
