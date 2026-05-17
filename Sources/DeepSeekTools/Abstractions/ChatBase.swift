import Foundation

/// Classe astratta base per la chat: orchestra il flow
/// `Question → answer(question:) → Answer`. Mantiene la `Chat`
/// interna come stato; la subclass concreta implementa solo la
/// produzione della `Answer` (può chiamare un `AgentBase`, un
/// endpoint remoto, un mock test).
///
/// Il template method `ask(_:)` è `final` e garantisce un punto
/// unico di:
/// 1. costruzione della `Question`,
/// 2. registrazione del turno "in volo",
/// 3. pubblicazione dell'envelope `Question` a observer
///    (plugin/UI) tramite `publish`,
/// 4. delega ad `answer(question:)` (astratto),
/// 5. chiusura del turno con la `Answer`,
/// 6. pubblicazione dell'envelope `Answer`.
///
/// **Invariante per `@unchecked Sendable`**: lo stato mutabile
/// è la sola `chat` property. `ChatBase` non è thread-safe; un
/// caller che chiama `ask` in parallelo deve sincronizzare
/// esternamente (es. dietro un actor).
open class ChatBase: @unchecked Sendable {
    /// Stato della conversazione. Read-only dall'esterno;
    /// modificato solo dal template method `ask`.
    public private(set) var chat: Chat

    public init(id: UUID = UUID(), title: String = "Untitled") {
        self.chat = Chat(id: id, title: title)
    }

    /// ASTRATTO — la subclass produce la `Answer` per la
    /// `Question`. Può essere sincrona/asincrona, locale/remota.
    open func answer(question: Question) async throws -> Answer {
        fatalError(
            "Subclass \(type(of: self)) must override answer(question:)")
    }

    /// Default sovrascrivibile: notifica plugin/UI di un envelope.
    /// La subclass tipica binda un `PluginRegistry` o un event hub.
    open func publish(envelope: any MessageEnvelope) async { /* noop */ }

    /// Template method NON sovrascrivibile.
    @discardableResult
    public final func ask(_ content: String,
                          role: ChatRole = .user) async throws -> Answer {
        let q = Question(content: content, role: role)
        chat.turns.append(ChatTurn(question: q, answer: nil))
        await publish(envelope: q)
        let a = try await answer(question: q)
        chat.turns[chat.turns.count - 1] = ChatTurn(question: q, answer: a)
        await publish(envelope: a)
        return a
    }

    /// Default sovrascrivibile: svuota i turni mantenendo id e
    /// titolo. La subclass può estendere per resettare l'agent
    /// sottostante.
    open func clear() {
        chat = Chat(id: chat.id,
                    title: chat.title,
                    createdAt: chat.createdAt,
                    turns: [])
    }
}
