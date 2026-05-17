import Foundation

/// DEMO: backend finto che, dato un `GenerationRequest`, emette
/// `GenerationDelta` carattere-per-carattere per la stringa
/// `"echo: <ultimo messaggio>"`, terminando con un delta
/// `isFinal == true`.
///
/// Comportamento osservabile equivalente a `DemoEchoAgent`, ma
/// a un livello d'astrazione inferiore: l'agent costruisce
/// **sopra** il backend (in Phase B), il backend non sa nulla
/// di turni/agent.
public final class DemoEchoBackend: ModelBackendBase,
                                     @unchecked Sendable {
    public init() {
        super.init(backendName: "demo.echo")
    }

    public override func generate(_ request: GenerationRequest)
        -> AsyncThrowingStream<GenerationDelta, Error> {
        AsyncThrowingStream { cont in
            let last = request.messages.last?.content ?? ""
            for ch in "echo: \(last)" {
                cont.yield(GenerationDelta(token: String(ch)))
            }
            cont.yield(GenerationDelta(token: "", isFinal: true))
            cont.finish()
        }
    }
}
