import Foundation

/// DEMO: trasporto MCP in-memory che fa eco dei messaggi
/// inviati (loopback). Utile come fixture di test per verificare
/// che il contratto `MCPTransport` regga e come modello di
/// riferimento per i trasporti futuri (HTTP/SSE, WebSocket).
///
/// Usa il pattern `AsyncStream.makeStream()` per esporre il
/// canale receiving: `send` accoda il messaggio sulla
/// `inboundContinuation`, `receive` lo restituisce.
public final class DemoStubMCPTransport: MCPTransportBase,
                                          @unchecked Sendable {
    private let inbound: AsyncStream<Data>
    private let inboundContinuation: AsyncStream<Data>.Continuation

    public init(transportName: String = "demo.stub") {
        let pair = AsyncStream<Data>.makeStream()
        self.inbound = pair.stream
        self.inboundContinuation = pair.continuation
        super.init(transportName: transportName)
    }

    public override func connect() async throws { /* noop */ }

    public override func send(_ msg: Data) async throws {
        inboundContinuation.yield(msg)
    }

    public override func receive() -> AsyncThrowingStream<Data, Error> {
        let stream = self.inbound
        return AsyncThrowingStream { cont in
            let task = Task {
                for await msg in stream {
                    cont.yield(msg)
                }
                cont.finish()
            }
            cont.onTermination = { _ in task.cancel() }
        }
    }

    public override func disconnect() async {
        inboundContinuation.finish()
    }
}
