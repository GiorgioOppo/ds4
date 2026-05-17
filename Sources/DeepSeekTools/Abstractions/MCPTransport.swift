import Foundation

/// Trasporto JSON-RPC astratto per MCP. Separa il framing
/// (chi mette/leva la lunghezza, chi accumula bytes fino al
/// messaggio completo) dal trasporto fisico (stdio su un
/// Process spawnato, HTTP/SSE, WebSocket, in-memory loopback).
///
/// Fonte: `docs/DEVELOPING.md:565` ("Recipe: add a new MCP
/// transport"). La conversione di `MCPClient` da stdio cablato
/// a `init(transport: MCPTransport)` è Phase B.
public protocol MCPTransport: Sendable {
    /// Nome del trasporto (per logging/diagnostica).
    var transportName: String { get }

    /// Apre il canale: spawn processo, dial socket, crea
    /// connessione HTTP keep-alive.
    func connect() async throws

    /// Spedisce un messaggio JSON-RPC framed.
    func send(_ jsonRPCMessage: Data) async throws

    /// Stream di messaggi JSON-RPC in arrivo. La subclass è
    /// responsabile del framing in lettura.
    func receive() -> AsyncThrowingStream<Data, Error>

    /// Chiude il canale e libera le risorse.
    func disconnect() async
}

/// Classe astratta base. Le subclass concrete implementano i
/// tre metodi I/O; `disconnect` ha un default no-op
/// sovrascrivibile.
open class MCPTransportBase: MCPTransport, @unchecked Sendable {
    public let transportName: String

    public init(transportName: String) {
        self.transportName = transportName
    }

    /// ASTRATTO.
    open func connect() async throws {
        fatalError(
            "Subclass \(type(of: self)) must override connect()")
    }

    /// ASTRATTO.
    open func send(_ jsonRPCMessage: Data) async throws {
        fatalError(
            "Subclass \(type(of: self)) must override send(_:)")
    }

    /// ASTRATTO.
    open func receive() -> AsyncThrowingStream<Data, Error> {
        fatalError(
            "Subclass \(type(of: self)) must override receive()")
    }

    /// Default sovrascrivibile.
    open func disconnect() async { /* noop */ }
}
