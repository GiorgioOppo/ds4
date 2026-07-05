import Foundation
@preconcurrency import Network

public enum DistError: Error {
    case badFrame, closed, badPort
    case sliceGap(String), modelMismatch(String)
    case versionMismatch(String)     // mixed builds in the cluster
    case remote(String)              // an ERROR frame from a worker (its own words)
    case unreachable(String)         // TCP never became ready (fail-fast diagnosis)
}

/// Async framed connection over NWConnection: every message is a `DistFrameHeader`
/// (magic + type + length) followed by `length` payload bytes. Used by both the
/// worker (accepted connections) and the coordinator (outbound connections).
public final class DistConnection: @unchecked Sendable {
    private let conn: NWConnection

    public init(_ conn: NWConnection) { self.conn = conn }

    /// Open an outbound connection and WAIT until TCP is ready, failing fast
    /// with a diagnosable error instead of letting the first read hang for the
    /// OS timeout (~60–75 s of silence, then a cryptic POSIX 60). Intermediate
    /// `.waiting` states (connection refused, no route, firewall silence) are
    /// surfaced live through `onState`; cancelling the task cancels the attempt.
    public static func connect(host: String, port: UInt16, queue: DispatchQueue,
                               readyTimeout: TimeInterval = 15,
                               onState: @escaping @Sendable (String) -> Void = { _ in })
        async throws -> DistConnection {
        guard let p = NWEndpoint.Port(rawValue: port) else { throw DistError.badPort }
        let c = NWConnection(host: NWEndpoint.Host(host), port: p, using: .tcp)
        let dc = DistConnection(c)

        /// One-shot continuation guard + last `.waiting` reason. All mutations
        /// happen in SYNC closures (connection queue / timeout queue).
        final class Gate: @unchecked Sendable {
            private let lock = NSLock()
            private var done = false
            private var waiting = ""
            func lastWaiting() -> String { lock.lock(); defer { lock.unlock() }; return waiting }
            func noteWaiting(_ s: String) { lock.lock(); waiting = s; lock.unlock() }
            func once(_ body: () -> Void) {
                lock.lock(); defer { lock.unlock() }
                guard !done else { return }
                done = true
                body()
            }
        }
        let gate = Gate()

        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
                // Handler installed BEFORE start(): no state change can be missed.
                c.stateUpdateHandler = { state in
                    switch state {
                    case .ready:
                        c.stateUpdateHandler = nil
                        gate.once { cont.resume() }
                    case .failed(let error):
                        gate.once { cont.resume(throwing: DistError.unreachable(
                            "\(host):\(port) — \(error)")) }
                    case .cancelled:
                        gate.once { cont.resume(throwing: CancellationError()) }
                    case .waiting(let error):
                        // Not fatal yet (the OS keeps retrying) but it IS the
                        // diagnosis: refused = port/worker down; timeout/silence
                        // = firewall or Local-Network permission.
                        gate.noteWaiting("\(error)")
                        onState("connessione a \(host):\(port) in attesa: \(error)\n")
                    default:
                        break
                    }
                }
                c.start(queue: queue)
                queue.asyncAfter(deadline: .now() + readyTimeout) {
                    gate.once {
                        let waiting = gate.lastWaiting()
                        cont.resume(throwing: DistError.unreachable(
                            "\(host):\(port) non raggiungibile dopo \(Int(readyTimeout))s"
                            + (waiting.isEmpty ? "" : " (ultimo stato: \(waiting))")
                            + " — verifica: worker avviato e 'in ascolto'; firewall macOS sul Mac worker"
                            + " (consenti le connessioni in entrata a DwarfStar); permesso 'Rete locale'"
                            + " su entrambi i Mac (Impostazioni → Privacy e sicurezza → Rete locale);"
                            + " IP e porta (test: nc -vz \(host) \(port))"))
                        c.cancel()
                    }
                }
            }
        } onCancel: {
            c.cancel()   // → .failed on the connection queue → continuation resumes
        }
        return dc
    }

    public func start(queue: DispatchQueue) { conn.start(queue: queue) }
    public func cancel() { conn.cancel() }

    public func sendFrame(_ type: Dist.MsgType, _ payload: Data) async throws {
        var frame = DistFrameHeader(type: type, length: UInt32(payload.count)).encoded()
        frame.append(payload)
        try await sendRaw(frame)
    }

    public func readFrame() async throws -> (type: Dist.MsgType, payload: Data) {
        let head = try await readExact(DistFrameHeader.byteSize)
        guard let h = DistFrameHeader.decode(head) else { throw DistError.badFrame }
        let payload = h.length == 0 ? Data() : try await readExact(Int(h.length))
        return (h.type, payload)
    }

    // MARK: NWConnection async wrappers

    private func sendRaw(_ data: Data) async throws {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            conn.send(content: data, completion: .contentProcessed { error in
                if let error { cont.resume(throwing: error) } else { cont.resume() }
            })
        }
    }

    private func readExact(_ n: Int) async throws -> Data {
        var buf = Data()
        while buf.count < n {
            guard let chunk = try await receiveOnce(max: n - buf.count), !chunk.isEmpty else {
                throw DistError.closed
            }
            buf.append(chunk)
        }
        return buf
    }

    private func receiveOnce(max: Int) async throws -> Data? {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Data?, Error>) in
            conn.receive(minimumIncompleteLength: 1, maximumLength: max) { data, _, _, error in
                if let error { cont.resume(throwing: error) } else { cont.resume(returning: data) }
            }
        }
    }
}

/// One worker in the coordinator's route: its address and the layer slice it owns.
public struct DistRouteEntry: Sendable {
    public var host: String
    public var port: UInt16
    public var layerStart: Int
    public var layerEnd: Int     // inclusive
    public var hasOutput: Bool
    public init(host: String, port: UInt16, layerStart: Int, layerEnd: Int, hasOutput: Bool) {
        self.host = host; self.port = port; self.layerStart = layerStart
        self.layerEnd = layerEnd; self.hasOutput = hasOutput
    }

    func encode(into d: inout Data) {
        let h = Data(host.utf8)
        d.appendLE(UInt32(h.count)); d.append(h)
        d.appendLE(UInt32(port))
        d.appendLE(UInt32(layerStart)); d.appendLE(UInt32(layerEnd))
        d.appendLE(UInt32(hasOutput ? 1 : 0))
    }

    static func decode(_ d: Data, _ o: inout Data.Index) -> DistRouteEntry? {
        guard o + 4 <= d.endIndex else { return nil }
        let hLen = Int(d.readLE(&o) as UInt32)
        guard o + hLen + 16 <= d.endIndex else { return nil }
        let host = String(decoding: d[o..<o+hLen], as: UTF8.self); o += hLen
        let port = UInt16(clamping: d.readLE(&o) as UInt32)
        let ls = Int(d.readLE(&o) as UInt32), le = Int(d.readLE(&o) as UInt32)
        let hasOut = (d.readLE(&o) as UInt32) != 0
        return DistRouteEntry(host: host, port: port, layerStart: ls, layerEnd: le, hasOutput: hasOut)
    }
}

/// Coordinator-side listener for worker→worker forwarding: the TERMINAL worker
/// connects back here and delivers RESULT frames. Results are surfaced as an
/// AsyncStream the generation loop awaits (one in-flight chunk at a time).
public final class DistReturnListener: @unchecked Sendable {
    private let queue = DispatchQueue(label: "ds4.dist.return")
    private var listener: NWListener?
    public let results: AsyncStream<DistResult>
    private let cont: AsyncStream<DistResult>.Continuation

    public init() {
        (results, cont) = AsyncStream<DistResult>.makeStream()
    }

    public func start(port: UInt16) throws {
        let params = NWParameters.tcp
        params.allowLocalEndpointReuse = true
        guard let p = NWEndpoint.Port(rawValue: port) else { throw DistError.badPort }
        let l = try NWListener(using: params, on: p)
        l.newConnectionHandler = { [weak self] c in
            guard let self else { return }
            c.start(queue: self.queue)
            let conn = DistConnection(c)
            Task {
                while let (type, payload) = try? await conn.readFrame() {
                    if type == .result, let r = DistResult.decode(payload) { self.cont.yield(r) }
                }
                conn.cancel()
            }
        }
        l.start(queue: queue)
        listener = l
    }

    public func stop() {
        listener?.cancel(); listener = nil
        cont.finish()
    }
}
