import Foundation
@preconcurrency import Network

/// A distributed WORKER: owns a contiguous layer slice and runs it on demand.
/// On each accepted connection it announces its slice (HELLO), then serves WORK
/// frames (run `forwardSlice`, or the output head if the coordinator flagged this
/// as the last slice) and replies with RESULT frames. One coordinator at a time.
public final class DistWorker: @unchecked Sendable {
    public struct Config: Sendable {
        public var modelPath: String
        public var port: UInt16
        public var layerStart: Int
        public var layerEnd: Int      // inclusive
        public var hasOutput: Bool    // also owns the output head (last slice)
        public var contextSize: Int
        public init(modelPath: String, port: UInt16, layerStart: Int, layerEnd: Int,
                    hasOutput: Bool, contextSize: Int) {
            self.modelPath = modelPath; self.port = port; self.layerStart = layerStart
            self.layerEnd = layerEnd; self.hasOutput = hasOutput; self.contextSize = contextSize
        }
    }

    private let config: Config
    private let engine: DistEngine
    private let onLog: @Sendable (String) -> Void
    private let queue = DispatchQueue(label: "ds4.dist.worker")
    private let gate = DistGate()
    private var listener: NWListener?

    public init(config: Config, onLog: @escaping @Sendable (String) -> Void) throws {
        self.config = config
        self.onLog = onLog
        // KV/compressor allocated ONLY for this worker's slice (the rest of the
        // model's layers are never run here).
        self.engine = try DistEngine(modelPath: config.modelPath, contextSize: config.contextSize,
                                     kvLayers: config.layerStart..<(config.layerEnd + 1))
    }

    public func start() throws {
        let params = NWParameters.tcp
        params.allowLocalEndpointReuse = true
        guard let port = NWEndpoint.Port(rawValue: config.port) else { throw DistError.badPort }
        let l = try NWListener(using: params, on: port)
        l.stateUpdateHandler = { [onLog, config] state in
            switch state {
            case .ready: onLog("worker layer \(config.layerStart)…\(config.layerEnd) in ascolto su :\(config.port)\n")
            case .failed(let e): onLog("worker listener fallito: \(e)\n")
            default: break
            }
        }
        l.newConnectionHandler = { [weak self] c in self?.accept(c) }
        l.start(queue: queue)
        listener = l
    }

    public func stop() { listener?.cancel(); listener = nil }

    private func accept(_ c: NWConnection) {
        c.start(queue: queue)
        let conn = DistConnection(c)
        Task { [weak self] in await self?.serve(conn) }
    }

    private func serve(_ conn: DistConnection) async {
        onLog("connessione in ingresso\n")
        // Outbound connections (next-hop worker / coordinator return), per session.
        var downstream: [String: DistConnection] = [:]
        defer { for c in downstream.values { c.cancel() } }

        // `expectHello`: next-hop workers greet new connections with a HELLO frame
        // (consume it once); the coordinator's return listener does not.
        func outbound(_ host: String, _ port: UInt16, expectHello: Bool) async throws -> DistConnection {
            let key = "\(host):\(port)"
            if let c = downstream[key] { return c }
            let c = try DistConnection.connect(host: host, port: port, queue: queue)
            if expectHello { _ = try await c.readFrame() }
            downstream[key] = c
            return c
        }

        do {
            let hello = DistHello(modelName: engine.modelName, layerStart: config.layerStart,
                                  layerEnd: config.layerEnd, hasOutput: config.hasOutput,
                                  nLayers: engine.nLayers, contextSize: engine.contextSize)
            try await conn.sendFrame(.hello, hello.encoded())

            while true {
                let (type, payload) = try await conn.readFrame()
                guard type == .work, let work = DistWork.decode(payload) else { continue }

                // Serialize compute: one chunk at a time against the shard.
                // The chunk's hc holds nTokens states; split, run, re-concat.
                let stateLen = engine.hcStateCount
                let n = max(1, work.nTokens)
                let outStates: [[Float]] = try await gate.run {
                    var hcs: [[Float]] = []
                    hcs.reserveCapacity(n)
                    for i in 0..<n { hcs.append(Array(work.hc[i*stateLen..<(i+1)*stateLen])) }
                    return try self.engine.forwardSliceBatch(hcs: hcs, posBase: work.pos,
                                                             start: work.layerStart, end: work.layerEnd)
                }

                let isTerminal = work.route.isEmpty || work.routeIndex >= work.route.count - 1
                if isTerminal {
                    // Terminal hop: produce logits for the chunk's LAST token if asked,
                    // else hidden states (relay) / a bare ack (forwarding flow control).
                    let result: DistResult
                    if work.flags.contains(.outputLogits) {
                        result = DistResult(kind: .logits, bits: 32, values: try engine.head(hc: outStates[n-1]))
                    } else if work.route.isEmpty {
                        result = DistResult(kind: .hidden, bits: work.hcBits,
                                            values: outStates.flatMap { $0 })
                    } else {
                        result = DistResult(kind: .ack, bits: 32, values: [])
                    }
                    if work.route.isEmpty {
                        try await conn.sendFrame(.result, result.encoded())     // relay: reply upstream
                    } else {
                        let back = try await outbound(work.returnHost, work.returnPort, expectHello: false)
                        try await back.sendFrame(.result, result.encoded())     // forwarding: reply to coordinator
                    }
                } else {
                    // Forward the chunk to the next hop in the route.
                    let nextIdx = work.routeIndex + 1
                    let next = work.route[nextIdx]
                    let fwd = DistWork(pos: work.pos, nTokens: n,
                                       layerStart: next.layerStart, layerEnd: next.layerEnd,
                                       flags: work.flags, hcBits: work.hcBits,
                                       route: work.route, routeIndex: nextIdx,
                                       returnHost: work.returnHost, returnPort: work.returnPort,
                                       hc: outStates.flatMap { $0 })
                    let c = try await outbound(next.host, next.port, expectHello: true)
                    try await c.sendFrame(.work, fwd.encoded())
                }
            }
        } catch {
            onLog("sessione chiusa: \(error)\n")
            conn.cancel()
        }
    }
}

/// Serializes async closures (the shard runs one step at a time).
actor DistGate {
    func run<T: Sendable>(_ body: @Sendable () throws -> T) async rethrows -> T { try body() }
}
