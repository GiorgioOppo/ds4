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
        self.engine = try DistEngine(modelPath: config.modelPath, contextSize: config.contextSize)
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
        onLog("coordinatore connesso\n")
        do {
            let hello = DistHello(modelName: engine.modelName, layerStart: config.layerStart,
                                  layerEnd: config.layerEnd, hasOutput: config.hasOutput,
                                  nLayers: engine.nLayers, contextSize: engine.contextSize)
            try await conn.sendFrame(.hello, hello.encoded())

            while true {
                let (type, payload) = try await conn.readFrame()
                guard type == .work, let work = DistWork.decode(payload) else { continue }
                // Serialize compute: one generation step at a time against the shard.
                let result: DistResult = try await gate.run {
                    let outHC = try self.engine.forwardSlice(hc: work.hc, pos: work.pos, nKeys: work.nKeys,
                                                             start: work.layerStart, end: work.layerEnd)
                    if work.flags.contains(.outputLogits) {
                        return DistResult(kind: .logits, bits: 32, values: try self.engine.head(hc: outHC))
                    }
                    return DistResult(kind: .hidden, bits: work.hcBits, values: outHC)
                }
                try await conn.sendFrame(.result, result.encoded())
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
