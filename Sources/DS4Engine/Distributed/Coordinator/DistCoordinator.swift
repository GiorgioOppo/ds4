import Foundation
import DS4Core

/// The distributed COORDINATOR: owns the embedding, the sampling loop and the
/// conversation. `connect()` establishes a persistent route to the workers,
/// ASSIGNING each one its job (gguf, settings, layer slice); each `send(...)`
/// re-renders the WHOLE conversation and runs it across the cluster, streaming
/// the reply. Re-rendering keeps correctness trivially checkable — and when
/// the render EXACTLY extends the prefix committed by the last clean turn, the
/// prefill covers only the suffix (in-memory reuse), else a disk restore is
/// negotiated across the shards. Any mismatch falls back to a cold prefill.
///
/// Transports: RELAY (default; coordinator round-trips each chunk through every
/// worker) or FORWARDING (`forward:true`; workers pass the HC state worker→worker
/// and the terminal worker replies to this coordinator's return listener).
public final class DistCoordinator: @unchecked Sendable {
    public struct Peer: Sendable {
        public var host: String
        public var port: UInt16
        public init(host: String, port: UInt16) { self.host = host; self.port = port }
    }
    public struct Config: Sendable {
        public var modelPath: String
        public var contextSize: Int
        public var peers: [Peer]
        public var activationBits: Int
        public var prefillChunk: Int
        public var forward: Bool
        public var returnHost: String
        public var returnPort: UInt16
        /// Expert slot-cache budget ASSIGNed to each worker (0 = disabled).
        public var workerCacheSlots: Int
        /// Disk-KV token budget ASSIGNed to each worker's shard (0 = disabled).
        public var diskKVBudgetTokens: Int
        /// Workers should use the expert-bundle sidecar (the coordinator's
        /// bundle is offered for transfer when it exists on disk).
        public var useExpertBundle: Bool
        /// Workers should use the Q4 dense requant; the coordinator's cache
        /// file is offered so no worker pays the minutes-long requant again.
        public var useDenseQ4: Bool
        public init(modelPath: String, contextSize: Int, peers: [Peer], activationBits: Int,
                    prefillChunk: Int = 32, forward: Bool = false,
                    returnHost: String = "", returnPort: UInt16 = 9099,
                    workerCacheSlots: Int = 0, diskKVBudgetTokens: Int = 0,
                    useExpertBundle: Bool = false, useDenseQ4: Bool = false) {
            self.modelPath = modelPath; self.contextSize = contextSize
            self.peers = peers; self.activationBits = activationBits
            self.prefillChunk = max(1, prefillChunk); self.forward = forward
            self.returnHost = returnHost; self.returnPort = returnPort
            self.workerCacheSlots = max(0, workerCacheSlots)
            self.diskKVBudgetTokens = max(0, diskKVBudgetTokens)
            self.useExpertBundle = useExpertBundle
            self.useDenseQ4 = useDenseQ4
        }
    }

    let engine: DistEngine
    let config: Config
    let queue = DispatchQueue(label: "ds4.dist.coord")

    // Persistent session state (set by connect, used by send).
    var conns: [DistConnection] = []
    var entries: [DistRouteEntry] = []
    var returnListener: DistReturnListener?
    var returnIter: AsyncStream<DistResult>.Iterator?
    /// Per-turn session id, echoed by workers in every RESULT. A turn abandoned
    /// mid-chunk (Stop) leaves its reply in a TCP/listener buffer; the next
    /// turn's id differs, so the stale frame is discarded instead of being
    /// mistaken for the new turn's first reply. Sends are serialized by the
    /// caller (one chat turn / benchmark at a time), so a plain counter is enough.
    var sessionCounter: UInt32 = 0

    /// KV continuity across turns. `committedIds` are the tokens whose KV every
    /// worker shard holds after the last CLEAN turn; `kvValid` mirrors the local
    /// engine's dirty-until-clean rule (false during a turn, after Stop/error,
    /// and around a benchmark — the NSA compressor is recurrent and cannot
    /// rewind, so a partial turn invalidates the whole prefix).
    var committedIds: [Int] = []
    var kvValid = false
    /// Tokens covered by the last disk checkpoint (interval gate, like local).
    var lastDiskStoreCount = 0

    /// Peer verticali attivi: connessione + mask (bit e = esperto posseduto).
    var expertPeers: [(conn: DistConnection, mask: [Bool])] = []
    /// Backbone denso locale (route/attention/KV/head) con la FFN routed
    /// instradata sui peer. nil finché connectVertical non completa.
    var verticalEngine: DistEngine?
    let seqLock = NSLock()
    var seqCounter: UInt32 = 0
    public var verticalReady: Bool { verticalEngine != nil }

    public var routeSummary: String { "\(engine.nLayers) layers · \(entries.count) workers" }

    public init(config: Config) throws {
        self.config = config
        self.engine = try DistEngine(modelPath: config.modelPath, contextSize: config.contextSize,
                                     kvLayers: 0..<0)   // pure coordinator: embed + head only
    }
}
