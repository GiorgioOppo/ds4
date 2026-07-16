import Foundation
@preconcurrency import Network

/// A distributed WORKER: starts IDLE (listening, no model loaded) and receives
/// EVERYTHING from the COORDINATOR — the files (gguf + sidecar, via the
/// hash-verified FILE OFFER/transfer; the huge copy happens once, afterwards
/// the managed-store manifest answers in milliseconds) and the job (ASSIGN:
/// context size, cache budgets, bundle on/off, layer slice). On ASSIGN the
/// worker loads (or reuses) its slice engine and replies READY; then it serves
/// WORK frames (run the slice, or the output head when flagged) and answers
/// with RESULT frames.
public final class DistWorker: @unchecked Sendable {
    public struct Config: Sendable {
        public var port: UInt16
        /// Local gguf hint: the worker tries the coordinator's path verbatim
        /// first, then a file with the assigned NAME next to this hint, then
        /// the hint itself when the filename matches (per-Mac disk layouts).
        public var localModelPath: String
        public init(port: UInt16, localModelPath: String) {
            self.port = port; self.localModelPath = localModelPath
        }
    }

    let config: Config
    let onLog: @Sendable (String) -> Void
    /// Best-effort metadata from the optional local hint. Zero means that the
    /// worker is genuinely geometry-agnostic until the coordinator assigns and
    /// transfers a model.
    let localModelLayers: Int
    let queue = DispatchQueue(label: "ds4.dist.worker")
    let gate = DistGate()
    var listener: NWListener?

    /// The coordinator-defined job this worker currently serves.
    struct Assignment: Equatable, Sendable {
        var resolvedModelPath: String
        var contextSize: Int
        var expertCacheSlots: Int
        var useExpertBundle: Bool
        var useDenseQ4: Bool
        var layerStart: Int
        var layerEnd: Int
        var hasOutput: Bool
    }
    /// Engine + assignment, set by ASSIGN. Guarded by `stateLock`: WORK frames
    /// can arrive on a different connection (forwarding) than the assigning one.
    let stateLock = NSLock()
    var engine: DistEngine?
    var assignment: Assignment?
    var loadingAssignment = false
    /// L'assegnazione del load in corso (v8): un retry del coordinatore con la
    /// STESSA assegnazione si aggancia al load invece di ricevere "busy".
    var pendingAssignment: Assignment?
    /// Expert parallelism (Fase B): lo shard verticale di esperti, alternativo
    /// all'assegnazione a layer (un worker fa l'uno o l'altro).
    var expertShard: ExpertShardEngine?
    var loadingShard = false
    /// Where this shard persists its usage imatrix (slice-keyed: counts are
    /// collected only for the owned layers). nil until assigned.
    var usageFile: URL?
    /// One TURN at a time, enforced at the session level (NOT per connection:
    /// in forwarding mode a worker legitimately holds one connection from the
    /// coordinator AND one from the previous worker). A pos==0 chunk ADOPTS its
    /// session id as current; any chunk from a different session is refused
    /// with an ERROR frame — a competing coordinator fails loudly instead of
    /// silently resetting the active turn's KV shard.
    let sessionLock = NSLock()
    var currentSession: UInt32?
    public init(config: Config, onLog: @escaping @Sendable (String) -> Void) {
        self.config = config
        self.onLog = onLog
        self.localModelLayers = (try? DistEngine.inspectLayout(
            modelPath: config.localModelPath))?.nLayers ?? 0
    }
}
