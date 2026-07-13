import Foundation
import DS4Core
@preconcurrency import Network

extension DistWorker {
    /// App Support home for worker-side per-shard state (usage + disk KV),
    /// keyed by model AND slice: a checkpoint holds only the shard's layers,
    /// so a changed slice must never see another slice's entries.
    static func shardStateDirectory(modelName: String, layerStart: Int, layerEnd: Int) -> URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("DwarfStar/dist-worker/\(modelName)-\(layerStart)-\(layerEnd)",
                                    isDirectory: true)
    }

    /// Persist the usage collected so far (cheap JSON; called between turns).
    func persistUsage() {
        stateLock.lock()
        let file = usageFile
        let data = engine?.usageData()
        stateLock.unlock()
        guard let file, let data else { return }
        try? FileManager.default.createDirectory(at: file.deletingLastPathComponent(),
                                                 withIntermediateDirectories: true)
        try? data.write(to: file)
    }

    func admit(_ work: DistWork) -> Bool {
        sessionLock.lock(); defer { sessionLock.unlock() }
        // turnStart (not pos==0): with KV reuse/restore a turn may begin
        // mid-context, and it is still the legitimate start of a new turn.
        if work.flags.contains(.turnStart) { currentSession = work.session; return true }
        return currentSession == work.session
    }

    public func start() throws {
        let params = NWParameters.tcp
        params.allowLocalEndpointReuse = true
        guard let port = NWEndpoint.Port(rawValue: config.port) else { throw DistError.badPort }
        let l = try NWListener(using: params, on: port)
        l.stateUpdateHandler = { [onLog, config] state in
            switch state {
            case .ready: onLog("worker in ascolto su :\(config.port) — in attesa dell'assegnazione dal coordinatore\n")
            case .failed(let e): onLog("worker listener failed: \(e)\n")
            default: break
            }
        }
        l.newConnectionHandler = { [weak self] c in self?.accept(c) }
        l.start(queue: queue)
        listener = l
    }

    public func stop() {
        persistUsage()               // keep what this shard learned across sessions
        listener?.cancel(); listener = nil
        stateLock.lock(); engine = nil; assignment = nil; usageFile = nil; stateLock.unlock()
    }

    /// Resolve the gguf named by an ASSIGN to a LOCAL file: the coordinator's
    /// path verbatim (shared disk layouts), else a file with the same name in
    /// the local hint's directory, else the hint itself when the name matches.
    static func resolveModelPath(requestedPath: String, modelName: String,
                                 localHint: String,
                                 exists: (String) -> Bool = { FileManager.default.fileExists(atPath: $0) })
        -> String? {
        if !requestedPath.isEmpty, exists(requestedPath) { return requestedPath }
        if !modelName.isEmpty {
            let sibling = ((localHint as NSString).deletingLastPathComponent as NSString)
                .appendingPathComponent(modelName)
            if exists(sibling) { return sibling }
            if (localHint as NSString).lastPathComponent == modelName, exists(localHint) {
                return localHint
            }
        }
        return nil
    }

    func accept(_ c: NWConnection) {
        c.start(queue: queue)
        let conn = DistConnection(c)
        Task { [weak self] in await self?.serve(conn) }
    }

    /// Snapshot for HELLO/READY: the active assignment, or the idle state.
    func helloPayload() -> DistHello {
        stateLock.lock(); defer { stateLock.unlock() }
        if let engine, let a = assignment {
            return DistHello(modelName: engine.modelName, layerStart: a.layerStart,
                             layerEnd: a.layerEnd, hasOutput: a.hasOutput,
                             nLayers: engine.nLayers, contextSize: a.contextSize)
        }
        return .idle(localModelName: (config.localModelPath as NSString).lastPathComponent,
                     nLayers: DistEngine.modelLayers)
    }

    /// Current (engine, assignment) if ready to serve WORK; nil while idle/loading.
    func activeEngine() -> (engine: DistEngine, assignment: Assignment)? {
        stateLock.lock(); defer { stateLock.unlock() }
        guard let engine, let assignment, !loadingAssignment else { return nil }
        return (engine, assignment)
    }

}
