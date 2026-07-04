import Foundation
import DS4Engine

/// One benchmark point: prefill + generation throughput at a context frontier.
struct BenchRow: Identifiable {
    let id = UUID()
    let ctxTokens: Int
    let prefillTps: Double
    let genTps: Double
    let kvcacheBytes: Int64
}

/// Which engine the benchmark exercises: the in-process local engine, or the
/// already-connected distributed cluster (Chat → Distribuito).
enum BenchMode: String, CaseIterable, Identifiable {
    case local = "Local"
    case distributed = "Distributed"
    var id: String { rawValue }
}

/// Native benchmark: measures prefill + generation throughput across context
/// frontiers (replaces the removed `ds4-bench` subprocess). In **Locale** mode it
/// reuses the single in-process engine loaded in Settings; in **Distribuito**
/// mode it reuses the coordinator already connected in `DistributedController` —
/// no second connection, no second model copy.
@MainActor
@Observable
final class BenchController {
    let settings: AppSettings
    let dist: DistributedController
    let store: ChatStore
    var modelPath: String { settings.modelPath }
    var contextSize: Int { settings.contextSize }

    init(settings: AppSettings, dist: DistributedController, store: ChatStore) {
        self.settings = settings
        self.dist = dist
        self.store = store
    }
    var mode: BenchMode = .local
    var ctxStart = 512
    var ctxMax = 4096
    var stepIncr = 512
    var genTokens = 32

    var rows: [BenchRow] = []
    var log = ""
    var isRunning = false
    /// Which engine the in-flight run is actually using (nil when idle). Drives the
    /// "running on Local / Distributed" indicator; authoritative even though the
    /// mode picker is locked during a run.
    var runningMode: BenchMode?

    /// Whether a distributed benchmark is possible right now (route connected, idle).
    var distConnected: Bool { dist.connectedCoordinator != nil }
    var distRoute: String { dist.connectedCoordinator?.routeSummary ?? "not connected" }

    /// Human label for the engine currently running (nil when idle).
    var runningLabel: String? {
        switch runningMode {
        case .local:       return "Local (in-process engine)"
        case .distributed: return "Distributed · \(distRoute)"
        case nil:          return nil
        }
    }

    private var benchWork: Task<String?, Never>?
    private var work: Task<Void, Never>?
    private var logTask: Task<Void, Never>?
    private var rowTask: Task<Void, Never>?

    func run() {
        guard !isRunning else { return }
        rows = []; log = ""; isRunning = true; runningMode = mode
        let gen = genTokens
        let frontiers = stride(from: ctxStart, through: max(ctxStart, ctxMax),
                               by: max(1, stepIncr)).map { $0 }
        switch mode {
        case .local:       runLocal(frontiers: frontiers, gen: gen)
        case .distributed: runDistributed(frontiers: frontiers, gen: gen)
        }
    }

    /// Local in-process benchmark on THE shared engine (the one loaded in
    /// Settings). It never loads a second model: a second engine doubles the
    /// resident Q4 + mlocked buffers and OOM-crashes on a 16 GB Mac. The run
    /// rewrites the shared KV — the next chat turn rebuilds from committed ids.
    private func runLocal(frontiers: [Int], gen: Int) {
        guard let svc = store.benchmarkService else {
            log = store.isReady
                ? "The engine is generating. Wait for the chat turn to finish, then run the benchmark.\n"
                : "No model loaded. Load the model in Settings first — the benchmark reuses that single engine.\n"
            isRunning = false; runningMode = nil
            return
        }
        let (logCont, rowCont) = makeChannels()
        let onLog: @Sendable (String) -> Void = { logCont.yield($0) }

        // Heavy work off the main actor; the actor reference is Sendable.
        let benchWork = Task.detached(priority: .userInitiated) { () -> String? in
            do {
                onLog("Running on the shared engine (no second model copy)...\n")
                for c in frontiers {
                    try Task.checkCancellation()
                    onLog("context \(c): prefill + \(gen) tokens...\n")
                    let p = try await svc.benchmark(contextTokens: c, genTokens: gen)
                    rowCont.yield(BenchRow(ctxTokens: p.contextTokens, prefillTps: p.prefillTps,
                                           genTps: p.genTps, kvcacheBytes: Int64(p.kvBytes)))
                    onLog(String(format: "  ctx %d · prefill %.1f t/s · gen %.2f t/s\n",
                                 p.contextTokens, p.prefillTps, p.genTps))
                }
                return nil
            } catch is CancellationError { return nil }
            catch { return "\(error)" }
        }
        finish(benchWork: benchWork, logCont: logCont, rowCont: rowCont)
    }

    /// Distributed benchmark: reuse the coordinator already connected in
    /// `DistributedController` (same route, same loaded weights). Refuses if no
    /// route is connected or a chat generation is in flight (it resets cluster KV).
    private func runDistributed(frontiers: [Int], gen: Int) {
        guard let coord = dist.connectedCoordinator else {
            log = "No coordinator connected. Open Chat -> Distributed and press Connect before running a distributed benchmark.\n"
            isRunning = false; runningMode = nil
            return
        }
        guard !dist.isGenerating else {
            log = "The coordinator is generating a response. Wait or stop distributed chat before benchmarking.\n"
            isRunning = false; runningMode = nil
            return
        }
        dist.benchmarkActive = true          // lock out chat turns on the shared route
        let (logCont, rowCont) = makeChannels()
        let onLog: @Sendable (String) -> Void = { logCont.yield($0) }

        let benchWork = Task.detached(priority: .userInitiated) { () -> String? in
            do {
                onLog("Distributed benchmark on the already-connected route (\(coord.routeSummary))...\n")
                for c in frontiers {
                    try Task.checkCancellation()
                    onLog("context \(c): prefill + \(gen) tokens...\n")
                    let p = try await coord.benchmark(contextTokens: c, genTokens: gen)
                    rowCont.yield(BenchRow(ctxTokens: p.contextTokens, prefillTps: p.prefillTps,
                                           genTps: p.genTps, kvcacheBytes: Int64(p.kvBytes)))
                    onLog(String(format: "  ctx %d · prefill %.1f t/s · gen %.2f t/s\n",
                                 p.contextTokens, p.prefillTps, p.genTps))
                }
                return nil
            } catch is CancellationError { return nil }
            catch { return "\(error)" }
        }
        finish(benchWork: benchWork, logCont: logCont, rowCont: rowCont) { [weak self] in
            self?.dist.benchmarkActive = false
        }
    }

    /// Wire the log/row AsyncStreams into `self` and return their continuations.
    private func makeChannels() -> (AsyncStream<String>.Continuation, AsyncStream<BenchRow>.Continuation) {
        let (logStream, logCont) = AsyncStream<String>.makeStream()
        let (rowStream, rowCont) = AsyncStream<BenchRow>.makeStream()
        logTask?.cancel(); rowTask?.cancel()
        logTask = Task { [weak self] in for await s in logStream { self?.log += s } }
        rowTask = Task { [weak self] in for await r in rowStream { self?.rows.append(r) } }
        return (logCont, rowCont)
    }

    /// Drain the work task, report any error, run `onComplete`, and clear the flag.
    private func finish(benchWork: Task<String?, Never>,
                        logCont: AsyncStream<String>.Continuation,
                        rowCont: AsyncStream<BenchRow>.Continuation,
                        onComplete: @escaping @MainActor () -> Void = {}) {
        self.benchWork = benchWork
        work = Task {
            if let err = await benchWork.value { logCont.yield("error: \(err)\n") }
            logCont.finish(); rowCont.finish()
            onComplete()
            self.isRunning = false; self.runningMode = nil
        }
    }

    func stop() {
        benchWork?.cancel(); benchWork = nil
        log += "\n[stopping...]\n"
    }
}
