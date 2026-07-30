import Foundation
import DS4Engine

/// One benchmark point: prefill + generation throughput at a context frontier.
struct BenchRow: Identifiable, Sendable {
    let id = UUID()
    let ctxTokens: Int
    let prefillTps: Double
    let genTps: Double
    let kvcacheBytes: Int64
}

/// The two measurements exposed by the Benchmark panel. Speed keeps the
/// existing synthetic throughput sweep; Correctness evaluates exact next-token
/// prediction on user-provided text with teacher forcing.
enum BenchKind: String, CaseIterable, Identifiable {
    case speed = "Speed"
    case accuracy = "Correctness"
    var id: String { rawValue }
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
    /// Keep both the live and final charts responsive even when the user asks
    /// for millions of scored tokens across many pieces.
    private static let accuracyChartPointLimit = 4_096

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
    var kind: BenchKind = .speed
    var mode: BenchMode = .local
    var ctxStart = 512
    var ctxMax = 4096
    var stepIncr = 512
    var genTokens = 32

    /// Corpus and deterministic random-sampling bounds for the teacher-forced
    /// next-token benchmark. Every sampled piece gets its own unscored context.
    var accuracyText = """
    Roma nacque, secondo la tradizione, nel 753 avanti Cristo sulle rive del Tevere. La sua posizione favoriva gli scambi tra le comunità dell'interno e i popoli che navigavano lungo la costa. Inizialmente governata da re, la città divenne poi una repubblica fondata su magistrature, assemblee e un senato influente. Nei secoli successivi Roma estese il proprio controllo prima sul Lazio, poi sull'intera penisola italiana e infine sul Mediterraneo. Le guerre puniche contro Cartagine segnarono una svolta decisiva: la vittoria trasformò una potenza regionale in un dominio vastissimo. L'espansione portò ricchezze, schiavi e nuove tensioni sociali. Dopo una lunga stagione di conflitti civili, Augusto inaugurò l'età imperiale e riorganizzò lo Stato. Strade, acquedotti, porti e città collegarono province molto diverse, mentre il diritto romano fornì regole comuni. L'impero attraversò periodi di prosperità e crisi, fino alla divisione tra Oriente e Occidente. Nel 476 l'ultimo imperatore romano d'Occidente fu deposto, ma l'eredità di Roma continuò a vivere nelle lingue, nelle leggi, nell'architettura e nelle istituzioni europee.
    """
    var accuracyMinContextTokens = 32
    var accuracyMaxContextTokens = 256
    var accuracyMaxTokensPerPiece = 128
    var accuracyPieceCount = 4
    var accuracySeed: UInt64 = 0xD54
    var accuracyBucketSize = 16

    /// Upper bound exposed by the UI. The engine may lower the effective range
    /// further when the corpus cannot hold both context and scored targets.
    var accuracyContextLimit: Int { max(1, contextSize - 1) }

    var rows: [BenchRow] = []
    /// Decimated graph samples only. The exact live counters below still see
    /// every observation, and the final result remains exact.
    var accuracyObservations: [InferenceService.AccuracyObservation] = []
    var accuracyLiveEvaluatedTokens = 0
    var accuracyLiveTop1CorrectTokens = 0
    var accuracyLiveTop2CorrectTokens = 0
    var accuracyLiveTop3CorrectTokens = 0
    var accuracyLivePieceIndex = 0
    var accuracyResult: InferenceService.AccuracyResult?
    var log = ""
    var isRunning = false
    /// Which engine the in-flight run is actually using (nil when idle). Drives the
    /// "running on Local / Distributed" indicator; authoritative even though the
    /// mode picker is locked during a run.
    var runningMode: BenchMode?
    /// Like `runningMode`, this snapshots the selected measurement for the
    /// in-flight run so changing UI state can never relabel its results.
    var runningKind: BenchKind?

    /// Whether a distributed benchmark is possible right now (route connected, idle).
    var distConnected: Bool { dist.connectedCoordinator != nil }
    var distRoute: String { dist.connectedCoordinator?.routeSummary ?? "not connected" }

    /// Human label for the engine currently running (nil when idle).
    var runningLabel: String? {
        let suffix = runningKind == .accuracy ? " · Correctness" : ""
        switch runningMode {
        case .local:       return "Local (in-process engine)\(suffix)"
        case .distributed: return "Distributed · \(distRoute)\(suffix)"
        case nil:          return nil
        }
    }

    /// Correctness currently has no efficient distributed implementation: the
    /// distributed protocol returns logits for the last token of a chunk only.
    var accuracyUnavailableForSelectedMode: Bool {
        kind == .accuracy && mode == .distributed
    }

    var accuracyTextIsEmpty: Bool {
        accuracyText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var benchWork: Task<String?, Never>?
    private var work: Task<Void, Never>?
    private var logTask: Task<Void, Never>?
    private var rowTask: Task<Void, Never>?
    private var accuracyObservationTask: Task<Void, Never>?
    private var accuracyResultTask: Task<Void, Never>?
    private var accuracyLiveObservationStride = 1
    private var engineLease: EngineActivityGate.Lease?

    func run() {
        guard !isRunning else { return }
        guard !accuracyUnavailableForSelectedMode else {
            log = "The Correctness benchmark is currently available only with the Local engine.\n"
            return
        }
        guard kind != .accuracy || !accuracyTextIsEmpty else {
            log = "Paste or enter a text corpus before starting the Correctness benchmark.\n"
            return
        }
        let activityGate = EngineActivityGate.shared
        guard let lease = activityGate.acquire(.benchmark) else {
            let owner = activityGate.activeOwner?.displayName ?? "another engine operation"
            log = "The engine is busy with \(owner). Stop it before starting a benchmark.\n"
            return
        }
        engineLease = lease

        log = ""; isRunning = true; runningMode = mode; runningKind = kind
        switch kind {
        case .speed:
            rows = []
            let gen = genTokens
            let frontiers = stride(from: ctxStart, through: max(ctxStart, ctxMax),
                                   by: max(1, stepIncr)).map { $0 }
            switch mode {
            case .local:       runLocal(frontiers: frontiers, gen: gen)
            case .distributed: runDistributed(frontiers: frontiers, gen: gen)
            }
        case .accuracy:
            accuracyObservations = []
            accuracyLiveEvaluatedTokens = 0
            accuracyLiveTop1CorrectTokens = 0
            accuracyLiveTop2CorrectTokens = 0
            accuracyLiveTop3CorrectTokens = 0
            accuracyLivePieceIndex = 0
            accuracyResult = nil
            runAccuracyLocal()
        }
    }

    /// Local in-process benchmark on THE shared engine (the one loaded in
    /// Settings). It never loads a second model: a second engine doubles the
    /// resident Q4 + mlocked buffers and OOM-crashes on a 16 GB Mac. The run
    /// rewrites the shared KV — the next chat turn rebuilds from committed ids.
    private func runLocal(frontiers: [Int], gen: Int) {
        if let svc = store.benchmarkService {
            runLocalDeepSeek(svc: svc, frontiers: frontiers, gen: gen)
        } else if let glm = store.glmBenchmarkService {
            runLocalGLM(svc: glm, frontiers: frontiers, gen: gen)
        } else if let laguna = store.lagunaBenchmarkService {
            runLocalLaguna(svc: laguna, frontiers: frontiers, gen: gen)
        } else {
            log = store.isReady
                ? "The engine is generating. Wait for the chat turn to finish, then run the benchmark.\n"
                : "No model loaded. Load the model in Settings first — the benchmark reuses that single engine.\n"
            clearRunningState()
        }
    }

    private func runLocalDeepSeek(svc: InferenceService, frontiers: [Int], gen: Int) {
        let (logCont, rowCont) = makeChannels()
        let onLog: @Sendable (String) -> Void = { logCont.yield($0) }

        // Heavy work off the main actor; the actor reference is Sendable.
        let benchWork = Task.detached(priority: .userInitiated) { () -> String? in
            do {
                onLog("Running on the shared engine (no second model copy)...\n")
                // No-op se il motore è già caldo: il PRIMO punto del grafico
                // non deve misurare la creazione dei pool esperti (partenza
                // fredda) al posto del regime.
                await svc.warmup()
                for c in frontiers {
                    try Task.checkCancellation()
                    onLog("context \(c): prefill + \(gen) tokens...\n")
                    let p = try await svc.benchmark(contextTokens: c, genTokens: gen)
                    // Il grafico/report usano il throughput equivalente al
                    // p99 della latenza: la coda lenta del decode.
                    rowCont.yield(BenchRow(ctxTokens: p.contextTokens, prefillTps: p.prefillTps,
                                           genTps: p.genTpsP99, kvcacheBytes: Int64(p.kvBytes)))
                    onLog(String(format: "  ctx %d · prefill %.1f t/s · gen 1/p99lat %.2f t/s (media %.2f)\n",
                                 p.contextTokens, p.prefillTps, p.genTpsP99, p.genTps))
                }
                return nil
            } catch is CancellationError { return nil }
            catch { return "\(error)" }
        }
        finish(benchWork: benchWork, logCont: logCont, rowCont: rowCont)
    }

    /// Same sweep on the GLM streaming engine: identical chart semantics
    /// (throughput at p99 decode latency, kv bytes per frontier) plus the per-phase
    /// streaming profile appended to the log for each point.
    private func runLocalGLM(svc: GLM52ChatService, frontiers: [Int], gen: Int) {
        let (logCont, rowCont) = makeChannels()
        let onLog: @Sendable (String) -> Void = { logCont.yield($0) }

        let benchWork = Task.detached(priority: .userInitiated) { () -> String? in
            do {
                onLog("Running on the shared GLM streaming engine (no second model copy)...\n")
                await svc.warmup()
                for c in frontiers {
                    try Task.checkCancellation()
                    onLog("context \(c): prefill + \(gen) tokens...\n")
                    let p = try await svc.benchmark(contextTokens: c, genTokens: gen)
                    rowCont.yield(BenchRow(ctxTokens: p.contextTokens, prefillTps: p.prefillTps,
                                           genTps: p.genTpsP99, kvcacheBytes: Int64(p.kvBytes)))
                    onLog(String(format: "  ctx %d · prefill %.1f t/s · gen 1/p99lat %.2f t/s (media %.2f)\n",
                                 p.contextTokens, p.prefillTps, p.genTpsP99, p.genTps))
                    onLog(p.report.split(separator: "\n")
                        .map { "    \($0)\n" }.joined())
                }
                return nil
            } catch is CancellationError { return nil }
            catch { return "\(error)" }
        }
        finish(benchWork: benchWork, logCont: logCont, rowCont: rowCont)
    }

    /// Same sweep on the Laguna resident/streaming engine: identical chart
    /// semantics (throughput at p99 decode latency, kv bytes per frontier) plus the
    /// per-phase profile appended to the log for each point.
    private func runLocalLaguna(svc: LagunaChatService, frontiers: [Int],
                                gen: Int) {
        let (logCont, rowCont) = makeChannels()
        let onLog: @Sendable (String) -> Void = { logCont.yield($0) }

        let benchWork = Task.detached(priority: .userInitiated) { () -> String? in
            do {
                onLog("Running on the shared Laguna engine (no second model copy)...\n")
                await svc.warmup()
                for c in frontiers {
                    try Task.checkCancellation()
                    onLog("context \(c): prefill + \(gen) tokens...\n")
                    let p = try await svc.benchmark(contextTokens: c, genTokens: gen)
                    rowCont.yield(BenchRow(ctxTokens: p.contextTokens, prefillTps: p.prefillTps,
                                           genTps: p.genTpsP99, kvcacheBytes: Int64(p.kvBytes)))
                    onLog(String(format: "  ctx %d · prefill %.1f t/s · gen 1/p99lat %.2f t/s (media %.2f)\n",
                                 p.contextTokens, p.prefillTps, p.genTpsP99, p.genTps))
                    onLog(p.report.split(separator: "\n")
                        .map { "    \($0)\n" }.joined())
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
            clearRunningState()
            return
        }
        guard !dist.isGenerating else {
            log = "The coordinator is generating a response. Wait or stop distributed chat before benchmarking.\n"
            clearRunningState()
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
                                           genTps: p.genTpsP99, kvcacheBytes: Int64(p.kvBytes)))
                    onLog(String(format: "  ctx %d · prefill %.1f t/s · gen 1/p99lat %.2f t/s (media %.2f)\n",
                                 p.contextTokens, p.prefillTps, p.genTpsP99, p.genTps))
                }
                return nil
            } catch is CancellationError { return nil }
            catch { return "\(error)" }
        }
        finish(benchWork: benchWork, logCont: logCont, rowCont: rowCont) { [weak self] in
            self?.dist.benchmarkActive = false
        }
    }

    /// Teacher-forced exact next-token evaluation on the shared local engine.
    /// The detached task only talks to Sendable AsyncStream continuations; all
    /// observable state is consumed and mutated by MainActor tasks.
    private func runAccuracyLocal() {
        guard mode == .local else {
            log = "The Correctness benchmark is currently available only with the Local engine.\n"
            clearRunningState()
            return
        }
        let deepSeek = store.benchmarkService
        let glm = store.glmBenchmarkService
        let laguna = store.lagunaBenchmarkService
        guard deepSeek != nil || glm != nil || laguna != nil else {
            log = store.isReady
                ? "The engine is generating. Wait for the chat turn to finish, then run the benchmark.\n"
                : "No model loaded. Load the model in Settings first — the benchmark reuses that single engine.\n"
            clearRunningState()
            return
        }

        let text = accuracyText
        let contextLimit = accuracyContextLimit
        accuracyMinContextTokens = min(max(1, accuracyMinContextTokens), contextLimit)
        accuracyMaxContextTokens = min(
            max(accuracyMinContextTokens, accuracyMaxContextTokens),
            contextLimit
        )
        accuracyMaxTokensPerPiece = max(1, accuracyMaxTokensPerPiece)
        accuracyPieceCount = min(256, max(1, accuracyPieceCount))
        accuracyBucketSize = max(1, accuracyBucketSize)

        let minContextTokens = accuracyMinContextTokens
        let maxContextTokens = accuracyMaxContextTokens
        let maxTokensPerPiece = accuracyMaxTokensPerPiece
        let pieceCount = accuracyPieceCount
        let seed = accuracySeed
        let chartPointLimit = Self.accuracyChartPointLimit
        let requestedBucketSize = accuracyBucketSize
        let (evaluationProduct, evaluationOverflow) =
            pieceCount.multipliedReportingOverflow(by: maxTokensPerPiece)
        let maximumEvaluations = evaluationOverflow ? Int.max : evaluationProduct
        let minimumChartBlock = max(
            1,
            maximumEvaluations / chartPointLimit
                + (maximumEvaluations % chartPointLimit == 0 ? 0 : 1)
        )
        let bucketSize = max(requestedBucketSize, minimumChartBlock)
        accuracyLiveObservationStride = max(1, bucketSize)
        let (logCont, observationCont, resultCont) = makeAccuracyChannels()
        let onLog: @Sendable (String) -> Void = { logCont.yield($0) }
        let onObservation: @Sendable (InferenceService.AccuracyObservation) -> Void = {
            observationCont.yield($0)
        }
        // Same parameters, same result type: only the backend call differs.
        let runEvaluation: @Sendable () async throws -> InferenceService.AccuracyResult
        if let deepSeek {
            runEvaluation = {
                try await deepSeek.accuracyBenchmark(
                    text: text, minContextTokens: minContextTokens,
                    maxContextTokens: maxContextTokens,
                    maxTokensPerPiece: maxTokensPerPiece,
                    pieceCount: pieceCount, seed: seed,
                    bucketSize: bucketSize, onObservation: onObservation)
            }
        } else if let glm {
            runEvaluation = {
                try await glm.accuracyBenchmark(
                    text: text, minContextTokens: minContextTokens,
                    maxContextTokens: maxContextTokens,
                    maxTokensPerPiece: maxTokensPerPiece,
                    pieceCount: pieceCount, seed: seed,
                    bucketSize: bucketSize, onObservation: onObservation)
            }
        } else if let laguna {
            runEvaluation = {
                try await laguna.accuracyBenchmark(
                    text: text, minContextTokens: minContextTokens,
                    maxContextTokens: maxContextTokens,
                    maxTokensPerPiece: maxTokensPerPiece,
                    pieceCount: pieceCount, seed: seed,
                    bucketSize: bucketSize, onObservation: onObservation)
            }
        } else {
            clearRunningState()
            return
        }

        let benchWork = Task.detached(priority: .userInitiated) { () -> String? in
            do {
                onLog(
                    "Correctness benchmark on the shared local engine (teacher forcing)...\n" +
                    "  sampling \(pieceCount) random pieces · context \(minContextTokens)...\(maxContextTokens)" +
                    " · up to \(maxTokensPerPiece) evaluated tokens/piece · seed \(seed)" +
                    " · chart block \(bucketSize)\n"
                )
                if bucketSize > requestedBucketSize {
                    onLog("  chart block automatically raised from \(requestedBucketSize) to \(bucketSize) to cap rendering at about \(chartPointLimit) points\n")
                }
                let result = try await runEvaluation()
                resultCont.yield(result)
                onLog(
                    "  sampled \(result.pieces.count)/\(result.requestedPieceCount) pieces" +
                    " · effective context \(result.effectiveMinContextTokens)...\(result.effectiveMaxContextTokens)" +
                    " · seed \(result.seed)\n"
                )
                onLog(String(
                    format: "  Top-1 %d/%d (%.2f%%) · Top-2 %d/%d (%.2f%%) · Top-3 %d/%d (%.2f%%) · %.2f evaluated t/s\n",
                    result.top1CorrectTokens, result.evaluatedTokens, result.top1Accuracy * 100,
                    result.top2CorrectTokens, result.evaluatedTokens, result.top2Accuracy * 100,
                    result.top3CorrectTokens, result.evaluatedTokens, result.top3Accuracy * 100,
                    result.evaluatedTps
                ))
                for piece in result.pieces {
                    onLog(String(
                        format: "  piece %d · source %d · target %d · context %d · evaluated %d · Top-1 %.2f%% · Top-2 %.2f%% · Top-3 %.2f%%%@\n",
                        piece.index + 1, piece.sourceStartTokenIndex,
                        piece.targetStartTokenIndex, piece.contextTokens,
                        piece.evaluatedTokens, piece.top1Accuracy * 100,
                        piece.top2Accuracy * 100, piece.top3Accuracy * 100,
                        piece.truncated ? " · truncated" : ""
                    ))
                }
                if result.truncated {
                    onLog("  Sampling bounds/count were reduced, or one or more pieces were truncated, to fit the corpus and context limit.\n")
                }
                return nil
            } catch is CancellationError {
                return nil
            } catch {
                return "\(error)"
            }
        }
        finishAccuracy(benchWork: benchWork, logCont: logCont,
                       observationCont: observationCont, resultCont: resultCont)
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

    /// Separate typed streams keep engine values Sendable across the detached
    /// boundary while preserving all observable mutations on the main actor.
    private func makeAccuracyChannels() -> (
        AsyncStream<String>.Continuation,
        AsyncStream<InferenceService.AccuracyObservation>.Continuation,
        AsyncStream<InferenceService.AccuracyResult>.Continuation
    ) {
        let (logStream, logCont) = AsyncStream<String>.makeStream()
        let (observationStream, observationCont) =
            AsyncStream<InferenceService.AccuracyObservation>.makeStream()
        let (resultStream, resultCont) = AsyncStream<InferenceService.AccuracyResult>.makeStream()

        logTask?.cancel()
        accuracyObservationTask?.cancel()
        accuracyResultTask?.cancel()
        logTask = Task { [weak self] in
            for await entry in logStream { self?.log += entry }
        }
        accuracyObservationTask = Task { [weak self] in
            for await observation in observationStream {
                guard let self else { continue }
                self.accuracyLiveEvaluatedTokens += 1
                if observation.top1Correct { self.accuracyLiveTop1CorrectTokens += 1 }
                if observation.top2Correct { self.accuracyLiveTop2CorrectTokens += 1 }
                if observation.top3Correct { self.accuracyLiveTop3CorrectTokens += 1 }
                self.accuracyLivePieceIndex = observation.pieceIndex
                if observation.index == 1
                    || observation.index % self.accuracyLiveObservationStride == 0 {
                    self.accuracyObservations.append(observation)
                }
            }
        }
        accuracyResultTask = Task { [weak self] in
            for await result in resultStream { self?.accuracyResult = result }
        }
        return (logCont, observationCont, resultCont)
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
            self.clearRunningState()
        }
    }

    /// Accuracy counterpart of `finish`: close and drain all streams before the
    /// run becomes idle, so the final KPI/chart state is visible atomically.
    private func finishAccuracy(
        benchWork: Task<String?, Never>,
        logCont: AsyncStream<String>.Continuation,
        observationCont: AsyncStream<InferenceService.AccuracyObservation>.Continuation,
        resultCont: AsyncStream<InferenceService.AccuracyResult>.Continuation
    ) {
        self.benchWork = benchWork
        let logTask = self.logTask
        let observationTask = self.accuracyObservationTask
        let resultTask = self.accuracyResultTask
        work = Task {
            if let err = await benchWork.value { logCont.yield("error: \(err)\n") }
            logCont.finish(); observationCont.finish(); resultCont.finish()
            _ = await logTask?.value
            _ = await observationTask?.value
            _ = await resultTask?.value
            self.clearRunningState()
        }
    }

    private func clearRunningState() {
        releaseEngineLease()
        isRunning = false
        runningMode = nil
        runningKind = nil
    }

    private func releaseEngineLease() {
        guard let lease = engineLease else { return }
        engineLease = nil
        EngineActivityGate.shared.release(lease)
    }

    func stop() {
        benchWork?.cancel(); benchWork = nil
        log += "\n[stopping...]\n"
    }
}
