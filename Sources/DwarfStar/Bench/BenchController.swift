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

/// Native benchmark: loads its OWN in-process engine and measures prefill +
/// generation throughput across context frontiers (replaces the removed
/// `ds4-bench` subprocess). Weights are mmap-shared with any other loaded engine.
@MainActor
@Observable
final class BenchController {
    let settings: AppSettings
    var modelPath: String { settings.modelPath }      // inherited (Impostazioni)
    var contextSize: Int { settings.contextSize }

    init(settings: AppSettings) { self.settings = settings }
    var ctxStart = 512
    var ctxMax = 4096
    var stepIncr = 512
    var genTokens = 32

    var rows: [BenchRow] = []
    var log = ""
    var isRunning = false

    private var benchWork: Task<String?, Never>?
    private var work: Task<Void, Never>?
    private var logTask: Task<Void, Never>?
    private var rowTask: Task<Void, Never>?

    func run() {
        guard !isRunning else { return }
        rows = []; log = ""; isRunning = true

        let path = ProcessStream.absolutePath(modelPath)
        let ctx = contextSize, gen = genTokens
        let frontiers = stride(from: ctxStart, through: max(ctxStart, ctxMax),
                               by: max(1, stepIncr)).map { $0 }

        let (logStream, logCont) = AsyncStream<String>.makeStream()
        let (rowStream, rowCont) = AsyncStream<BenchRow>.makeStream()
        logTask?.cancel(); rowTask?.cancel()
        logTask = Task { [weak self] in for await s in logStream { self?.log += s } }
        rowTask = Task { [weak self] in for await r in rowStream { self?.rows.append(r) } }
        let onLog: @Sendable (String) -> Void = { logCont.yield($0) }

        // Heavy work off the main actor; results stream back via the channels.
        let benchWork = Task.detached(priority: .userInitiated) { () -> String? in
            do {
                onLog("Caricamento modello…\n")
                let svc = try InferenceService(modelPath: path, contextSize: ctx, systemPrompt: nil)
                for c in frontiers {
                    try Task.checkCancellation()
                    onLog("contesto \(c): prefill + \(gen) token…\n")
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
        self.benchWork = benchWork
        work = Task {
            if let err = await benchWork.value { logCont.yield("errore: \(err)\n") }
            logCont.finish(); rowCont.finish()
            self.isRunning = false
        }
    }

    func stop() {
        benchWork?.cancel(); benchWork = nil
        log += "\n[interruzione…]\n"
    }
}
