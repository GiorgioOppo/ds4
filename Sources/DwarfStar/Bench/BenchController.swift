import Foundation

/// One CSV row from ds4-bench:
/// ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,kvcache_bytes
struct BenchRow: Identifiable {
    let id = UUID()
    let ctxTokens: Int
    let prefillTokens: Int
    let prefillTps: Double
    let genTokens: Int
    let genTps: Double
    let kvcacheBytes: Int64
}

/// Runs `ds4-bench` and parses its streamed CSV into chartable rows.
@MainActor
@Observable
final class BenchController {
    var binaryPath = AppEnvironment.binary("ds4-bench")
    var workingDir = AppEnvironment.resourceDir
    var modelPath = AppEnvironment.defaultModelPath
    var promptFile = "speed-bench/promessi_sposi.txt"
    var ctxStart = 2048
    var ctxMax = 16384
    var stepIncr = 2048
    var genTokens = 128

    var rows: [BenchRow] = []
    var log = ""
    var isRunning = false

    private let proc = ProcessStream()
    private var pending = ""

    func run() {
        guard !isRunning else { return }
        rows = []
        log = ""
        pending = ""
        isRunning = true
        let args = ["-m", ProcessStream.absolutePath(modelPath),
                    "--prompt-file", promptFile,
                    "--ctx-start", String(ctxStart),
                    "--ctx-max", String(ctxMax),
                    "--step-incr", String(stepIncr),
                    "--gen-tokens", String(genTokens)]
        let error = proc.start(executable: binaryPath,
                               arguments: args,
                               workingDir: workingDir,
                               onOutput: { [weak self] text in self?.ingest(text) },
                               onExit: { [weak self] status in
                                   self?.log += "\n[exit \(status)]\n"
                                   self?.isRunning = false
                               })
        if let error {
            log += "ds4-bench: \(error)\n"
            isRunning = false
        }
    }

    func stop() { proc.interrupt() }

    private func ingest(_ text: String) {
        log += text
        pending += text
        while let nl = pending.firstIndex(of: "\n") {
            let line = String(pending[..<nl])
            pending = String(pending[pending.index(after: nl)...])
            if let row = Self.parse(line) { rows.append(row) }
        }
    }

    /// Parse a CSV data row; returns nil for the header or any non-data line.
    static func parse(_ line: String) -> BenchRow? {
        let f = line.split(separator: ",", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespaces) }
        guard f.count == 6, let ctx = Int(f[0]) else { return nil }
        return BenchRow(ctxTokens: ctx,
                        prefillTokens: Int(f[1]) ?? 0,
                        prefillTps: Double(f[2]) ?? 0,
                        genTokens: Int(f[3]) ?? 0,
                        genTps: Double(f[4]) ?? 0,
                        kvcacheBytes: Int64(f[5]) ?? 0)
    }
}
