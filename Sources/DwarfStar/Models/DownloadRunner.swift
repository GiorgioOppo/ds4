import Foundation
import DS4Engine

/// Downloads a GGUF model with the NATIVE Swift downloader (DS4Engine.ModelDownloader)
/// — resumable HTTP straight from Hugging Face, no shell script / curl subprocess.
/// Files land in `<scriptDir>/gguf`; partial downloads resume.
@MainActor
@Observable
final class DownloadRunner {
    var log = ""
    var isRunning = false
    var currentTarget: String?
    var progress: Double = 0        // 0...1
    var progressText = ""

    private var task: Task<Void, Never>?

    /// Download events multiplexed onto one stream: byte progress + phase messages.
    private enum DLEvent: Sendable { case progress(Int64, Int64); case status(String) }

    func run(target: String, scriptDir: String) {
        guard !isRunning else { return }
        guard let t = ModelDownloader.target(target) else {
            log += "Unknown target: \(target)\n"; return
        }
        let ggufDir = (scriptDir as NSString).appendingPathComponent("gguf")
        log = "Scarico \(t.file)\n→ \(ggufDir)\n(circa \(t.approxGB) GB; i download parziali riprendono)\n"
        currentTarget = target
        isRunning = true
        progress = 0
        progressText = ""

        // Task inherits @MainActor (DownloadRunner is @MainActor), so we consume
        // events and update state directly here; the download runs in an async
        // let child that captures only Sendable values (target, dir, continuation).
        task = Task {
            let (stream, cont) = AsyncStream<DLEvent>.makeStream()
            async let dl: String = {
                let url = try await ModelDownloader.download(
                    target: t, ggufDir: ggufDir,
                    onProgress: { done, total in cont.yield(.progress(done, total)) },
                    onStatus: { msg in cont.yield(.status(msg)) })
                cont.finish()
                return url.path
            }()

            for await ev in stream {
                switch ev {
                case .progress(let done, let total) where total > 0:
                    progress = Double(done) / Double(total)
                    progressText = String(format: "%.1f / %.1f GB", Double(done) / 1e9, Double(total) / 1e9)
                case .progress:
                    break
                case .status(let msg):
                    log += msg + "\n"
                }
            }

            do {
                let path = try await dl
                log += "\nCompletato: \(path)\n"; progress = 1
            } catch is CancellationError {
                log += "\n[annullato — il file .part resta per riprendere]\n"
            } catch {
                log += "\nErrore: \(error)\n"
            }
            isRunning = false
        }
    }

    func cancel() { task?.cancel() }
}
