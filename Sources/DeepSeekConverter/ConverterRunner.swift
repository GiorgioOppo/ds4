import Foundation

/// Implementation of `Converter.runQuantize`. Spawns the existing
/// `converter` CLI executable and translates its stdout into a
/// stream of `ConversionEvent`s. The CLI is the authoritative
/// implementation; this wrapper lets SwiftUI (or anything else)
/// drive a conversion through a clean async/progress API instead
/// of having to manage subprocess plumbing inline.
///
/// Future iteration can replace the spawn with a fully native call
/// (migrating the imperative body from `Sources/converter/main.swift`)
/// without changing this function's signature, so callers continue
/// to work.
public enum ConverterRunner {

    /// Resolve the `converter` binary location. Resolution order:
    ///   1. The explicit `overridePath` argument (UI Settings field).
    ///   2. Bundle.main.bundleURL/Contents/Resources/converter
    ///      (set by an Xcode Copy Files build phase when shipping
    ///      the .app — empty in dev right now).
    ///   3. The directory containing the currently-running
    ///      executable (i.e. when the UI was invoked via
    ///      `swift run DeepSeekUI` from the repo's `.build/debug`).
    ///   4. The package's `.build/release/converter` and
    ///      `.build/debug/converter` paths relative to the CWD.
    ///   5. `/usr/local/bin/converter` (user installed).
    ///
    /// Returns `nil` if no plausible binary is found.
    public static func locateConverterBinary(overridePath: String? = nil) -> URL? {
        let fm = FileManager.default

        if let p = overridePath, !p.isEmpty {
            let u = URL(fileURLWithPath: p)
            if fm.isExecutableFile(atPath: u.path) { return u }
        }

        var candidates: [URL] = []

        // (2) Bundle Resources/converter
        candidates.append(Bundle.main.bundleURL
            .appendingPathComponent("Contents/Resources/converter"))
        // (3) Sibling of the running executable
        candidates.append(Bundle.main.bundleURL
            .deletingLastPathComponent()
            .appendingPathComponent("converter"))
        // (4) Common SwiftPM build outputs from current working dir
        let cwd = URL(fileURLWithPath: fm.currentDirectoryPath)
        candidates.append(cwd.appendingPathComponent(".build/release/converter"))
        candidates.append(cwd.appendingPathComponent(".build/debug/converter"))
        candidates.append(cwd.appendingPathComponent(".build/arm64-apple-macosx/release/converter"))
        candidates.append(cwd.appendingPathComponent(".build/arm64-apple-macosx/debug/converter"))
        // (5) System install
        candidates.append(URL(fileURLWithPath: "/usr/local/bin/converter"))

        for u in candidates where fm.isExecutableFile(atPath: u.path) {
            return u
        }
        return nil
    }

    /// Spawn-and-stream implementation. Synchronous from the
    /// async caller's perspective: blocks until the subprocess
    /// exits or is cancelled. Each stdout line becomes a
    /// `.log(...)` event; a coarse `.progress` event is emitted on
    /// the well-known phase strings the CLI prints
    /// ("Indexing input", "Collected N tensors", "Packing N tensors
    /// into M shard(s)", "[i/total] filename — ..." and "Done.").
    public static func runQuantizeViaSubprocess(
        spec: QuantizeSpec,
        binaryURL: URL,
        cancellation: CancellationToken,
        onEvent: @escaping @Sendable (ConversionEvent) -> Void
    ) throws {
        // Pre-flight: source dir exists, output dir creatable.
        let fm = FileManager.default
        guard fm.fileExists(atPath: spec.hfPath.path) else {
            throw NSError(
                domain: "ConverterRunner", code: 10,
                userInfo: [NSLocalizedDescriptionKey:
                    "Source directory does not exist: \(spec.hfPath.path)"])
        }
        try fm.createDirectory(at: spec.savePath,
                                withIntermediateDirectories: true)

        let process = Process()
        process.executableURL = binaryURL
        process.arguments = [
            "--hf-ckpt-path", spec.hfPath.path,
            "--save-path",    spec.savePath.path,
            "--n-experts",    String(spec.nExperts),
            "--model-parallel", String(spec.modelParallel),
            "--target-dtype", spec.target.rawValue,
            "--shard-size-gb", String(format: "%.3f", spec.shardSizeGB),
        ]

        let outPipe = Pipe()
        let errPipe = Pipe()
        process.standardOutput = outPipe
        process.standardError = errPipe

        // Streaming readers: feed every newline-terminated chunk to
        // onEvent. The handler runs on Pipe's internal queue, so we
        // keep it short — just buffer + split + emit.
        let outBuf = LineBuffer()
        let errBuf = LineBuffer()
        outPipe.fileHandleForReading.readabilityHandler = { fh in
            let data = fh.availableData
            if data.isEmpty { return }
            for line in outBuf.feed(data) {
                onEvent(.log(line))
                Self.matchProgress(line: line, onEvent: onEvent)
            }
        }
        errPipe.fileHandleForReading.readabilityHandler = { fh in
            let data = fh.availableData
            if data.isEmpty { return }
            for line in errBuf.feed(data) {
                onEvent(.log("[stderr] " + line))
            }
        }

        try process.run()

        // Poll cancellation while the process runs.
        while process.isRunning {
            if cancellation.isCancelled {
                process.terminate()
                // Give the process a moment to exit cleanly.
                let deadline = Date().addingTimeInterval(2.0)
                while process.isRunning && Date() < deadline {
                    Thread.sleep(forTimeInterval: 0.05)
                }
                outPipe.fileHandleForReading.readabilityHandler = nil
                errPipe.fileHandleForReading.readabilityHandler = nil
                throw ConversionCancelled()
            }
            Thread.sleep(forTimeInterval: 0.1)
        }

        // Drain any final bytes the readabilityHandler might have
        // missed (Apple's pipe API doesn't guarantee one last
        // callback after termination).
        outPipe.fileHandleForReading.readabilityHandler = nil
        errPipe.fileHandleForReading.readabilityHandler = nil
        let tailOut = outPipe.fileHandleForReading.readDataToEndOfFile()
        if !tailOut.isEmpty {
            for line in outBuf.feed(tailOut) {
                onEvent(.log(line))
                Self.matchProgress(line: line, onEvent: onEvent)
            }
        }
        let tailErr = errPipe.fileHandleForReading.readDataToEndOfFile()
        if !tailErr.isEmpty {
            for line in errBuf.feed(tailErr) {
                onEvent(.log("[stderr] " + line))
            }
        }

        guard process.terminationStatus == 0 else {
            throw NSError(
                domain: "ConverterRunner", code: 11,
                userInfo: [NSLocalizedDescriptionKey:
                    "converter exited with status \(process.terminationStatus). See log for details."])
        }

        // Best-effort output-size accounting (sums all
        // .safetensors in savePath).
        var outBytes: UInt64 = 0
        if let it = fm.enumerator(at: spec.savePath,
                                    includingPropertiesForKeys: [.fileSizeKey]) {
            for case let url as URL in it where url.pathExtension == "safetensors" {
                let s = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
                outBytes &+= UInt64(s)
            }
        }
        onEvent(.finished(outputBytes: outBytes))
    }

    /// Heuristics over the CLI's existing print statements. We
    /// don't have structured progress yet — the CLI prints a few
    /// well-known phase markers, so we match those to drive a
    /// coarse progress bar in the UI.
    private static func matchProgress(line: String,
                                        onEvent: (ConversionEvent) -> Void) {
        if line.hasPrefix("Converter: ") && line.contains("input shard(s)") {
            let n = Int(line
                .replacingOccurrences(of: "Converter: ", with: "")
                .components(separatedBy: " ").first ?? "0") ?? 0
            onEvent(.discovered(inputShards: n, inputBytes: 0))
            onEvent(.progress(completed: 0, total: max(n, 1),
                               phase: "Plan", note: nil))
        }
        else if line.hasPrefix("Collected ") && line.contains("tensors.") {
            onEvent(.progress(completed: 1, total: 1,
                               phase: "Plan", note: line))
        }
        else if line.hasPrefix("Packing ") {
            onEvent(.progress(completed: 0, total: 1,
                               phase: "Pack", note: line))
        }
        else if let m = line.range(of: #"^\s*\[(\d+)/(\d+)\]"#,
                                    options: .regularExpression) {
            let bracket = String(line[m]).trimmingCharacters(
                in: CharacterSet(charactersIn: "[] "))
            let nums = bracket.split(separator: "/").compactMap { Int($0) }
            if nums.count == 2 {
                onEvent(.progress(completed: nums[0], total: nums[1],
                                   phase: "Write", note: line))
            }
        }
        else if line == "Done." {
            onEvent(.progress(completed: 1, total: 1,
                               phase: "Finalize", note: nil))
        }
    }
}

/// Splits an incremental byte stream into newline-terminated UTF-8
/// strings. Holds the last partial line until more data arrives.
final class LineBuffer {
    private var pending = Data()

    func feed(_ chunk: Data) -> [String] {
        pending.append(chunk)
        var lines: [String] = []
        while let nlRange = pending.firstRange(of: Data([0x0A])) {
            let lineData = pending.subdata(in: 0..<nlRange.lowerBound)
            pending.removeSubrange(0..<nlRange.upperBound)
            if let s = String(data: lineData, encoding: .utf8) {
                // Strip trailing \r for stray CRLF.
                lines.append(s.hasSuffix("\r") ? String(s.dropLast()) : s)
            }
        }
        return lines
    }
}
