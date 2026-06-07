import Foundation

/// Small reusable wrapper to run a subprocess and stream its merged
/// stdout/stderr as text, with a termination callback. Used by every panel that
/// drives a ds4 binary (download, server, benchmark, diagnostics).
@MainActor
final class ProcessStream {
    private var process: Process?

    var isRunning: Bool { process != nil }

    /// Resolve a possibly-relative path against the GUI's working directory.
    nonisolated static func absolutePath(_ path: String) -> String {
        if (path as NSString).isAbsolutePath { return path }
        let base = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        return URL(fileURLWithPath: path, relativeTo: base).standardizedFileURL.path
    }

    /// Start `executable` with `arguments`. `onOutput` receives output chunks on
    /// the main actor; `onExit` receives the exit status. Returns an error
    /// string if the process could not be started, else nil.
    @discardableResult
    func start(executable: String,
               arguments: [String],
               workingDir: String?,
               onOutput: @escaping @MainActor (String) -> Void,
               onExit: @escaping @MainActor (Int32) -> Void) -> String? {
        guard process == nil else { return "già in esecuzione" }
        let binary = Self.absolutePath(executable)
        guard FileManager.default.fileExists(atPath: binary) else {
            return "non trovato: \(binary)"
        }

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: binary)
        proc.arguments = arguments
        if let workingDir {
            proc.currentDirectoryURL = URL(fileURLWithPath: Self.absolutePath(workingDir))
        }

        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor in onOutput(text) }
        }
        proc.terminationHandler = { proc in
            pipe.fileHandleForReading.readabilityHandler = nil
            Task { @MainActor in
                self.process = nil
                onExit(proc.terminationStatus)
            }
        }

        do {
            try proc.run()
            self.process = proc
            return nil
        } catch {
            return "avvio fallito: \(error)"
        }
    }

    /// Cooperative stop (SIGINT) — lets ds4 binaries drain and persist state.
    func interrupt() { process?.interrupt() }

    /// Forceful stop (SIGTERM).
    func terminate() { process?.terminate() }
}
