import Foundation

/// Launches and monitors the `ds4-server` binary as a subprocess.
///
/// The server holds its own copy of the model, so it must not run at the same
/// time as the in-process chat engine on a machine where the model fits only
/// once (that would load the weights twice). The UI surfaces that warning.
@MainActor
@Observable
final class ServerController {
    // Configuration.
    var binaryPath = AppEnvironment.binary("ds4-server")
    var workingDir = AppEnvironment.resourceDir   // resolves metal/ and relative paths
    var modelPath = AppEnvironment.defaultModelPath
    var host = "127.0.0.1"
    var port = 8000
    var contextSize = 100_000
    var cors = false
    var kvDiskDir = ""
    var kvDiskSpaceMB = 8192
    var streamingEnabled = false
    var streamingCacheSpec = ""

    // Live state.
    var log = ""
    var isRunning = false

    private let proc = ProcessStream()

    var endpoint: String { "http://\(host):\(port)/v1" }

    func start() {
        guard !isRunning else { return }

        var args = ["-m", ProcessStream.absolutePath(modelPath),
                    "--ctx", String(contextSize),
                    "--host", host,
                    "--port", String(port)]
        if cors { args.append("--cors") }
        if !kvDiskDir.isEmpty {
            args += ["--kv-disk-dir", kvDiskDir, "--kv-disk-space-mb", String(kvDiskSpaceMB)]
        }
        if streamingEnabled {
            args.append("--ssd-streaming")
            if !streamingCacheSpec.isEmpty {
                args += ["--ssd-streaming-cache-experts", streamingCacheSpec]
            }
        }

        log = "$ ds4-server " + args.joined(separator: " ") + "\n"
        isRunning = true
        let error = proc.start(executable: binaryPath,
                               arguments: args,
                               workingDir: workingDir,
                               onOutput: { [weak self] text in self?.log += text },
                               onExit: { [weak self] status in
                                   self?.log += "\n[server terminato, exit \(status)]\n"
                                   self?.isRunning = false
                               })
        if let error {
            log += "ds4-server: \(error)\nCompila con `make` nella cartella del progetto.\n"
            isRunning = false
        }
    }

    /// Cooperative stop (SIGINT): lets the server drain and persist KV state.
    func stop() { proc.interrupt() }
}

/// Opens Terminal and runs the interactive `ds4-agent` in the project directory.
/// The agent is a terminal program (REPL + TUI), so it is launched in a real
/// terminal rather than embedded.
enum AgentLauncher {
    static func openInTerminal(projectDir: String) {
        let dir = ProcessStream.absolutePath(projectDir)
        let escaped = dir.replacingOccurrences(of: "\"", with: "\\\"")
        let script = "tell application \"Terminal\" to do script \"cd \(escaped) && ./ds4-agent\""
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
        proc.arguments = ["-e", script]
        try? proc.run()
    }
}
