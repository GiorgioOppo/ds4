import Foundation

struct MiniToolBenchRunEvent: Decodable, Sendable {
    let event: String
    let message: String?
    let directory: String?
}

struct MiniToolBenchMachine: Decodable, Sendable {
    let name: String
    let running: Bool
    enum CodingKeys: String, CodingKey { case name = "Name", running = "Running" }

    static func preferred(in machines: [Self]) -> Self? {
        machines.first { $0.name == "dwarfstar-bench" }
            ?? machines.first { $0.running }
            ?? machines.first
    }
}

/// Owns only this run's CLI processes. Existing VMs and unrelated containers
/// remain available after completion and cancellation.
@MainActor
final class MiniToolBenchRunner {
    private(set) var isRunning = false
    private var process: MiniToolBenchProcess?
    private var podman: String?
    private var machine: String?
    private var request: Data?
    private var remoteStarted = false
    private var cancelled = false
    private var logHandle: FileHandle?

    func run(configuration: MiniToolBenchConfiguration, apiKey: String? = nil,
             onEvent: @escaping @MainActor (MiniToolBenchRunEvent) -> Void) async throws -> URL? {
        guard !isRunning else { throw MiniToolBenchError.invalid("Una prova Podman è già in corso.") }
        guard ProcessInfo.processInfo.environment["APP_SANDBOX_CONTAINER_ID"] == nil else {
            throw MiniToolBenchError.invalid("La gestione di Podman richiede la build locale di DwarfStar (schema Local o make app). La build App Store può importare i risultati e preparare i comandi manuali.")
        }
        isRunning = true; cancelled = false; remoteStarted = false
        defer {
            isRunning = false; process = nil; request = nil; remoteStarted = false
            try? logHandle?.close(); logHandle = nil
        }
        let binary = try Self.findPodman()
        podman = binary
        let runID = "dwarfstar-" + UUID().uuidString.lowercased()
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let exports = support.appendingPathComponent("DwarfStar/MiniToolBench/runs", isDirectory: true)
        try FileManager.default.createDirectory(at: exports, withIntermediateDirectories: true,
                                                attributes: [.posixPermissions: 0o700])
        let logURL = exports.appendingPathComponent(runID + ".log")
        FileManager.default.createFile(atPath: logURL.path, contents: nil, attributes: [.posixPermissions: 0o600])
        logHandle = try FileHandle(forWritingTo: logURL)
        func emit(_ kind: String, _ message: String) {
            let safe = Self.redact(message, key: apiKey)
            try? logHandle?.write(contentsOf: Data((safe + "\n").utf8))
            onEvent(.init(event: kind, message: safe, directory: nil))
        }

        emit("phase", "Controllo delle macchine Podman…")
        let inventory = try await execute(binary, ["machine", "list", "--format", "json"])
        let machines = try JSONDecoder().decode([MiniToolBenchMachine].self, from: Data(inventory.output.utf8))
        try checkCancellation()
        let selected = MiniToolBenchMachine.preferred(in: machines)
        let name = selected?.name ?? "dwarfstar-bench"
        machine = name
        if selected == nil {
            emit("phase", "Creazione della VM Linux dwarfstar-bench (4 GiB di RAM)…")
            _ = try await execute(binary, ["machine", "init", "--cpus", "4", "--memory", "4096",
                                           "--disk-size", "60", "--volume", exports.path + ":" + exports.path,
                                           name], onLine: { emit("log", $0) })
            try checkCancellation()
        }
        if selected?.running != true {
            emit("phase", "Avvio della VM Linux \(name)…")
            _ = try await execute(binary, ["machine", "start", name], onLine: { emit("log", $0) })
            try checkCancellation()
        } else { emit("log", "Uso la VM già attiva: \(name).") }

        // Send credentials through stdin, never argv, files, or copied commands.
        let config = try JSONSerialization.jsonObject(with: JSONEncoder().encode(configuration))
        var envelope: [String: Any] = ["action": "run", "run_id": runID,
                                       "export_directory": exports.path, "configuration": config]
        if let apiKey, !apiKey.isEmpty { envelope["api_key"] = apiKey }
        let data = try JSONSerialization.data(withJSONObject: envelope)
        request = data
        try checkCancellation()
        var result: URL?
        var remoteError: String?
        remoteStarted = true
        emit("phase", "Preparazione del runner e dei verificatori nella VM…")
        let command = Self.remoteCommand
        let exit = try await execute(binary, ["machine", "ssh", name, command], input: data,
                                     allowFailure: true, onLine: { line in
            if let event = try? JSONDecoder().decode(MiniToolBenchRunEvent.self, from: Data(line.utf8)) {
                if let message = event.message { emit(event.event, message) }
                if event.event == "error" { remoteError = event.message }
                if event.event == "result", let path = event.directory {
                    let url = URL(fileURLWithPath: path).resolvingSymlinksInPath()
                    let expected = exports.appendingPathComponent(runID).resolvingSymlinksInPath()
                    if url == expected { result = url }
                    else { remoteError = "Il runner ha restituito una cartella risultati inattesa." }
                }
            } else if !line.isEmpty { emit("log", line) }
        })
        if exit.code != 0, remoteError == nil { try? await requestRemoteCancellation() }
        remoteStarted = false
        if cancelled || Task.isCancelled { throw CancellationError() }
        if exit.code != 0 || remoteError != nil {
            throw MiniToolBenchError.invalid(remoteError ?? "Il runner si è fermato (codice \(exit.code)). Consulta il log: \(logURL.path)")
        }
        guard let result else {
            throw MiniToolBenchError.invalid("Il runner è terminato senza esportare un report. Log: \(logURL.path)")
        }
        emit("phase", "Test terminati. Importazione dei risultati…")
        return result
    }

    func cancel() async throws {
        guard isRunning else { return }
        cancelled = true
        if remoteStarted {
            try await requestRemoteCancellation()
        } else { process?.interrupt() }
    }

    private func requestRemoteCancellation() async throws {
        if let binary = podman, let name = machine, let data = request {
            var envelope = try JSONSerialization.jsonObject(with: data) as! [String: Any]
            envelope["action"] = "cancel"
            envelope.removeValue(forKey: "api_key")
            let cancelInput = try JSONSerialization.data(withJSONObject: envelope)
            // Separate SSH request: closing the original SSH connection alone
            // would leave the benchmark and its child processes running in Linux.
            _ = try await execute(binary, ["machine", "ssh", name, Self.remoteCommand],
                                  input: cancelInput, auxiliary: true)
        }
    }

    static var remoteCommand: String {
        // Podman hands its command to the remote login shell. Quote once for
        // that shell; Process.arguments avoids an additional local shell layer.
        "python3 -u -c " + MiniToolBenchCommandBuilder.shellQuote(MiniToolBenchBootstrap.script)
    }

    private func checkCancellation() throws {
        if cancelled || Task.isCancelled { throw CancellationError() }
    }

    private static func redact(_ value: String, key: String?) -> String {
        guard let key, !key.isEmpty else { return value }
        return value.replacingOccurrences(of: key, with: "[API key]")
    }

    static func findPodman() throws -> String {
        let candidates = ["/opt/podman/bin/podman", "/opt/homebrew/bin/podman", "/usr/local/bin/podman"]
        if let path = candidates.first(where: { FileManager.default.isExecutableFile(atPath: $0) }) { return path }
        throw MiniToolBenchError.invalid("Podman non è installato. Installa Podman Desktop da podman-desktop.io; poi questo pulsante creerà e preparerà la VM automaticamente.")
    }

    private func execute(_ executable: String, _ arguments: [String], input: Data? = nil,
                         allowFailure: Bool = false, auxiliary: Bool = false,
                         onLine: (@MainActor (String) -> Void)? = nil) async throws -> MiniToolBenchProcess.Result {
        let child = MiniToolBenchProcess()
        if !auxiliary { process = child }
        let (lines, continuation) = AsyncStream<String>.makeStream()
        let delivery = Task { for await line in lines { onLine?(line) } }
        do {
            let value = try await Task.detached(priority: .userInitiated) {
                defer { continuation.finish() }
                return try child.run(executable, arguments, input: input,
                                     timeout: auxiliary ? 30 : nil, output: { continuation.yield($0) })
            }.value
            await delivery.value
            if !auxiliary { process = nil }
            if !allowFailure, value.code != 0 {
                throw MiniToolBenchError.invalid("Podman: \(value.output.suffix(4_000))")
            }
            return value
        } catch {
            continuation.finish(); await delivery.value
            if !auxiliary { process = nil }
            if !auxiliary, remoteStarted { try? await requestRemoteCancellation() }
            throw error
        }
    }
}

/// Blocking pipe reads stay off the main actor and preserve the last output
/// chunk before process termination. A bounded tail is retained for errors.
private final class MiniToolBenchProcess: @unchecked Sendable {
    struct Result: Sendable { let code: Int32; let output: String }
    private let lock = NSLock()
    private var process: Process?
    private var interrupted = false

    func interrupt() {
        lock.withLock {
            interrupted = true
            if let process, process.isRunning { process.interrupt() }
        }
    }

    private func terminate() {
        lock.withLock { if let process, process.isRunning { process.terminate() } }
    }

    func run(_ executable: String, _ arguments: [String], input: Data?, timeout: TimeInterval? = nil,
             output: @Sendable (String) -> Void) throws -> Result {
        let child = Process(), out = Pipe(), stdin = Pipe()
        child.executableURL = URL(fileURLWithPath: executable)
        child.arguments = arguments
        child.standardOutput = out; child.standardError = out; child.standardInput = stdin
        var environment = ProcessInfo.processInfo.environment
        environment["PATH"] = "/opt/podman/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:" + (environment["PATH"] ?? "")
        child.environment = environment
        try lock.withLock {
            if interrupted { throw CancellationError() }
            try child.run(); process = child
        }
        let deadline = DispatchWorkItem { [weak self] in self?.terminate() }
        if let timeout { DispatchQueue.global().asyncAfter(deadline: .now() + timeout, execute: deadline) }
        defer {
            deadline.cancel()
            // A failed stdin write/read must not abandon the owned SSH/Podman
            // process. Closing the connection is also a HUP to the bootstrap;
            // execute() additionally sends a separate durable cancel request.
            try? stdin.fileHandleForWriting.close()
            try? out.fileHandleForReading.close()
            if child.isRunning { child.terminate(); child.waitUntilExit() }
            lock.withLock { process = nil }
        }
        if let input { try stdin.fileHandleForWriting.write(contentsOf: input) }
        try stdin.fileHandleForWriting.close()
        var pending = Data(), tail = ""
        while let data = try out.fileHandleForReading.read(upToCount: 16_384), !data.isEmpty {
            pending.append(data)
            while let end = pending.firstIndex(of: 10) {
                let line = String(decoding: pending[..<end], as: UTF8.self)
                pending.removeSubrange(...end)
                output(line); tail += line + "\n"
                if tail.utf8.count > 256_000 { tail = String(tail.suffix(128_000)) }
            }
            if pending.count > 1_048_576 {
                let line = String(decoding: pending, as: UTF8.self)
                output(line); tail = String(line.suffix(128_000)); pending.removeAll(keepingCapacity: true)
            }
        }
        if !pending.isEmpty { let line = String(decoding: pending, as: UTF8.self); output(line); tail += line }
        child.waitUntilExit()
        return Result(code: child.terminationStatus, output: tail)
    }
}
