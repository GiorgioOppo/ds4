import Foundation

/// Run a shell command and stream its combined stdout/stderr back to
/// the model. The `.dangerous` category means every call hits the
/// permission delegate unless the user has marked it as alwaysAllow
/// for the session, and `.plan` mode prompts on every call even
/// when alwaysAllow is set (enforced at the registry level — we
/// don't see plan-mode shell calls here).
///
/// `useSandbox` triggers a `sandbox-exec` wrapper using the profile
/// at `sandbox/default.sb` (if present) — opt-in because it's
/// macOS-only and the profile takes tuning. Without it the command
/// runs in the host's shell with the agent's environment.
public struct ShellTool: Tool {
    public let useSandbox: Bool
    public let shellPath: String
    public let timeoutSeconds: TimeInterval

    public init(useSandbox: Bool = false,
                shellPath: String = "/bin/zsh",
                timeoutSeconds: TimeInterval = 120) {
        self.useSandbox = useSandbox
        self.shellPath = shellPath
        self.timeoutSeconds = timeoutSeconds
    }

    public var schema: ToolSchema {
        ToolSchema(
            name: "shell",
            description:
                "Esegue un comando shell nella working directory dell'agente. " +
                "Restituisce stdout+stderr combinati (troncati a 32 KB) e il " +
                "codice di uscita. Preferisci sempre i tool dedicati (read/edit/grep) " +
                "quando uno è adatto — shell è per tutto il resto.",
            category: .dangerous,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "command": SchemaBuilder.string(description: "Il comando shell da eseguire."),
                    "cwd": SchemaBuilder.string(description: "Working directory, relativa alla root dell'agente. Default: root."),
                    "timeout": SchemaBuilder.integer(description: "Timeout per chiamata in secondi. Default 120.", minimum: 1),
                ],
                required: ["command"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        let cmd = (input["command"] as? String ?? "?")
        let preview = cmd.count > 60 ? String(cmd.prefix(60)) + "…" : cmd
        return "shell: \(preview)"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let command = try input.string("command")
        let cwdOverride = input.optionalString("cwd")
        let cwd: URL
        if let cwdOverride {
            cwd = try resolveInsideRoot(cwdOverride, context: context)
        } else {
            cwd = context.rootDirectory
        }
        let timeout = TimeInterval(input.optionalInteger("timeout") ?? Int(timeoutSeconds))

        let process = Process()
        process.currentDirectoryURL = cwd
        if let env = context.environment { process.environment = env }

        if useSandbox {
            // sandbox-exec is deprecated-but-present on macOS 14. The
            // profile file is expected to live at <root>/sandbox/default.sb;
            // if it's missing we fall through to a plain shell run.
            let profile = context.rootDirectory
                .appendingPathComponent("sandbox/default.sb")
            if FileManager.default.fileExists(atPath: profile.path) {
                process.launchPath = "/usr/bin/sandbox-exec"
                process.arguments = ["-f", profile.path, shellPath, "-c", command]
            } else {
                process.launchPath = shellPath
                process.arguments = ["-c", command]
            }
        } else {
            process.launchPath = shellPath
            process.arguments = ["-c", command]
        }

        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr

        do {
            try process.run()
        } catch {
            throw ToolError.spawnFailed(error.localizedDescription)
        }

        // Watchdog. Foundation's Process has no native timeout — we
        // launch a Task that checks elapsed wall-time and the
        // context's cancel hook, then terminates the child on either.
        let deadline = Date().addingTimeInterval(timeout)
        let watcher = Task {
            while process.isRunning {
                if Date() > deadline || context.isCancelled() {
                    process.terminate()
                    return
                }
                try? await Task.sleep(nanoseconds: 200_000_000)
            }
        }
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            process.terminationHandler = { _ in cont.resume() }
        }
        watcher.cancel()

        let outData = stdout.fileHandleForReading.readDataToEndOfFile()
        let errData = stderr.fileHandleForReading.readDataToEndOfFile()
        var combined = (String(data: outData, encoding: .utf8) ?? "")
        let errString = String(data: errData, encoding: .utf8) ?? ""
        if !errString.isEmpty {
            if !combined.isEmpty && !combined.hasSuffix("\n") { combined += "\n" }
            combined += errString
        }
        // 32 KB cap so a runaway log doesn't blow the context window.
        let capped: String
        let cap = 32 * 1024
        if combined.utf8.count > cap {
            let prefix = String(combined.prefix(cap))
            capped = prefix + "\n[output truncated at \(cap) bytes]"
        } else {
            capped = combined
        }
        let status = process.terminationStatus
        return ToolOutput(
            output: "exit \(status)\n\(capped)",
            isError: status != 0,
            metadata: ["exit": "\(status)"]
        )
    }
}
