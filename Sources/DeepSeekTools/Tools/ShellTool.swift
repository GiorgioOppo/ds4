import Foundation

/// Run a shell command and stream its combined stdout/stderr back to
/// the model. The `.dangerous` category means every call hits the
/// permission delegate unless the user has marked it as alwaysAllow
/// for the session, and `.plan` mode prompts on every call even
/// when alwaysAllow is set (enforced at the registry level — we
/// don't see plan-mode shell calls here).
///
/// `useSandbox` triggers a `sandbox-exec` wrapper. By default it
/// reads the profile at `sandbox/default.sb` under the agent root —
/// opt-in because it's macOS-only and the profile takes tuning.
///
/// `profileBuilder`, when non-nil, lets the host supply a profile
/// string dynamically per call. This is the path the project-bound
/// chat uses: it routes `context.additionalReadRoots` (the user's
/// real source folders behind the symlink farm) through the closure
/// so the rendered profile authorises reads under each of them.
/// Without that, the seatbelt deny-by-default would block every
/// read through a farm symlink because the resolved target lives
/// outside `DEEPSEEK_ROOT`. See `DeepSeekIntegrations.Sandbox.
/// renderProfile(extraReadRoots:)` for the canonical builder.
public struct ShellTool: Tool {
    public let useSandbox: Bool
    public let shellPath: String
    public let timeoutSeconds: TimeInterval
    public let profileBuilder: (@Sendable ([URL]) -> String)?

    public init(useSandbox: Bool = false,
                shellPath: String = "/bin/zsh",
                timeoutSeconds: TimeInterval = 120,
                profileBuilder: (@Sendable ([URL]) -> String)? = nil) {
        self.useSandbox = useSandbox
        self.shellPath = shellPath
        self.timeoutSeconds = timeoutSeconds
        self.profileBuilder = profileBuilder
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

        // Track any per-call temp profile so we delete it after the
        // child exits. nil when we're either not sandboxing or we
        // fell back to the on-disk default profile.
        var tempProfile: URL? = nil

        if useSandbox {
            // sandbox-exec is deprecated-but-present on macOS 14.
            // Preferred path: render a per-call profile via the host-
            // supplied closure so the `additionalReadRoots` (the
            // user's real source folders behind the symlink farm)
            // are baked into the seatbelt rules — otherwise every
            // read through a farm symlink fails because the resolved
            // target lives outside `DEEPSEEK_ROOT`.
            // Fallback path: on-disk profile at <root>/sandbox/default.sb
            // when no builder was provided (CLI / tests).
            let profilePath: String?
            if let builder = profileBuilder {
                let rendered = builder(context.additionalReadRoots)
                do {
                    let tmp = FileManager.default.temporaryDirectory
                        .appendingPathComponent(
                            "deepseek-sb-\(UUID().uuidString).sb")
                    try Data(rendered.utf8).write(to: tmp, options: .atomic)
                    tempProfile = tmp
                    profilePath = tmp.path
                } catch {
                    // Couldn't write a temp profile — fall back to the
                    // unsandboxed shell rather than silently dropping
                    // the user's intent on the floor.
                    profilePath = nil
                }
            } else {
                let onDisk = context.rootDirectory
                    .appendingPathComponent("sandbox/default.sb")
                profilePath = FileManager.default.fileExists(atPath: onDisk.path)
                    ? onDisk.path : nil
            }
            if let profilePath {
                process.launchPath = "/usr/bin/sandbox-exec"
                // `-D KEY=VAL` populates `(param "KEY")` in the
                // profile. We bind DEEPSEEK_ROOT to the cwd so the
                // subpath rules anchor to the farm.
                let rootArg = "DEEPSEEK_ROOT=" + context.rootDirectory.path
                process.arguments = [
                    "-D", rootArg,
                    "-f", profilePath,
                    shellPath, "-c", command,
                ]
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

        if let tempProfile {
            try? FileManager.default.removeItem(at: tempProfile)
        }
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
