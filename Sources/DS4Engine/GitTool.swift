import Foundation

/// The `git` tool for the agentic Code role: runs /usr/bin/git INSIDE the active
/// project root, restricted to a whitelist of local subcommands. No shell (args
/// are tokenized, not interpreted), no network subcommands (push/pull/fetch/
/// clone), no flags that could re-point git outside the project (-C, --git-dir,
/// --work-tree, --exec-path). Output is hard-capped: every byte of tool result
/// is prefill cost on a local model.
enum GitTool {
    static let allowedSubcommands: Set<String> = [
        "status", "diff", "log", "show", "branch", "blame", "grep",
        "add", "commit", "stash", "tag", "rev-parse", "ls-files",
    ]
    static let forbiddenTokens: Set<String> = [
        "-C", "--git-dir", "--work-tree", "--exec-path", "--upload-pack",
        "--receive-pack", "--config-env", "-c",
    ]
    static let maxOutput = 8_000
    static let timeoutSeconds: TimeInterval = 20

    static func run(argsLine: String) -> String {
        guard let root = ProjectCache.shared.rootURL() else {
            return "Nessun progetto importato: il tool git lavora sulla cartella del progetto attivo."
        }
        let tokens = tokenize(argsLine)
        guard let sub = tokens.first else { return "Argomento 'args' vuoto. Esempio: {\"args\":\"diff --stat\"}." }
        guard allowedSubcommands.contains(sub) else {
            return "Sottocomando git non permesso: '\(sub)'. Permessi: \(allowedSubcommands.sorted().joined(separator: ", "))."
        }
        if let bad = tokens.first(where: { forbiddenTokens.contains($0) }) {
            return "Opzione non permessa: '\(bad)'."
        }

        // Sandbox blocks ~/.gitconfig, so a commit may lack an identity: probe and
        // fall back to an agent identity WITHOUT overriding a configured repo.
        var args = tokens
        if sub == "commit", probe(["config", "user.email"], in: root).isEmpty {
            args = ["-c", "user.name=DwarfStar Agent", "-c", "user.email=agent@dwarfstar.local"] + args
        }
        let out = execute(args, in: root, timeout: timeoutSeconds)
        return out.isEmpty ? "(nessun output — comando riuscito)" : out
    }

    /// Whitespace tokenizer with double/single-quote support (no shell semantics).
    static func tokenize(_ line: String) -> [String] {
        var tokens: [String] = []
        var cur = ""
        var quote: Character? = nil
        for ch in line {
            if let q = quote {
                if ch == q { quote = nil } else { cur.append(ch) }
            } else if ch == "\"" || ch == "'" {
                quote = ch
            } else if ch == " " || ch == "\t" || ch == "\n" {
                if !cur.isEmpty { tokens.append(cur); cur = "" }
            } else {
                cur.append(ch)
            }
        }
        if !cur.isEmpty { tokens.append(cur) }
        return tokens
    }

    private static func probe(_ args: [String], in root: URL) -> String {
        execute(args, in: root, timeout: 5).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Run git with merged stdout/stderr, a timeout (SIGTERM), and a size cap.
    private static func execute(_ args: [String], in root: URL, timeout: TimeInterval) -> String {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        proc.arguments = args
        proc.currentDirectoryURL = root
        var env = ProcessInfo.processInfo.environment
        env["GIT_TERMINAL_PROMPT"] = "0"        // never block waiting for input
        env["GIT_PAGER"] = "cat"
        proc.environment = env
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = pipe
        do { try proc.run() } catch {
            return "git non eseguibile: \(error.localizedDescription) (servono i Command Line Tools)."
        }
        // Read concurrently (a full pipe would deadlock waitUntilExit). Only the
        // reader thread mutates `data`; the main thread reads it after the process
        // has exited (and a short drain), so the access is serialized in practice.
        nonisolated(unsafe) var data = Data()
        let reader = Thread {
            while let chunk = try? pipe.fileHandleForReading.read(upToCount: 64 * 1024), !chunk.isEmpty {
                data.append(chunk)
                if data.count > maxOutput * 4 { proc.terminate() }
            }
        }
        reader.start()
        let deadline = Date().addingTimeInterval(timeout)
        while proc.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.05)
        }
        if proc.isRunning {
            proc.terminate()
            return "git interrotto: timeout di \(Int(timeout))s."
        }
        // Drain whatever the reader has collected.
        Thread.sleep(forTimeInterval: 0.05)
        var text = String(data: data, encoding: .utf8) ?? ""
        if proc.terminationStatus != 0 {
            text = "exit \(proc.terminationStatus)\n" + text
        }
        if text.count > maxOutput {
            text = String(text.prefix(maxOutput)) + "\n… (output troncato a \(maxOutput) caratteri)"
        }
        return text
    }
}
