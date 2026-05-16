import Foundation

/// Shallow-clone a git repository into the agent's working directory.
/// Implemented by shelling out to `git` (so it inherits the user's
/// credentials, SSH keys, gitconfig). Marked `.dangerous` because
/// arbitrary code can ship inside hooks — the user should approve.
public struct RepoCloneTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "repo_clone",
            description:
                "Shallow git clone (depth=1) of a remote into a subdirectory " +
                "of the agent root. Uses the user's local git credentials.",
            category: .dangerous,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "url": SchemaBuilder.string(description: "Git remote URL (https or ssh)."),
                    "destination": SchemaBuilder.string(description: "Subdirectory name. Default: repo basename."),
                    "depth": SchemaBuilder.integer(description: "Clone depth. Default 1.", minimum: 1),
                ],
                required: ["url"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "repo_clone \(input["url"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let url = try input.string("url")
        let depth = input.optionalInteger("depth") ?? 1
        let dest: String
        if let explicit = input.optionalString("destination") {
            dest = explicit
        } else {
            dest = URL(string: url)?.deletingPathExtension().lastPathComponent
                ?? "cloned-repo"
        }
        let destURL = try resolveInsideRoot(dest, context: context)
        if FileManager.default.fileExists(atPath: destURL.path) {
            throw ToolError.invalidInput("destination '\(dest)' already exists")
        }
        let process = Process()
        process.launchPath = "/usr/bin/env"
        process.arguments = ["git", "clone", "--depth", "\(depth)", url, destURL.path]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        do { try process.run() } catch {
            throw ToolError.spawnFailed(error.localizedDescription)
        }
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            process.terminationHandler = { _ in cont.resume() }
        }
        let log = String(data: pipe.fileHandleForReading.readDataToEndOfFile(),
                         encoding: .utf8) ?? ""
        if process.terminationStatus != 0 {
            throw ToolError.external("git exit \(process.terminationStatus): \(log)")
        }
        return ToolOutput(
            output: "cloned into \(dest)\n\(log)",
            metadata: ["path": destURL.path]
        )
    }
}
