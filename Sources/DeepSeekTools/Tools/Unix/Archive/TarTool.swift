import Foundation

/// Tar wrapper. Supports the two operations the model usually wants:
/// list contents (read-only operation but the tool itself is
/// classified `.mutating` because the extract path needs the same
/// consent flow) and extract to a directory inside the agent root.
/// Archive *creation* is intentionally not in this v1 schema — that
/// can come back as a follow-up if needed.
public struct TarTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "tar",
            description:
                "Operate on a tar archive. operation='list' prints the contents; " +
                "operation='extract' extracts into 'destination' (must be inside the agent root). " +
                "Auto-detects gzip/bzip2/xz compression. Archive creation is not supported in this tool.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "operation": SchemaBuilder.string(
                        description: "What to do.",
                        enumValues: ["list", "extract"]),
                    "archive": SchemaBuilder.string(description: "Archive path, relative to agent root."),
                    "destination": SchemaBuilder.string(description: "Extract destination dir, relative to agent root. Required for extract."),
                ],
                required: ["operation", "archive"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "tar \(input["operation"] as? String ?? "?") \(input["archive"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let op = try input.string("operation")
        let archiveRel = try input.string("archive")
        let archive = try resolveInsideRoot(archiveRel, context: context)

        switch op {
        case "list":
            return try await UnixBinary.runBinary(
                launchPath: "/usr/bin/tar",
                arguments: ["-tf", archive.path],
                context: context)
        case "extract":
            guard let destRel = input.optionalString("destination") else {
                throw ToolError.invalidInput("'destination' required for extract")
            }
            let dest = try resolveInsideRoot(destRel, context: context)
            try FileManager.default.createDirectory(at: dest, withIntermediateDirectories: true)
            return try await UnixBinary.runBinary(
                launchPath: "/usr/bin/tar",
                arguments: ["-xf", archive.path, "-C", dest.path],
                context: context)
        default:
            throw ToolError.invalidInput("unknown operation '\(op)'")
        }
    }
}
