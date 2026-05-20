import Foundation

/// Extract a .zip archive into a destination directory.
public struct UnzipTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "unzip",
            description:
                "Extract a .zip archive. Both the archive and the destination must be inside the agent root. " +
                "Set operation='list' to just list the contents without extracting.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "archive": SchemaBuilder.string(description: ".zip path, relative to agent root."),
                    "destination": SchemaBuilder.string(description: "Destination dir, relative to agent root. Default = archive parent."),
                    "operation": SchemaBuilder.string(
                        description: "What to do.",
                        enumValues: ["extract", "list"]),
                ],
                required: ["archive"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        let op = input["operation"] as? String ?? "extract"
        return "unzip \(op) \(input["archive"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let op = input.optionalString("operation") ?? "extract"
        let archiveRel = try input.string("archive")
        let archive = try resolveInsideRoot(archiveRel, context: context)
        if op == "list" {
            return try await UnixBinary.runBinary(
                launchPath: "/usr/bin/unzip",
                arguments: ["-l", archive.path],
                context: context)
        }
        let dest: URL
        if let destRel = input.optionalString("destination") {
            dest = try resolveInsideRoot(destRel, context: context)
        } else {
            dest = archive.deletingLastPathComponent()
        }
        try FileManager.default.createDirectory(at: dest, withIntermediateDirectories: true)
        return try await UnixBinary.runBinary(
            launchPath: "/usr/bin/unzip",
            arguments: ["-o", archive.path, "-d", dest.path],
            context: context)
    }
}
