import Foundation

/// Extract a .zip archive into a destination directory.
public struct UnzipTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "unzip",
            description:
                "Estrae un archivio .zip. Sia l'archivio sia la destinazione devono trovarsi dentro la root dell'agente. " +
                "Imposta operation='list' per elencare solo il contenuto senza estrarlo.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "archive": SchemaBuilder.string(description: "Path del .zip, relativo alla root dell'agente."),
                    "destination": SchemaBuilder.string(description: "Directory di destinazione, relativa alla root dell'agente. Default = directory padre dell'archivio."),
                    "operation": SchemaBuilder.string(
                        description: "Cosa fare.",
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
