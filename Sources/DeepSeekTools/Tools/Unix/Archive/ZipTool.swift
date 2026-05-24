import Foundation

/// Create a .zip archive from one or more inputs. Recursive by default.
public struct ZipTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "zip",
            description:
                "Crea un archivio .zip da uno o più input. Ricorsivo (-r) per default. " +
                "Tutti gli input e la destinazione dell'archivio devono trovarsi dentro la root dell'agente.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "archive": SchemaBuilder.string(description: "Path del .zip di output, relativo alla root dell'agente."),
                    "inputs": SchemaBuilder.array(itemsType: "string", description: "File o directory da includere, relativi alla root dell'agente."),
                    "recursive": SchemaBuilder.boolean(description: "Discende nelle directory. Default true.", defaultValue: true),
                ],
                required: ["archive", "inputs"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "zip \(input["archive"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let archiveRel = try input.string("archive")
        let inputs = input.optionalStringArray("inputs") ?? []
        let recursive = input.optionalBool("recursive") ?? true
        guard !inputs.isEmpty else {
            throw ToolError.invalidInput("'inputs' must be non-empty")
        }
        let archive = try resolveInsideRoot(archiveRel, context: context)
        var args: [String] = []
        if recursive { args.append("-r") }
        args.append(archive.path)
        for rel in inputs {
            let url = try resolveInsideRoot(rel, context: context)
            args.append(url.path)
        }
        return try await UnixBinary.runBinary(
            launchPath: "/usr/bin/zip",
            arguments: args,
            context: context)
    }
}
