import Foundation

/// Create a .zip archive from one or more inputs. Recursive by default.
public struct ZipTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "zip",
            description:
                "Create a .zip archive from one or more inputs. Recursive (-r) by default. " +
                "All inputs and the archive destination must be inside the agent root.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "archive": SchemaBuilder.string(description: "Output .zip path, relative to agent root."),
                    "inputs": SchemaBuilder.array(itemsType: "string", description: "Files/dirs to include, relative to agent root."),
                    "recursive": SchemaBuilder.boolean(description: "Recurse into directories. Default true.", defaultValue: true),
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
