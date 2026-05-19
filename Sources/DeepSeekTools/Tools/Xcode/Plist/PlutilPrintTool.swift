import Foundation

/// `plutil -p / -convert` — read and pretty-print a plist file.
/// JSON or XML output for machine parsing.
public struct PlutilPrintTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "plutil_print",
            description:
                "Print a plist file in a parseable format. " +
                "'format' picks 'human' (default, plutil -p), 'json', or 'xml1'. " +
                "Output is written to stdout (the file on disk is not converted).",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "Plist file, relative to agent root."),
                    "format": SchemaBuilder.string(
                        description: "Output format.",
                        enumValues: ["human", "json", "xml1"]),
                ],
                required: ["path"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "plutil -p \(input["path"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let url = try resolveInsideRoot(try input.string("path"), context: context)
        let fmt = input.optionalString("format") ?? "human"
        let args: [String]
        switch fmt {
        case "json":  args = ["-convert", "json", "-o", "-", url.path]
        case "xml1":  args = ["-convert", "xml1", "-o", "-", url.path]
        default:      args = ["-p", url.path]
        }
        return try await UnixBinary.runBinary(
            launchPath: "/usr/bin/plutil",
            arguments: args,
            context: context,
            timeout: 30)
    }
}
