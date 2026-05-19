import Foundation

/// `otool` — inspect a Mach-O binary's headers, load commands,
/// symbol table, or linked libraries. Schema picks one inspection
/// mode at a time so the output stays bounded.
public struct OtoolInfoTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "otool_info",
            description:
                "Inspect a Mach-O binary. 'mode' picks the section: " +
                "'header' (-h), 'loadcommands' (-l), 'libraries' (-L), 'symbols' (-Iv), 'archs' (-fh).",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "Mach-O binary path, relative to agent root."),
                    "mode": SchemaBuilder.string(
                        description: "What to dump. Default 'header'.",
                        enumValues: ["header", "loadcommands", "libraries", "symbols", "archs"]),
                ],
                required: ["path"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "otool \(input["mode"] as? String ?? "header") \(input["path"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let url = try resolveInsideRoot(try input.string("path"), context: context)
        let mode = input.optionalString("mode") ?? "header"
        let flag: String
        switch mode {
        case "header":       flag = "-h"
        case "loadcommands": flag = "-l"
        case "libraries":    flag = "-L"
        case "symbols":      flag = "-Iv"
        case "archs":        flag = "-fh"
        default: throw ToolError.invalidInput("unknown mode '\(mode)'")
        }
        return try await UnixBinary.runBinary(
            launchPath: "/usr/bin/otool",
            arguments: [flag, url.path],
            context: context,
            timeout: 60)
    }
}
