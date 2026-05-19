import Foundation

/// `swift package <subcommand>` — package-level operations (resolve
/// dependencies, update, init a new package, describe, clean).
///
/// `init` is the only sub-command that creates files outside the
/// existing repo footprint, so the input bundles the typed knobs
/// (init type / name) instead of letting the model pass raw flags.
public struct SwiftPackageTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "swift_package",
            description:
                "Run a `swift package` subcommand. " +
                "Supported: 'resolve' (fetch deps), 'update' (bump deps), 'init' (create a new package " +
                "— requires 'initType' and 'name'), 'describe' (read-only summary), 'clean'.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "command": SchemaBuilder.string(
                        description: "Subcommand.",
                        enumValues: ["resolve", "update", "init", "describe", "clean"]),
                    "packagePath": SchemaBuilder.string(description: "Directory containing Package.swift. Default agent root."),
                    "initType": SchemaBuilder.string(
                        description: "Required for 'init'. Type of package.",
                        enumValues: ["library", "executable", "tool", "build-tool-plugin", "command-plugin", "macro", "empty"]),
                    "name": SchemaBuilder.string(description: "Required for 'init'. Package name."),
                ],
                required: ["command"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "swift package \(input["command"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let command = try input.string("command")
        let pkg = try resolveInsideRoot(input.optionalString("packagePath") ?? ".",
                                        context: context)
        var args: [String] = ["package"]
        switch command {
        case "resolve":  args.append("resolve")
        case "update":   args.append("update")
        case "clean":    args.append("clean")
        case "describe":
            args.append("describe")
            args.append("--type"); args.append("json")
        case "init":
            guard let initType = input.optionalString("initType"),
                  let name = input.optionalString("name") else {
                throw ToolError.invalidInput("'init' requires 'initType' and 'name'")
            }
            args.append("init")
            args.append("--type"); args.append(initType)
            args.append("--name"); args.append(name)
        default:
            throw ToolError.invalidInput("unknown command '\(command)'")
        }
        return try await Xcrun.run(tool: "swift",
                                   arguments: args,
                                   context: context,
                                   cwd: pkg,
                                   timeout: 300)
    }
}
