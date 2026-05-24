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
                "Esegue un subcomando `swift package`. " +
                "Supportati: 'resolve' (recupera le dipendenze), 'update' (aggiorna le dipendenze), 'init' (crea un nuovo pacchetto " +
                "— richiede 'initType' e 'name'), 'describe' (riepilogo in sola lettura), 'clean'.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "command": SchemaBuilder.string(
                        description: "Subcomando.",
                        enumValues: ["resolve", "update", "init", "describe", "clean"]),
                    "packagePath": SchemaBuilder.string(description: "Directory contenente Package.swift. Default: root dell'agente."),
                    "initType": SchemaBuilder.string(
                        description: "Richiesto per 'init'. Tipo di pacchetto.",
                        enumValues: ["library", "executable", "tool", "build-tool-plugin", "command-plugin", "macro", "empty"]),
                    "name": SchemaBuilder.string(description: "Richiesto per 'init'. Nome del pacchetto."),
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
