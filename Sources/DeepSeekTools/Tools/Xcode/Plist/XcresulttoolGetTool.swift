import Foundation

/// `xcrun xcresulttool get` — extract structured data from a
/// `.xcresult` bundle produced by `xcodebuild test`.
///
/// Note: in Xcode 16+ the legacy `get`/`get object` invocations are
/// deprecated in favor of `get test-results`. We pick the right
/// sub-form via the `kind` argument.
public struct XcresulttoolGetTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "xcresulttool_get",
            description:
                "Estrae dati da un bundle .xcresult. " +
                "'kind' seleziona: 'summary' (riepilogo dei test-results), 'tests' (JSON completo dei test-results), " +
                "'object' (object graph raw — stile legacy Xcode 15, richiede 'id'), " +
                "'log' (build log).",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "xcresultPath": SchemaBuilder.string(description: "Bundle .xcresult, relativo alla root dell'agente."),
                    "kind": SchemaBuilder.string(
                        description: "Cosa estrarre. Default 'summary'.",
                        enumValues: ["summary", "tests", "object", "log"]),
                    "id": SchemaBuilder.string(description: "Id dell'oggetto (per kind='object')."),
                ],
                required: ["xcresultPath"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "xcresulttool get \(input["xcresultPath"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let path = try resolveInsideRoot(try input.string("xcresultPath"), context: context)
        let kind = input.optionalString("kind") ?? "summary"
        var args: [String] = ["get"]
        switch kind {
        case "summary":
            args += ["test-results", "summary", "--path", path.path]
        case "tests":
            args += ["test-results", "tests", "--path", path.path]
        case "object":
            guard let id = input.optionalString("id") else {
                throw ToolError.invalidInput("kind='object' requires 'id'")
            }
            args += ["object", "--legacy", "--format", "json", "--path", path.path, "--id", id]
        case "log":
            args += ["log", "--path", path.path]
        default:
            throw ToolError.invalidInput("unknown kind '\(kind)'")
        }
        return try await Xcrun.run(tool: "xcresulttool",
                                   arguments: args,
                                   context: context,
                                   cwd: context.rootDirectory,
                                   timeout: 60)
    }
}
