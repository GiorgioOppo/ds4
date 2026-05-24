import Foundation

/// `xcodebuild -list` — schemes, targets, configurations. JSON output
/// by default so the model can parse it cleanly.
public struct XcodebuildListTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "xcodebuild_list",
            description:
                "Elenca scheme, target e configurazioni di build in un progetto Xcode o workspace. " +
                "Fornisci 'workspace' (.xcworkspace) o 'project' (.xcodeproj); se nessuno dei due, " +
                "xcodebuild rileva automaticamente nella root dell'agente. Output JSON per default.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "workspace": SchemaBuilder.string(description: "Path del .xcworkspace, relativo alla root dell'agente."),
                    "project": SchemaBuilder.string(description: "Path del .xcodeproj, relativo alla root dell'agente."),
                    "json": SchemaBuilder.boolean(description: "Emette JSON. Default true.", defaultValue: true),
                ]
            )
        )
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        var args: [String] = ["-list"]
        if input.optionalBool("json") ?? true { args.append("-json") }
        if let ws = input.optionalString("workspace") {
            let url = try resolveInsideRoot(ws, context: context)
            args.append("-workspace"); args.append(url.path)
        } else if let proj = input.optionalString("project") {
            let url = try resolveInsideRoot(proj, context: context)
            args.append("-project"); args.append(url.path)
        }
        return try await Xcrun.run(tool: "xcodebuild",
                                   arguments: args,
                                   context: context,
                                   cwd: context.rootDirectory)
    }
}
