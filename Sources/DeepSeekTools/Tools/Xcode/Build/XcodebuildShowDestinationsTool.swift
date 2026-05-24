import Foundation

/// `xcodebuild -showdestinations -scheme X` — list the destinations a
/// scheme can target. The model usually needs this before composing a
/// `-destination` argument for build/test/archive.
public struct XcodebuildShowDestinationsTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "xcodebuild_showdestinations",
            description:
                "Elenca le destinazioni disponibili per uno scheme (simulator, dispositivi reali, my Mac, Mac Catalyst, …). " +
                "Fornisci 'scheme' e 'workspace' o 'project'.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "scheme": SchemaBuilder.string(description: "Nome dello scheme."),
                    "workspace": SchemaBuilder.string(description: ".xcworkspace, relativo alla root dell'agente."),
                    "project": SchemaBuilder.string(description: ".xcodeproj, relativo alla root dell'agente."),
                ],
                required: ["scheme"]
            )
        )
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let scheme = try input.string("scheme")
        var args: [String] = ["-showdestinations", "-scheme", scheme]
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
