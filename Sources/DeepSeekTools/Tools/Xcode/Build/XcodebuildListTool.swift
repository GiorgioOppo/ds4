import Foundation

/// `xcodebuild -list` — schemes, targets, configurations. JSON output
/// by default so the model can parse it cleanly.
public struct XcodebuildListTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "xcodebuild_list",
            description:
                "List schemes, targets, and build configurations in an Xcode project or workspace. " +
                "Provide either 'workspace' (.xcworkspace) or 'project' (.xcodeproj); if neither, " +
                "xcodebuild auto-detects in the agent root. JSON output by default.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "workspace": SchemaBuilder.string(description: "Path to .xcworkspace, relative to agent root."),
                    "project": SchemaBuilder.string(description: "Path to .xcodeproj, relative to agent root."),
                    "json": SchemaBuilder.boolean(description: "Emit JSON. Default true.", defaultValue: true),
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
