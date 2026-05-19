import Foundation

/// Create a symbolic link. Both the link path AND the target must
/// resolve inside the agent root — otherwise a link inside the root
/// would silently widen the sandbox for any subsequent reads through
/// it. Hard links are NOT supported (would also need cross-fs check).
public struct LnTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "ln",
            description:
                "Create a symbolic link inside the agent root. Both the link path and the target " +
                "must resolve inside the root — a link to outside is rejected. Hard links are not supported.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "target": SchemaBuilder.string(description: "Target path (what the link points to), relative to agent root."),
                    "linkPath": SchemaBuilder.string(description: "Path where the link is created, relative to agent root."),
                ],
                required: ["target", "linkPath"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "ln \(input["target"] as? String ?? "?") -> \(input["linkPath"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let targetRel = try input.string("target")
        let linkRel = try input.string("linkPath")
        // Both calls validate the path stays inside the root.
        let target = try resolveInsideRoot(targetRel, context: context)
        let linkPath = try resolveInsideRoot(linkRel, context: context)
        let fm = FileManager.default
        if fm.fileExists(atPath: linkPath.path) {
            throw ToolError.invalidInput("link path already exists: \(linkRel)")
        }
        try fm.createSymbolicLink(at: linkPath, withDestinationURL: target)
        return ToolOutput(output: "linked \(linkRel) -> \(targetRel)")
    }
}
