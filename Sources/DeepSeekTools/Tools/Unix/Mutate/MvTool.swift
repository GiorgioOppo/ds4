import Foundation

/// Move/rename. Both endpoints inside the agent root. On the same
/// volume `FileManager.moveItem` is atomic; across volumes it falls
/// back to copy+delete and an interruption may leave both copies on
/// disk — we accept this trade-off (same reasoning as `CpTool`).
public struct MvTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "mv",
            description:
                "Move or rename a file/directory inside the agent root. " +
                "Refuses to overwrite an existing destination unless 'force=true'.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "src": SchemaBuilder.string(description: "Source path, relative to agent root."),
                    "dst": SchemaBuilder.string(description: "Destination path, relative to agent root."),
                    "force": SchemaBuilder.boolean(description: "Overwrite an existing destination. Default false.", defaultValue: false),
                ],
                required: ["src", "dst"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "mv \(input["src"] as? String ?? "?") -> \(input["dst"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let srcRel = try input.string("src")
        let dstRel = try input.string("dst")
        let force = input.optionalBool("force") ?? false
        let src = try resolveInsideRoot(srcRel, context: context)
        let dst = try resolveInsideRoot(dstRel, context: context)
        let fm = FileManager.default

        guard fm.fileExists(atPath: src.path) else {
            throw ToolError.notFound("source missing: \(srcRel)")
        }
        if fm.fileExists(atPath: dst.path) {
            if !force {
                throw ToolError.invalidInput("destination exists: \(dstRel); set force=true to overwrite")
            }
            try fm.removeItem(at: dst)
        }
        try fm.moveItem(at: src, to: dst)
        return ToolOutput(output: "moved \(srcRel) -> \(dstRel)")
    }
}
