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
                "Sposta o rinomina un file/directory dentro la root dell'agente. " +
                "Si rifiuta di sovrascrivere una destinazione esistente a meno che 'force=true'.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "src": SchemaBuilder.string(description: "Path di origine, relativo alla root dell'agente."),
                    "dst": SchemaBuilder.string(description: "Path di destinazione, relativo alla root dell'agente."),
                    "force": SchemaBuilder.boolean(description: "Sovrascrive una destinazione esistente. Default false.", defaultValue: false),
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
        // Dst side gates the write: a sneaky symlink at dstRel would
        // land the moved entry outside the trust boundary. Src side
        // operates on the link itself (moveItem renames the symlink),
        // so the default check is enough there.
        let dst = try resolveInsideRoot(dstRel, context: context,
                                         checkResolvedTarget: true)
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
