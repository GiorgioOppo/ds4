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
                "Crea un link simbolico dentro la root dell'agente. Sia il path del link sia il target " +
                "devono risolvere dentro la root — un link verso l'esterno viene rifiutato. Gli hard link non sono supportati.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "target": SchemaBuilder.string(description: "Path di destinazione (a cosa punta il link), relativo alla root dell'agente."),
                    "linkPath": SchemaBuilder.string(description: "Path in cui viene creato il link, relativo alla root dell'agente."),
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
