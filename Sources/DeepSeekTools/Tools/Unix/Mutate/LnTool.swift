import Foundation

/// Create a symbolic link. Both the link path AND the target must
/// resolve inside the trust boundary (the agent root or any
/// `additionalReadRoots` entry) — otherwise a link inside the root
/// would silently widen the sandbox for any subsequent reads through
/// it. Hard links are NOT supported (would also need cross-fs check).
public struct LnTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "ln",
            description:
                "Crea un link simbolico nello spazio di lavoro dell'agente. Sia il path del link sia il target " +
                "devono risolvere dentro la root o le source aggiuntive del progetto — un link verso l'esterno viene rifiutato. " +
                "Gli hard link non sono supportati.",
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
        // Both calls validate the path stays inside the root or any
        // `additionalReadRoots` entry; `linkPath` additionally enforces
        // the resolved-target check so a sneaky symlink in the agent
        // root can't make us materialise the new link outside the
        // trust boundary. The `target` string is stored verbatim
        // inside the link payload (no open() at creation time), so the
        // default boundary check covers it.
        let target = try resolveInsideRoot(targetRel, context: context)
        let linkPath = try resolveInsideRoot(linkRel, context: context,
                                              checkResolvedTarget: true)
        let fm = FileManager.default
        if fm.fileExists(atPath: linkPath.path) {
            throw ToolError.invalidInput("link path already exists: \(linkRel)")
        }
        try fm.createSymbolicLink(at: linkPath, withDestinationURL: target)
        return ToolOutput(output: "linked \(linkRel) -> \(targetRel)")
    }
}
