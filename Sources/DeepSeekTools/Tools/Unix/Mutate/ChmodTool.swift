import Foundation

/// Change POSIX permissions on a file or directory. Accepts an octal
/// triplet (e.g. "755", "0644"). Symbolic specs ("u+x") are NOT
/// supported — pre-compute the octal client-side.
public struct ChmodTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "chmod",
            description:
                "Imposta i permessi POSIX su un file o directory dentro la root dell'agente. " +
                "'mode' deve essere in cifre ottali (es. '755', '0644'). Le forme simboliche come 'u+x' non sono supportate.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "Path, relativo alla root dell'agente."),
                    "mode": SchemaBuilder.string(description: "Tripletta di permessi in ottale (3 o 4 cifre)."),
                ],
                required: ["path", "mode"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "chmod \(input["mode"] as? String ?? "?") \(input["path"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let rel = try input.string("path")
        let modeStr = try input.string("mode")
        guard let mode = Int(modeStr, radix: 8), mode >= 0, mode <= 0o7777 else {
            throw ToolError.invalidInput("'mode' must be octal (e.g. '755'), got '\(modeStr)'")
        }
        let url = try resolveInsideRoot(rel, context: context)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ToolError.notFound("'\(rel)' does not exist")
        }
        try FileManager.default.setAttributes(
            [.posixPermissions: NSNumber(value: mode)],
            ofItemAtPath: url.path)
        return ToolOutput(output: "chmod \(String(format: "%04o", mode)) \(rel)")
    }
}
