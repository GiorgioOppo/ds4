import Foundation
import CryptoKit

/// SHA-256 hex digest of a file. CryptoKit-backed.
public struct Sha256Tool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "sha256",
            description:
                "SHA-256 hex digest of a file. Cryptographically secure for integrity checking.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "File path, relative to agent root."),
                ],
                required: ["path"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "sha256 \(input["path"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let rel = try input.string("path")
        let url = try resolveInsideRoot(rel, context: context)
        guard let data = try? Data(contentsOf: url) else {
            throw ToolError.notFound("cannot read '\(rel)'")
        }
        let digest = SHA256.hash(data: data)
        let hex = digest.map { String(format: "%02x", $0) }.joined()
        return ToolOutput(output: "\(hex)  \(rel)")
    }
}
