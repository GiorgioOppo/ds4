import Foundation
import CryptoKit

/// SHA-1 hex digest of a file. CryptoKit-backed. SHA-1 is insecure
/// for collision resistance; useful only for legacy interop.
public struct Sha1Tool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "sha1",
            description:
                "SHA-1 hex digest of a file. Legacy — NOT collision-resistant. Prefer sha256 for new code.",
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
        "sha1 \(input["path"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let rel = try input.string("path")
        let url = try resolveInsideRoot(rel, context: context)
        guard let data = try? Data(contentsOf: url) else {
            throw ToolError.notFound("cannot read '\(rel)'")
        }
        let digest = Insecure.SHA1.hash(data: data)
        let hex = digest.map { String(format: "%02x", $0) }.joined()
        return ToolOutput(output: "\(hex)  \(rel)")
    }
}
