import Foundation
import CryptoKit

/// Compute the MD5 digest of a file. CryptoKit-backed.
///
/// Despite MD5 being cryptographically broken, it remains the
/// de-facto checksum format for many ecosystems (Homebrew bottle
/// integrity, legacy package indexes, etc.). Surface it but don't
/// claim collision resistance.
public struct Md5Tool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "md5",
            description:
                "MD5 hex digest of a file. NOT cryptographically secure — use for checksums only.",
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
        "md5 \(input["path"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let rel = try input.string("path")
        let url = try resolveInsideRoot(rel, context: context)
        guard let data = try? Data(contentsOf: url) else {
            throw ToolError.notFound("cannot read '\(rel)'")
        }
        let digest = Insecure.MD5.hash(data: data)
        let hex = digest.map { String(format: "%02x", $0) }.joined()
        return ToolOutput(output: "\(hex)  \(rel)")
    }
}
