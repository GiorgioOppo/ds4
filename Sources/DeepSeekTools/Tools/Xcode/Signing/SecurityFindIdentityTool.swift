import Foundation

/// `security find-identity` — list code-signing identities in the
/// keychain. The model uses this to pick a `--sign` argument for
/// xcodebuild or to confirm a profile / certificate is present.
public struct SecurityFindIdentityTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "security_find_identity",
            description:
                "List code-signing identities (Developer ID, Apple Distribution, etc.) in the keychain. " +
                "Default scope: only valid identities. Set 'validOnly=false' to include expired ones.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "validOnly": SchemaBuilder.boolean(description: "Only show valid identities. Default true.", defaultValue: true),
                    "keychain": SchemaBuilder.string(description: "Specific keychain to query (default: search list)."),
                ]
            )
        )
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        var args: [String] = ["find-identity", "-p", "codesigning"]
        if input.optionalBool("validOnly") ?? true { args.append("-v") }
        if let kc = input.optionalString("keychain") { args.append(kc) }
        return try await UnixBinary.runBinary(
            launchPath: "/usr/bin/security",
            arguments: args,
            context: context,
            timeout: 30)
    }
}
