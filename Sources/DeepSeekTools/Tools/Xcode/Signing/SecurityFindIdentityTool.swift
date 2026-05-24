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
                "Elenca le identità di firma del codice (Developer ID, Apple Distribution, ecc.) nel keychain. " +
                "Scope di default: solo identità valide. Imposta 'validOnly=false' per includere quelle scadute.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "validOnly": SchemaBuilder.boolean(description: "Mostra solo identità valide. Default true.", defaultValue: true),
                    "keychain": SchemaBuilder.string(description: "Keychain specifico da interrogare (default: la search list)."),
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
