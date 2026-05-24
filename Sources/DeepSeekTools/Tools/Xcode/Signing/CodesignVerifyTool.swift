import Foundation

/// `codesign --verify` — check that a binary or bundle has a valid
/// signature. `strict=true` adds strict checking (rejects bundles
/// with extra unsigned resources); `deep=true` recurses into nested
/// bundles.
public struct CodesignVerifyTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "codesign_verify",
            description:
                "Verifica la firma di codice di un binario o bundle .app. " +
                "Restituisce exit 0 + 'satisfies its Designated Requirement' in caso di successo; " +
                "exit non zero + dettagli in caso di fallimento.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "Path del binario o .app, relativo alla root dell'agente."),
                    "strict": SchemaBuilder.boolean(description: "Flag --strict. Default false.", defaultValue: false),
                    "deep": SchemaBuilder.boolean(description: "Flag --deep (ricorsivo). Default false.", defaultValue: false),
                ],
                required: ["path"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "codesign --verify \(input["path"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let url = try resolveInsideRoot(try input.string("path"), context: context)
        var args: [String] = ["--verify", "--verbose=2"]
        if input.optionalBool("strict") ?? false { args.append("--strict") }
        if input.optionalBool("deep") ?? false { args.append("--deep") }
        args.append(url.path)
        return try await UnixBinary.runBinary(
            launchPath: "/usr/bin/codesign",
            arguments: args,
            context: context,
            timeout: 60,
            separateStreams: true)
    }
}
