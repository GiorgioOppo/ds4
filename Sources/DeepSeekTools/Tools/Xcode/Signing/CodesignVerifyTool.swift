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
                "Verify the code signature of a binary or .app bundle. " +
                "Returns exit 0 + 'satisfies its Designated Requirement' on success; " +
                "non-zero exit + details on failure.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "Binary or .app path, relative to agent root."),
                    "strict": SchemaBuilder.boolean(description: "--strict flag. Default false.", defaultValue: false),
                    "deep": SchemaBuilder.boolean(description: "--deep flag (recurse). Default false.", defaultValue: false),
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
