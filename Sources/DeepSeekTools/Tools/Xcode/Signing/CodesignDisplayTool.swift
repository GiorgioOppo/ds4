import Foundation

/// `codesign -d` — display signing info, optionally entitlements
/// and designated requirements.
public struct CodesignDisplayTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "codesign_display",
            description:
                "Display the signing identity, team, entitlements, and designated requirement of a " +
                "signed binary or bundle.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "Binary or .app path, relative to agent root."),
                    "entitlements": SchemaBuilder.boolean(description: "Include entitlements. Default true.", defaultValue: true),
                    "requirements": SchemaBuilder.boolean(description: "Include designated requirement. Default true.", defaultValue: true),
                ],
                required: ["path"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "codesign -d \(input["path"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let url = try resolveInsideRoot(try input.string("path"), context: context)
        var args: [String] = ["-d", "--verbose=4"]
        if input.optionalBool("entitlements") ?? true {
            args.append("--entitlements"); args.append(":-")  // print to stdout
        }
        if input.optionalBool("requirements") ?? true {
            args.append("-r-")
        }
        args.append(url.path)
        return try await UnixBinary.runBinary(
            launchPath: "/usr/bin/codesign",
            arguments: args,
            context: context,
            timeout: 60,
            separateStreams: true)
    }
}
