import Foundation

/// `agvtool` — read or bump the project's build number
/// (CFBundleVersion) and marketing version (CFBundleShortVersionString).
/// Must be run from the directory containing the `.xcodeproj`.
public struct AgvtoolVersionTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "agvtool_version",
            description:
                "Read or update an Xcode project's version. " +
                "operation='read-build' / 'read-marketing' / 'next-build' / 'set-build' / 'set-marketing'. " +
                "'set-*' and 'next-*' write Info.plist + project files. " +
                "'projectDir' must be the directory containing the .xcodeproj.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "operation": SchemaBuilder.string(
                        description: "What to do.",
                        enumValues: ["read-build", "read-marketing", "next-build", "set-build", "set-marketing"]),
                    "projectDir": SchemaBuilder.string(description: "Directory containing the .xcodeproj. Default agent root."),
                    "value": SchemaBuilder.string(description: "Value for set-build / set-marketing."),
                ],
                required: ["operation"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "agvtool \(input["operation"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let op = try input.string("operation")
        let dir = try resolveInsideRoot(input.optionalString("projectDir") ?? ".",
                                        context: context)
        var args: [String] = []
        switch op {
        case "read-build":     args = ["what-version", "-terse"]
        case "read-marketing": args = ["what-marketing-version", "-terse1"]
        case "next-build":     args = ["next-version", "-all"]
        case "set-build":
            guard let v = input.optionalString("value") else {
                throw ToolError.invalidInput("'set-build' requires 'value'")
            }
            args = ["new-version", "-all", v]
        case "set-marketing":
            guard let v = input.optionalString("value") else {
                throw ToolError.invalidInput("'set-marketing' requires 'value'")
            }
            args = ["new-marketing-version", v]
        default:
            throw ToolError.invalidInput("unknown operation '\(op)'")
        }
        return try await Xcrun.run(tool: "agvtool",
                                   arguments: args,
                                   context: context,
                                   cwd: dir,
                                   timeout: 60)
    }
}
