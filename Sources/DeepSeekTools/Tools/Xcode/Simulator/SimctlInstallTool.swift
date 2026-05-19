import Foundation

/// `xcrun simctl install <device> <app>` — install a `.app` bundle
/// (the build product of `xcodebuild build` for an iOS/visionOS sim).
public struct SimctlInstallTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "simctl_install",
            description:
                "Install a .app bundle on a simulator. The simulator must already be booted " +
                "(use simctl_boot first). The app path must resolve inside the agent root.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "device": SchemaBuilder.string(description: "UDID, name, or 'booted'."),
                    "appPath": SchemaBuilder.string(description: ".app bundle, relative to agent root."),
                ],
                required: ["device", "appPath"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "simctl install \(input["appPath"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let device = try input.string("device")
        let appPath = try resolveInsideRoot(try input.string("appPath"), context: context)
        return try await Xcrun.run(tool: "simctl",
                                   arguments: ["install", device, appPath.path],
                                   context: context,
                                   cwd: context.rootDirectory,
                                   timeout: 120)
    }
}
