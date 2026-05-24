import Foundation

/// `xcrun simctl uninstall <device> <bundleId>` — remove an installed app.
public struct SimctlUninstallTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "simctl_uninstall",
            description: "Disinstalla un'app da un simulator.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "device": SchemaBuilder.string(description: "UDID, nome, o 'booted'."),
                    "bundleId": SchemaBuilder.string(description: "Bundle identifier."),
                ],
                required: ["device", "bundleId"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "simctl uninstall \(input["bundleId"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let device = try input.string("device")
        let bundleId = try input.string("bundleId")
        return try await Xcrun.run(tool: "simctl",
                                   arguments: ["uninstall", device, bundleId],
                                   context: context,
                                   cwd: context.rootDirectory,
                                   timeout: 60)
    }
}
