import Foundation

/// `xcrun simctl shutdown <device|all|booted>` — shut down a simulator.
public struct SimctlShutdownTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "simctl_shutdown",
            description:
                "Shut down a simulator. 'device' accepts a UDID, a display name, the literal 'all', " +
                "or 'booted' (every running simulator).",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "device": SchemaBuilder.string(description: "UDID, name, 'all', or 'booted'."),
                ],
                required: ["device"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "simctl shutdown \(input["device"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let device = try input.string("device")
        return try await Xcrun.run(tool: "simctl",
                                   arguments: ["shutdown", device],
                                   context: context,
                                   cwd: context.rootDirectory,
                                   timeout: 60)
    }
}
