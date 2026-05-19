import Foundation

/// `xcrun simctl erase <device|all>` — wipe a simulator back to a
/// fresh-install state. Requires the simulator to be shut down.
public struct SimctlEraseTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "simctl_erase",
            description:
                "Erase a simulator's contents and settings back to fresh-install. " +
                "The target must be shut down first (use simctl_shutdown). 'device' accepts UDID, name, or 'all'.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "device": SchemaBuilder.string(description: "UDID, name, or 'all'."),
                ],
                required: ["device"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "simctl erase \(input["device"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let device = try input.string("device")
        return try await Xcrun.run(tool: "simctl",
                                   arguments: ["erase", device],
                                   context: context,
                                   cwd: context.rootDirectory,
                                   timeout: 120)
    }
}
