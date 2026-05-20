import Foundation

/// `xcrun simctl boot <device>` — boot a simulator. Idempotent on
/// already-booted devices (simctl returns a non-zero exit but the
/// state is correct — we surface stderr but don't fail spuriously).
public struct SimctlBootTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "simctl_boot",
            description:
                "Boot a simulator. 'device' is a UDID or a display name (e.g. 'iPhone 15', " +
                "'Apple Vision Pro'). Already-booted devices return a non-fatal error.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "device": SchemaBuilder.string(description: "Simulator UDID or name."),
                ],
                required: ["device"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "simctl boot \(input["device"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let device = try input.string("device")
        return try await Xcrun.run(tool: "simctl",
                                   arguments: ["boot", device],
                                   context: context,
                                   cwd: context.rootDirectory,
                                   timeout: 180)
    }
}
