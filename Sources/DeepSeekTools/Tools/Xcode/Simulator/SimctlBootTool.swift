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
                "Avvia un simulator. 'device' è un UDID o un nome visualizzato (es. 'iPhone 15', " +
                "'Apple Vision Pro'). I device già avviati restituiscono un errore non fatale.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "device": SchemaBuilder.string(description: "UDID o nome del simulator."),
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
