import Foundation

/// `xcrun simctl list` — inventory of simulators, runtimes, device
/// types, and pairs. JSON output by default so the model can parse
/// UDIDs reliably (display names differ between Xcode versions).
public struct SimctlListTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "simctl_list",
            description:
                "Inventory simulators, runtimes (iOS / visionOS / watchOS / tvOS), device types, and " +
                "device pairs. Pick the 'category' to scope output.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "category": SchemaBuilder.string(
                        description: "Category to list. Default 'devices'.",
                        enumValues: ["devices", "devicetypes", "runtimes", "pairs", "all"]),
                    "json": SchemaBuilder.boolean(description: "Emit JSON. Default true.", defaultValue: true),
                ]
            )
        )
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let category = input.optionalString("category") ?? "devices"
        var args: [String] = ["list"]
        if category != "all" { args.append(category) }
        if input.optionalBool("json") ?? true { args.append("--json") }
        return try await Xcrun.run(tool: "simctl",
                                   arguments: args,
                                   context: context,
                                   cwd: context.rootDirectory)
    }
}
