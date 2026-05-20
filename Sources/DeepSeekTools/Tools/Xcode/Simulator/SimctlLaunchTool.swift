import Foundation

/// `xcrun simctl launch <device> <bundleId>` — launch an installed app
/// on the simulator. Returns the app's PID. Set `console=true` to
/// stream the app's stdout/stderr back through xcrun's launch.
public struct SimctlLaunchTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "simctl_launch",
            description:
                "Launch an installed app on a simulator. Returns the app PID. " +
                "Set 'console=true' to attach the app's stdout/stderr for the lifetime of the call " +
                "(blocks until the app exits — set 'timeoutSeconds' accordingly).",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "device": SchemaBuilder.string(description: "UDID, name, or 'booted'."),
                    "bundleId": SchemaBuilder.string(description: "Bundle identifier of the installed app."),
                    "args": SchemaBuilder.array(itemsType: "string", description: "Optional command-line arguments passed to the app."),
                    "console": SchemaBuilder.boolean(description: "Attach console (--console-pty). Default false.", defaultValue: false),
                    "timeoutSeconds": SchemaBuilder.integer(description: "Timeout. Default 60.", minimum: 1),
                ],
                required: ["device", "bundleId"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "simctl launch \(input["bundleId"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let device = try input.string("device")
        let bundleId = try input.string("bundleId")
        var args: [String] = ["launch"]
        if input.optionalBool("console") ?? false {
            args.append("--console-pty")
        }
        args.append(device)
        args.append(bundleId)
        for a in input.optionalStringArray("args") ?? [] { args.append(a) }
        let timeout = TimeInterval(input.optionalInteger("timeoutSeconds") ?? 60)
        return try await Xcrun.run(tool: "simctl",
                                   arguments: args,
                                   context: context,
                                   cwd: context.rootDirectory,
                                   timeout: timeout)
    }
}
