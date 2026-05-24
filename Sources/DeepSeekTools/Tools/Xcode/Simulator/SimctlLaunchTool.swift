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
                "Avvia un'app installata su un simulator. Restituisce il PID dell'app. " +
                "Imposta 'console=true' per agganciare stdout/stderr dell'app per la durata della chiamata " +
                "(blocca finché l'app non esce — imposta 'timeoutSeconds' di conseguenza).",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "device": SchemaBuilder.string(description: "UDID, nome, o 'booted'."),
                    "bundleId": SchemaBuilder.string(description: "Bundle identifier dell'app installata."),
                    "args": SchemaBuilder.array(itemsType: "string", description: "Argomenti da riga di comando opzionali passati all'app."),
                    "console": SchemaBuilder.boolean(description: "Aggancia la console (--console-pty). Default false.", defaultValue: false),
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
