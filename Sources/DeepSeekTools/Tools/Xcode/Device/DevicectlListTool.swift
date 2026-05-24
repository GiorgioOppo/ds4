import Foundation

/// `xcrun devicectl list devices` — inventory of REAL connected
/// devices (iPhone / iPad / Vision Pro / Apple TV via Network). Pure
/// observation, but `devicectl` is sometimes slow when no devices are
/// connected, so we keep the timeout short.
public struct DevicectlListTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "devicectl_list",
            description:
                "Elenca i dispositivi reali connessi (iPhone / iPad / Vision Pro / Apple TV). " +
                "Imposta 'json=true' per un output parsabile da macchina. Richiede Xcode 15+.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "json": SchemaBuilder.boolean(description: "Emette JSON. Default true.", defaultValue: true),
                ]
            )
        )
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        var args: [String] = ["list", "devices"]
        if input.optionalBool("json") ?? true {
            args.append("--json-output"); args.append("/dev/stdout")
        }
        return try await Xcrun.run(tool: "devicectl",
                                   arguments: ["device"] + args,
                                   context: context,
                                   cwd: context.rootDirectory,
                                   timeout: 30)
    }
}
