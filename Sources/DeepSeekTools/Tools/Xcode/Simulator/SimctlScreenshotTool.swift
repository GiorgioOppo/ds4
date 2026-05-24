import Foundation

/// `xcrun simctl io <device> screenshot <path>` — snapshot a booted
/// simulator's screen. Categorized `.mutating` because it writes a
/// file to disk; the read of the simulator itself is observational.
public struct SimctlScreenshotTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "simctl_screenshot",
            description:
                "Cattura uno screenshot di un simulator avviato. Il file di output viene scritto dentro la root dell'agente.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "device": SchemaBuilder.string(description: "UDID, nome, o 'booted'."),
                    "outputPath": SchemaBuilder.string(description: "Path dell'immagine di output, relativo alla root dell'agente."),
                    "format": SchemaBuilder.string(
                        description: "Formato dell'immagine. Default png.",
                        enumValues: ["png", "tiff", "bmp", "gif", "jpeg"]),
                ],
                required: ["device", "outputPath"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "simctl screenshot -> \(input["outputPath"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let device = try input.string("device")
        let outPath = try resolveInsideRoot(try input.string("outputPath"), context: context)
        var args: [String] = ["io", device, "screenshot"]
        if let fmt = input.optionalString("format") {
            args.append("--type=\(fmt)")
        }
        args.append(outPath.path)
        return try await Xcrun.run(tool: "simctl",
                                   arguments: args,
                                   context: context,
                                   cwd: context.rootDirectory,
                                   timeout: 30)
    }
}
