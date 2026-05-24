import Foundation

/// `xcrun simctl install <device> <app>` — install a `.app` bundle
/// (the build product of `xcodebuild build` for an iOS/visionOS sim).
public struct SimctlInstallTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "simctl_install",
            description:
                "Installa un bundle .app su un simulator. Il simulator deve essere già avviato " +
                "(usa prima simctl_boot). Il path dell'app deve risolvere dentro la root dell'agente.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "device": SchemaBuilder.string(description: "UDID, nome, o 'booted'."),
                    "appPath": SchemaBuilder.string(description: "Bundle .app, relativo alla root dell'agente."),
                ],
                required: ["device", "appPath"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "simctl install \(input["appPath"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let device = try input.string("device")
        let appPath = try resolveInsideRoot(try input.string("appPath"), context: context)
        return try await Xcrun.run(tool: "simctl",
                                   arguments: ["install", device, appPath.path],
                                   context: context,
                                   cwd: context.rootDirectory,
                                   timeout: 120)
    }
}
