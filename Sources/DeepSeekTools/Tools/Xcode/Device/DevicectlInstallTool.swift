import Foundation

/// `xcrun devicectl device install app -d <id> <path>` — install an
/// app on a REAL device.
///
/// Category `.dangerous` rather than `.mutating`: installing onto a
/// physical device a developer may be holding has a larger blast
/// radius than mutating a simulator state — wrong build, wrong
/// signing identity, accidental data wipe risks. Stays denied in
/// `.plan` mode and requires explicit consent in `.build`.
public struct DevicectlInstallTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "devicectl_install",
            description:
                "Installa un'app su un dispositivo reale. Richiede l'identificatore del dispositivo " +
                "(da devicectl_list) e un bundle .app o .ipa firmato dentro la root dell'agente. " +
                "Categorizzato come dangerous — coinvolge hardware fisico.",
            category: .dangerous,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "device": SchemaBuilder.string(description: "Identificatore del dispositivo (UDID o nome da devicectl_list)."),
                    "appPath": SchemaBuilder.string(description: ".app o .ipa, relativo alla root dell'agente."),
                ],
                required: ["device", "appPath"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "devicectl install \(input["appPath"] as? String ?? "?") -> \(input["device"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let device = try input.string("device")
        let appPath = try resolveInsideRoot(try input.string("appPath"), context: context)
        let args = ["device", "install", "app", "-d", device, appPath.path]
        return try await Xcrun.run(tool: "devicectl",
                                   arguments: args,
                                   context: context,
                                   cwd: context.rootDirectory,
                                   timeout: 300)
    }
}
