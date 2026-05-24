import Foundation

/// `xcodebuild archive` — produce a .xcarchive ready for export.
/// The archive path must be inside the agent root.
public struct XcodebuildArchiveTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "xcodebuild_archive",
            description:
                "Archivia uno scheme per la distribuzione (.xcarchive). Fornisci 'scheme', 'archivePath' " +
                "(relativo alla root dell'agente), 'workspace' o 'project', e 'configuration' (tipicamente Release).",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "scheme": SchemaBuilder.string(description: "Nome dello scheme."),
                    "archivePath": SchemaBuilder.string(description: ".xcarchive di output, relativo alla root dell'agente."),
                    "workspace": SchemaBuilder.string(description: ".xcworkspace, relativo alla root dell'agente."),
                    "project": SchemaBuilder.string(description: ".xcodeproj, relativo alla root dell'agente."),
                    "configuration": SchemaBuilder.string(description: "Configurazione. Default Release."),
                    "destination": SchemaBuilder.string(description: "Stringa di destinazione xcodebuild."),
                    "derivedDataPath": SchemaBuilder.string(description: "Directory DerivedData, relativa alla root dell'agente."),
                    "timeoutSeconds": SchemaBuilder.integer(description: "Timeout. Default 900.", minimum: 1),
                ],
                required: ["scheme", "archivePath"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "xcodebuild archive \(input["scheme"] as? String ?? "?") -> \(input["archivePath"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let scheme = try input.string("scheme")
        let archiveRel = try input.string("archivePath")
        let archive = try resolveInsideRoot(archiveRel, context: context)
        let cfg = input.optionalString("configuration") ?? "Release"

        var args: [String] = [
            "-scheme", scheme,
            "-archivePath", archive.path,
            "-configuration", cfg,
        ]
        if let ws = input.optionalString("workspace") {
            let url = try resolveInsideRoot(ws, context: context)
            args.append("-workspace"); args.append(url.path)
        } else if let proj = input.optionalString("project") {
            let url = try resolveInsideRoot(proj, context: context)
            args.append("-project"); args.append(url.path)
        }
        if let dest = input.optionalString("destination") {
            args.append("-destination"); args.append(dest)
        }
        if let ddpRel = input.optionalString("derivedDataPath") {
            let ddp = try resolveInsideRoot(ddpRel, context: context)
            args.append("-derivedDataPath"); args.append(ddp.path)
        }
        args.append("archive")
        let timeout = TimeInterval(input.optionalInteger("timeoutSeconds") ?? 900)
        return try await Xcrun.run(tool: "xcodebuild",
                                   arguments: args,
                                   context: context,
                                   cwd: context.rootDirectory,
                                   timeout: timeout)
    }
}
