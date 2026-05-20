import Foundation

/// `xcodebuild archive` — produce a .xcarchive ready for export.
/// The archive path must be inside the agent root.
public struct XcodebuildArchiveTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "xcodebuild_archive",
            description:
                "Archive a scheme for distribution (.xcarchive). Provide 'scheme', 'archivePath' " +
                "(relative to agent root), 'workspace' or 'project', and 'configuration' (typically Release).",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "scheme": SchemaBuilder.string(description: "Scheme name."),
                    "archivePath": SchemaBuilder.string(description: "Output .xcarchive, relative to agent root."),
                    "workspace": SchemaBuilder.string(description: ".xcworkspace, relative to agent root."),
                    "project": SchemaBuilder.string(description: ".xcodeproj, relative to agent root."),
                    "configuration": SchemaBuilder.string(description: "Configuration. Default Release."),
                    "destination": SchemaBuilder.string(description: "xcodebuild destination string."),
                    "derivedDataPath": SchemaBuilder.string(description: "DerivedData dir, relative to agent root."),
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
