import Foundation

/// `xcodebuild -exportArchive` — turn a `.xcarchive` into a
/// distributable artifact (`.ipa`, `.app`, `.pkg`, …) per an
/// `exportOptions.plist`.
public struct XcodebuildExportArchiveTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "xcodebuild_exportarchive",
            description:
                "Export a .xcarchive to .ipa/.app/.pkg using an exportOptions.plist. " +
                "All three paths (archive, export dir, options plist) must be inside the agent root.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "archivePath": SchemaBuilder.string(description: ".xcarchive path, relative to agent root."),
                    "exportPath": SchemaBuilder.string(description: "Output directory, relative to agent root."),
                    "exportOptionsPlist": SchemaBuilder.string(description: "exportOptions.plist path, relative to agent root."),
                ],
                required: ["archivePath", "exportPath", "exportOptionsPlist"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "xcodebuild exportArchive \(input["archivePath"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let archive = try resolveInsideRoot(try input.string("archivePath"), context: context)
        let exportDir = try resolveInsideRoot(try input.string("exportPath"), context: context)
        let plist = try resolveInsideRoot(try input.string("exportOptionsPlist"), context: context)
        try FileManager.default.createDirectory(at: exportDir, withIntermediateDirectories: true)
        let args = [
            "-exportArchive",
            "-archivePath", archive.path,
            "-exportPath", exportDir.path,
            "-exportOptionsPlist", plist.path,
        ]
        return try await Xcrun.run(tool: "xcodebuild",
                                   arguments: args,
                                   context: context,
                                   cwd: context.rootDirectory,
                                   timeout: 600)
    }
}
