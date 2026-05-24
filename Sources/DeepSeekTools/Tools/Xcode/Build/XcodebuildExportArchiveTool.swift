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
                "Esporta un .xcarchive in .ipa/.app/.pkg usando un exportOptions.plist. " +
                "Tutti e tre i path (archivio, directory di export, plist delle opzioni) devono trovarsi dentro la root dell'agente.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "archivePath": SchemaBuilder.string(description: "Path del .xcarchive, relativo alla root dell'agente."),
                    "exportPath": SchemaBuilder.string(description: "Directory di output, relativa alla root dell'agente."),
                    "exportOptionsPlist": SchemaBuilder.string(description: "Path di exportOptions.plist, relativo alla root dell'agente."),
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
