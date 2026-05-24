import Foundation

/// Copy a file or directory. Both endpoints must resolve inside the
/// agent root. Cross-volume copies are not atomic at the OS level —
/// `FileManager.copyItem` falls back to a streaming copy and a crash
/// mid-flight will leave the destination partially written. This is
/// accepted: the alternative (copy to .tmp + rename) only works
/// on-volume and adds bookkeeping for a corner case that doesn't
/// happen inside a normal repo checkout.
public struct CpTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "cp",
            description:
                "Copia un file o una directory dentro la root dell'agente. " +
                "Imposta 'recursive=true' per copiare un albero di directory. " +
                "Si rifiuta di sovrascrivere una destinazione esistente a meno che 'force=true'.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "src": SchemaBuilder.string(description: "Path di origine, relativo alla root dell'agente."),
                    "dst": SchemaBuilder.string(description: "Path di destinazione, relativo alla root dell'agente."),
                    "recursive": SchemaBuilder.boolean(description: "Consente la copia di un albero di directory. Default false.", defaultValue: false),
                    "force": SchemaBuilder.boolean(description: "Sovrascrive una destinazione esistente. Default false.", defaultValue: false),
                ],
                required: ["src", "dst"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "cp \(input["src"] as? String ?? "?") -> \(input["dst"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let srcRel = try input.string("src")
        let dstRel = try input.string("dst")
        let recursive = input.optionalBool("recursive") ?? false
        let force = input.optionalBool("force") ?? false
        let src = try resolveInsideRoot(srcRel, context: context)
        let dst = try resolveInsideRoot(dstRel, context: context)
        let fm = FileManager.default

        guard fm.fileExists(atPath: src.path) else {
            throw ToolError.notFound("source missing: \(srcRel)")
        }
        var srcIsDir: ObjCBool = false
        fm.fileExists(atPath: src.path, isDirectory: &srcIsDir)
        if srcIsDir.boolValue && !recursive {
            throw ToolError.invalidInput("'\(srcRel)' is a directory; set recursive=true")
        }
        if fm.fileExists(atPath: dst.path) {
            if !force {
                throw ToolError.invalidInput("destination exists: \(dstRel); set force=true to overwrite")
            }
            try fm.removeItem(at: dst)
        }
        try fm.copyItem(at: src, to: dst)
        return ToolOutput(output: "copied \(srcRel) -> \(dstRel)")
    }
}
