import Foundation

/// Language-Server-Protocol bridge — scaffolded only.
///
/// A real implementation would:
///   1. Spawn `sourcekit-lsp` (Swift), `pyright`, `typescript-language-server`,
///      etc. depending on the file extension under the cursor.
///   2. Speak the LSP JSON-RPC framing over stdio (similar to MCPClient).
///   3. Expose `textDocument/definition`, `textDocument/hover`,
///      `textDocument/references`, `textDocument/publishDiagnostics`.
///
/// This stub validates inputs and returns `notImplemented` so the
/// tool surface is registered (the model sees its schema, plan-mode
/// filters work) without us pretending we can answer. Tracked in
/// `docs/ROADMAP.md` and `TODO.md`. Don't ship as a default-registered
/// tool until the plumbing lands; the registry-builder in
/// `DefaultTools.swift` skips it by default.
public struct LSPTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "lsp",
            description:
                "[stub] Language Server Protocol queries (definition, hover, " +
                "references, diagnostics). Not yet wired up to a real LSP " +
                "client — registers only so the surface is reserved.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "operation": SchemaBuilder.string(
                        description: "One of: definition, hover, references, diagnostics.",
                        enumValues: ["definition", "hover", "references", "diagnostics"]),
                    "path": SchemaBuilder.string(description: "File path."),
                    "line": SchemaBuilder.integer(description: "1-based line number.", minimum: 1),
                    "column": SchemaBuilder.integer(description: "1-based column.", minimum: 1),
                ],
                required: ["operation", "path"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "lsp \(input["operation"] as? String ?? "?") \(input["path"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        _ = try input.string("operation")
        _ = try input.string("path")
        throw ToolError.notImplemented(
            "LSP bridge is scaffolded but not wired to a real language server. " +
            "See docs/ROADMAP.md → 'LSP integration'.")
    }
}
