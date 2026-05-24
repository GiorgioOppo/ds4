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
                "[stub] Query Language Server Protocol (definition, hover, " +
                "references, diagnostics). Non ancora collegato a un client LSP " +
                "reale — viene registrato solo per riservare la superficie.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "operation": SchemaBuilder.string(
                        description: "Uno tra: definition, hover, references, diagnostics.",
                        enumValues: ["definition", "hover", "references", "diagnostics"]),
                    "path": SchemaBuilder.string(description: "Path del file."),
                    "line": SchemaBuilder.integer(description: "Numero di riga, a base 1.", minimum: 1),
                    "column": SchemaBuilder.integer(description: "Colonna, a base 1.", minimum: 1),
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
