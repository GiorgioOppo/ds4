import Foundation

/// Stream editor wrapper. Schema is intentionally narrow: a single
/// substitution expression and a file path. Free-form sed flags are
/// NOT accepted — most importantly, no `-i` (in-place edit). For
/// disk-mutating edits the model should use `edit` or `apply_patch`,
/// which carry their own permission story.
public struct SedTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "sed",
            description:
                "Esegue una sostituzione sed su un file e restituisce il risultato su stdout (il file NON viene modificato). " +
                "Fornisci 'pattern' (regex), 'replacement', e 'path'. " +
                "Per modifiche su disco usa invece 'edit' o 'apply_patch'.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "pattern": SchemaBuilder.string(description: "Regex di ricerca (sed BRE per default, ERE con extended=true)."),
                    "replacement": SchemaBuilder.string(description: "Stringa di sostituzione."),
                    "path": SchemaBuilder.string(description: "File di input, relativo alla root dell'agente."),
                    "global": SchemaBuilder.boolean(description: "Sostituisce tutte le occorrenze per riga (flag g). Default true.", defaultValue: true),
                    "extended": SchemaBuilder.boolean(description: "Usa regex estesa (-E). Default false.", defaultValue: false),
                ],
                required: ["pattern", "replacement", "path"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "sed s/.../.../ \(input["path"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let pattern = try input.string("pattern")
        let replacement = try input.string("replacement")
        let rel = try input.string("path")
        let global = input.optionalBool("global") ?? true
        let extended = input.optionalBool("extended") ?? false
        let url = try resolveInsideRoot(rel, context: context)

        // Build the substitution expression. Use ASCII control char
        // 0x01 as separator to avoid escaping '/' in patterns. sed
        // accepts any single byte after `s` as the delimiter.
        let sep = "\u{01}"
        if pattern.contains(sep) || replacement.contains(sep) {
            throw ToolError.invalidInput("pattern/replacement may not contain control byte 0x01")
        }
        let expr = "s\(sep)\(pattern)\(sep)\(replacement)\(sep)\(global ? "g" : "")"

        var args: [String] = []
        if extended { args.append("-E") }
        args.append(expr)
        args.append(url.path)

        return try await UnixBinary.runBinary(
            launchPath: "/usr/bin/sed",
            arguments: args,
            context: context)
    }
}
