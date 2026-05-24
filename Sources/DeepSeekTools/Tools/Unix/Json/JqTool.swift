import Foundation

/// jq wrapper. `jq` is not part of base macOS, so we probe the common
/// install locations (Homebrew arm64, Homebrew Intel, MacPorts, Nix).
/// On miss we throw `notFound` with a brew install hint — the model
/// surfaces that to the user verbatim.
public struct JqTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "jq",
            description:
                "Esegue un filtro jq su un file JSON o su una stringa inline. Richiede jq installato " +
                "(brew install jq, MacPorts, o Nix). Per lookup banali di proprietà valuta invece il parsing " +
                "JSON di Foundation.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "filter": SchemaBuilder.string(description: "Espressione jq (es. '.users[0].name')."),
                    "input": SchemaBuilder.string(description: "Stringa JSON inline. Alternativa a 'path'."),
                    "path": SchemaBuilder.string(description: "Path del file JSON, relativo alla root dell'agente."),
                    "raw": SchemaBuilder.boolean(description: "Output raw (-r) — rimuove le virgolette JSON dalle stringhe. Default false.", defaultValue: false),
                ],
                required: ["filter"]
            )
        )
    }

    private static let candidates = [
        "/opt/homebrew/bin/jq",
        "/usr/local/bin/jq",
        "/opt/local/bin/jq",
    ]

    public func permissionSummary(input: [String: Any]) -> String {
        "jq \(input["filter"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let filter = try input.string("filter")
        let raw = input.optionalBool("raw") ?? false
        guard let jqPath = UnixBinary.resolveBinary(candidates: Self.candidates) else {
            throw ToolError.notFound("jq not installed. Try: brew install jq")
        }

        var args: [String] = []
        if raw { args.append("-r") }
        args.append(filter)

        if let rel = input.optionalString("path") {
            let url = try resolveInsideRoot(rel, context: context)
            args.append(url.path)
            return try await UnixBinary.runBinary(
                launchPath: jqPath, arguments: args, context: context)
        }
        guard let inline = input.optionalString("input") else {
            throw ToolError.invalidInput("provide 'path' or 'input'")
        }
        // jq reads stdin when no positional file is given. We use a
        // small detour: write 'input' to a temp file inside /tmp, run
        // jq on it, then delete. Avoids piping into Process which is
        // outside our `_UnixBinary` API surface.
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("jq-\(UUID().uuidString).json")
        try inline.data(using: .utf8)?.write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }
        args.append(tmp.path)
        return try await UnixBinary.runBinary(
            launchPath: jqPath, arguments: args, context: context)
    }
}
