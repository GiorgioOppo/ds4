import Foundation

/// Base64 encode/decode. Operates on inline strings (default) or on a
/// sandboxed file. Pure Swift via `Data` extensions.
public struct Base64Tool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "base64",
            description:
                "Codifica o decodifica base64. Fornisci 'input' (stringa) o 'path' (file). " +
                "Imposta 'decode=true' per decodificare (il default è encode). " +
                "L'output è il payload codificato/decodificato come stringa UTF-8 (o marker '<binary>' se non è testo).",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "input": SchemaBuilder.string(description: "Testo inline. Alternativa a 'path'."),
                    "path": SchemaBuilder.string(description: "Path del file, relativo alla root dell'agente."),
                    "decode": SchemaBuilder.boolean(description: "Decodifica invece di codificare. Default false.", defaultValue: false),
                ]
            )
        )
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let decode = input.optionalBool("decode") ?? false
        let payload: Data
        if let inline = input.optionalString("input") {
            guard let d = inline.data(using: .utf8) else {
                throw ToolError.invalidInput("'input' is not valid UTF-8")
            }
            payload = d
        } else if let rel = input.optionalString("path") {
            let url = try resolveInsideRoot(rel, context: context)
            guard let d = try? Data(contentsOf: url) else {
                throw ToolError.notFound("cannot read '\(rel)'")
            }
            payload = d
        } else {
            throw ToolError.invalidInput("provide 'input' or 'path'")
        }

        if decode {
            // Drop whitespace so wrapped base64 (76-col lines) decodes too.
            let stripped = String(data: payload, encoding: .utf8)?
                .filter { !$0.isWhitespace } ?? ""
            guard let decoded = Data(base64Encoded: stripped) else {
                throw ToolError.invalidInput("not valid base64")
            }
            if let text = String(data: decoded, encoding: .utf8) {
                return ToolOutput(output: text)
            }
            return ToolOutput(output: "<binary, \(decoded.count) bytes>",
                              metadata: ["bytes": "\(decoded.count)"])
        }
        return ToolOutput(output: payload.base64EncodedString())
    }
}
