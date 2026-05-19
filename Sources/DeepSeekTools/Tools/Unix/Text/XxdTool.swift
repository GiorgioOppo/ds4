import Foundation

/// Hex dump a file. Pure Swift; output uses the canonical `xxd` layout
/// (offset, 16 hex bytes, ASCII gutter).
public struct XxdTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "xxd",
            description:
                "Hex dump a file in canonical 16-bytes-per-row layout. " +
                "Optional 'offset' / 'length' (bytes) to dump a slice. " +
                "Output is capped at 32 KB.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "File path, relative to agent root."),
                    "offset": SchemaBuilder.integer(description: "Starting byte offset. Default 0.", minimum: 0),
                    "length": SchemaBuilder.integer(description: "Max bytes to dump. Default 4096.", minimum: 1),
                ],
                required: ["path"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "xxd \(input["path"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let rel = try input.string("path")
        let offset = input.optionalInteger("offset") ?? 0
        let length = input.optionalInteger("length") ?? 4096
        let url = try resolveInsideRoot(rel, context: context)
        guard let data = try? Data(contentsOf: url) else {
            throw ToolError.notFound("cannot read '\(rel)'")
        }
        let end = min(data.count, offset + length)
        guard offset < data.count else {
            return ToolOutput(output: "")
        }
        let slice = data.subdata(in: offset..<end)
        var lines: [String] = []
        for row in stride(from: 0, to: slice.count, by: 16) {
            let chunk = slice[row..<min(row + 16, slice.count)]
            let hex = chunk.map { String(format: "%02x", $0) }
                .chunked(into: 2)
                .map { $0.joined() }
                .joined(separator: " ")
                .padding(toLength: 39, withPad: " ", startingAt: 0)
            let ascii = chunk.map { byte -> String in
                let scalar = Unicode.Scalar(byte)
                let c = Character(scalar)
                return (c.isASCII && byte >= 0x20 && byte < 0x7f) ? String(c) : "."
            }.joined()
            lines.append(String(format: "%08x: %@ %@", offset + row, hex, ascii))
        }
        let body = lines.joined(separator: "\n")
        return ToolOutput(output: UnixBinary.capOutput(body, max: UnixBinary.defaultOutputCap),
                          metadata: ["bytes": "\(slice.count)"])
    }
}

private extension Array {
    func chunked(into size: Int) -> [[Element]] {
        stride(from: 0, to: count, by: size).map {
            Array(self[$0..<Swift.min($0 + size, count)])
        }
    }
}
