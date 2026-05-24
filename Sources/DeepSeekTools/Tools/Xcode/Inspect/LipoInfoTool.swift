import Foundation

/// `lipo -info` — list the architectures inside a Mach-O binary
/// (e.g. `arm64`, `x86_64`, or `arm64 x86_64` for a universal build).
public struct LipoInfoTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "lipo_info",
            description:
                "Stampa le architetture presenti in un binario Mach-O (single-arch, fat/universal, " +
                "o arm64+arm64e per slice iOS legacy).",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "Binario Mach-O, relativo alla root dell'agente."),
                    "detailed": SchemaBuilder.boolean(description: "Usa -detailed_info per offset/dimensione delle slice. Default false.", defaultValue: false),
                ],
                required: ["path"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "lipo -info \(input["path"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let url = try resolveInsideRoot(try input.string("path"), context: context)
        let flag = (input.optionalBool("detailed") ?? false) ? "-detailed_info" : "-info"
        return try await UnixBinary.runBinary(
            launchPath: "/usr/bin/lipo",
            arguments: [flag, url.path],
            context: context,
            timeout: 30)
    }
}
