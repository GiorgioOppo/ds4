import Foundation

/// Print the host name. Pure Swift via `ProcessInfo`.
public struct HostnameTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "hostname",
            description: "Stampa il nome dell'host.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(properties: [:])
        )
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        ToolOutput(output: ProcessInfo.processInfo.hostName)
    }
}
