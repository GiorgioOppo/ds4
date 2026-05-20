import Foundation

/// Print the effective user name.
public struct WhoamiTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "whoami",
            description: "Print the effective user name.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(properties: [:])
        )
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        ToolOutput(output: NSUserName())
    }
}
