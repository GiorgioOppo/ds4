import Foundation

/// List open files. Wraps `/usr/sbin/lsof` with a narrow schema —
/// either filter by PID or by file path. Free-form lsof selectors
/// aren't exposed.
public struct LsofTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "lsof",
            description:
                "List open files. Provide 'pid' to inspect one process, or 'path' to find who holds a file. " +
                "Without arguments, lists every open file the current user can see.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "pid": SchemaBuilder.integer(description: "Process ID to inspect.", minimum: 1),
                    "path": SchemaBuilder.string(description: "File path to look up (absolute or relative to agent root)."),
                ]
            )
        )
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        var args: [String] = []
        if let pid = input.optionalInteger("pid") {
            args.append("-p"); args.append("\(pid)")
        }
        if let rel = input.optionalString("path") {
            let url = try resolveInsideRoot(rel, context: context)
            args.append(url.path)
        }
        return try await UnixBinary.runBinary(
            launchPath: "/usr/sbin/lsof",
            arguments: args,
            context: context,
            timeout: 15)
    }
}
