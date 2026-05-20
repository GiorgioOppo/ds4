import Foundation

/// `git blame` of a file, optionally restricted to a line range.
public struct GitBlameTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "git_blame",
            description:
                "Show line-by-line authorship for a file. Optional 'startLine'+'endLine' " +
                "to restrict to a range.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "File path, relative to agent root."),
                    "startLine": SchemaBuilder.integer(description: "1-based starting line.", minimum: 1),
                    "endLine": SchemaBuilder.integer(description: "1-based ending line.", minimum: 1),
                ],
                required: ["path"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "git_blame \(input["path"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let rel = try input.string("path")
        let url = try resolveInsideRoot(rel, context: context)
        var args: [String] = ["blame"]
        if let start = input.optionalInteger("startLine") {
            let end = input.optionalInteger("endLine") ?? start
            args.append("-L"); args.append("\(start),\(end)")
        }
        args.append(url.path)
        return try await UnixBinary.runBinary(
            launchPath: "/usr/bin/git",
            arguments: args,
            context: context,
            cwd: context.rootDirectory)
    }
}
