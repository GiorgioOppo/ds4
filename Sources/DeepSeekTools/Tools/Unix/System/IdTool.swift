import Foundation
#if canImport(Darwin)
import Darwin
#endif

/// Print real and effective UID/GID. Pure Swift via libc.
public struct IdTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "id",
            description: "Stampa UID e GID reale/effettivo. L'output è 'uid=N euid=N gid=N egid=N user=NAME'.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(properties: [:])
        )
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let uid = getuid()
        let euid = geteuid()
        let gid = getgid()
        let egid = getegid()
        let user = NSUserName()
        return ToolOutput(output: "uid=\(uid) euid=\(euid) gid=\(gid) egid=\(egid) user=\(user)")
    }
}
