import Foundation

/// Snapshot the host's process list. Read-only.
public struct PsTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "ps",
            description:
                "Snapshot the process list. By default returns the current user's processes with " +
                "PID/CPU/MEM/CMD. Set 'all=true' for every process (system-wide).",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "all": SchemaBuilder.boolean(description: "Include all users' processes. Default false.", defaultValue: false),
                ]
            )
        )
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let all = input.optionalBool("all") ?? false
        // Portable-ish flags: `-o pid,pcpu,pmem,command` works on
        // macOS BSD ps and Linux procps.
        var args: [String] = ["-o", "pid,pcpu,pmem,command"]
        if all { args.insert(contentsOf: ["-A"], at: 0) }
        return try await UnixBinary.runBinary(
            launchPath: "/bin/ps",
            arguments: args,
            context: context)
    }
}
