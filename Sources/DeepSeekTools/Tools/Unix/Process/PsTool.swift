import Foundation

/// Snapshot the host's process list. Read-only.
public struct PsTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "ps",
            description:
                "Snapshot della lista dei processi. Per default restituisce i processi dell'utente corrente con " +
                "PID/CPU/MEM/CMD. Imposta 'all=true' per ogni processo (a livello di sistema).",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "all": SchemaBuilder.boolean(description: "Include i processi di tutti gli utenti. Default false.", defaultValue: false),
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
