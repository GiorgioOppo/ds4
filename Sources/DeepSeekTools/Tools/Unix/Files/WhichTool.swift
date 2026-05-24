import Foundation

/// Locate a command in the host PATH. Pure Swift — splits the inherited
/// `PATH` env var and checks each directory for an executable file
/// with the given name. Returns the absolute path or 'not_found'.
public struct WhichTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "which",
            description:
                "Localizza un eseguibile su PATH. Restituisce il path assoluto o un errore not_found. " +
                "Lookup PATH in puro Swift — non esegue il binario.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "command": SchemaBuilder.string(description: "Nome del comando (senza componenti di path)."),
                    "all": SchemaBuilder.boolean(description: "Restituisce tutti i match, non solo il primo. Default false.", defaultValue: false),
                ],
                required: ["command"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "which \(input["command"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let cmd = try input.string("command")
        if cmd.contains("/") {
            throw ToolError.invalidInput("'command' must not contain '/'")
        }
        let all = input.optionalBool("all") ?? false
        let env = context.environment ?? ProcessInfo.processInfo.environment
        let path = env["PATH"] ?? "/usr/bin:/bin:/usr/sbin:/sbin"
        var found: [String] = []
        for dir in path.split(separator: ":") {
            let candidate = "\(dir)/\(cmd)"
            if FileManager.default.isExecutableFile(atPath: candidate) {
                found.append(candidate)
                if !all { break }
            }
        }
        if found.isEmpty {
            throw ToolError.notFound("'\(cmd)' not on PATH")
        }
        return ToolOutput(output: found.joined(separator: "\n"))
    }
}
