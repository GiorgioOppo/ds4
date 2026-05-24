import Foundation

/// Inspect environment variables with a default redaction policy.
/// Unlike `ShellTool` (which only exposes env vars when a command
/// chooses to print them), `env` is a single-call exfiltration vector
/// for secrets, so the default behaviour redacts anything that smells
/// like a credential. The model can opt out via `unsafe: true`, which
/// is gated separately by elevating the category to `.dangerous`.
public struct EnvTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "env",
            description:
                "Elenca le variabili d'ambiente. I segreti (che corrispondono a *_TOKEN, *_KEY, *_SECRET, *_PASSWORD, " +
                "AWS_*, OPENAI_*, ANTHROPIC_*, GH_TOKEN, GITHUB_TOKEN) sono oscurati per default. " +
                "Usa 'pattern' (sottostringa) per filtrare per nome. Imposta 'unsafe=true' per bypassare l'oscuramento " +
                "(richiede consenso .dangerous).",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "pattern": SchemaBuilder.string(description: "Filtro sottostringa sui nomi delle variabili (case-insensitive)."),
                    "unsafe": SchemaBuilder.boolean(description: "Disattiva l'oscuramento dei segreti. Default false.", defaultValue: false),
                ]
            )
        )
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let pattern = input.optionalString("pattern")?.lowercased()
        let unsafe = input.optionalBool("unsafe") ?? false
        if unsafe {
            // The model is asking for the raw environment. Keep the
            // schema honest: refuse here unless the host has elevated
            // the request via the permission delegate. We don't have a
            // way to flip the category at runtime, so the simplest
            // correct behaviour is to deny with a clear hint.
            throw ToolError.permissionDenied(
                "env(unsafe=true) requires running this tool with category .dangerous; " +
                "the default registration is .readOnly. Re-register EnvTool with elevated category or " +
                "use 'shell' if the host has authorized that.")
        }

        let env = context.environment ?? ProcessInfo.processInfo.environment
        let prefixes = ["AWS_", "OPENAI_", "ANTHROPIC_"]
        let suffixes = ["_TOKEN", "_KEY", "_SECRET", "_PASSWORD"]
        let exactSecrets: Set<String> = ["GH_TOKEN", "GITHUB_TOKEN"]

        var lines: [String] = []
        for (k, v) in env.sorted(by: { $0.key < $1.key }) {
            if let pattern, !k.lowercased().contains(pattern) { continue }
            let upper = k.uppercased()
            let redact = prefixes.contains { upper.hasPrefix($0) }
                || suffixes.contains { upper.hasSuffix($0) }
                || exactSecrets.contains(upper)
            lines.append("\(k)=\(redact ? "<redacted>" : v)")
        }
        return ToolOutput(output: lines.joined(separator: "\n"))
    }
}
