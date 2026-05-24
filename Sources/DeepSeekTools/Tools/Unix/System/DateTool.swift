import Foundation

/// Print the current date/time. Pure Swift; deterministic format
/// option set so the model can choose ISO-8601, RFC 3339, or epoch.
public struct DateTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "date",
            description:
                "Data/ora corrente. Formati: 'iso8601' (default), 'epoch' (secondi dal 1970), 'rfc3339', 'human'.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "format": SchemaBuilder.string(
                        description: "Formato di output. Default 'iso8601'.",
                        enumValues: ["iso8601", "epoch", "rfc3339", "human"]),
                    "utc": SchemaBuilder.boolean(description: "Usa UTC invece dell'ora locale. Default false.", defaultValue: false),
                ]
            )
        )
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let format = input.optionalString("format") ?? "iso8601"
        let utc = input.optionalBool("utc") ?? false
        let now = Date()
        switch format {
        case "epoch":
            return ToolOutput(output: String(Int(now.timeIntervalSince1970)))
        case "rfc3339":
            let df = ISO8601DateFormatter()
            df.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            if utc { df.timeZone = TimeZone(identifier: "UTC") }
            return ToolOutput(output: df.string(from: now))
        case "human":
            let df = DateFormatter()
            df.dateStyle = .full
            df.timeStyle = .medium
            df.locale = Locale(identifier: "en_US_POSIX")
            if utc { df.timeZone = TimeZone(identifier: "UTC") }
            return ToolOutput(output: df.string(from: now))
        default:
            let df = ISO8601DateFormatter()
            if utc { df.timeZone = TimeZone(identifier: "UTC") }
            return ToolOutput(output: df.string(from: now))
        }
    }
}
