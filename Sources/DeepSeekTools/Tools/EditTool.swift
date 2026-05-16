import Foundation

/// Surgical text replacement, matching opencode's `edit` and Claude
/// Code's Edit tool. The model supplies an exact `oldString` (must
/// be unique in the file unless `replaceAll` is true) and the
/// `newString` to substitute. Failing to find or non-uniqueness is
/// surfaced as an error so the model can retry with more context.
public struct EditTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "edit",
            description:
                "Replace an exact substring inside a file. 'oldString' must " +
                "appear exactly once unless 'replaceAll' is true. Preserve " +
                "whitespace and indentation verbatim — the matcher is exact.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "File to edit."),
                    "oldString": SchemaBuilder.string(description: "Exact text to replace."),
                    "newString": SchemaBuilder.string(description: "Replacement."),
                    "replaceAll": SchemaBuilder.boolean(
                        description: "Replace every occurrence. Default false.",
                        defaultValue: false),
                ],
                required: ["path", "oldString", "newString"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "edit \(input["path"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let path = try input.string("path")
        let oldString = try input.string("oldString")
        let newString = try input.string("newString")
        let replaceAll = input.optionalBool("replaceAll") ?? false

        let url = try resolveInsideRoot(path, context: context)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ToolError.notFound(path)
        }
        guard let data = try? Data(contentsOf: url),
              let original = String(data: data, encoding: .utf8) else {
            throw ToolError.invalidInput("file is not valid UTF-8")
        }
        if oldString == newString {
            throw ToolError.invalidInput("'oldString' and 'newString' are identical")
        }
        let updated: String
        let replaced: Int
        if replaceAll {
            let parts = original.components(separatedBy: oldString)
            replaced = parts.count - 1
            if replaced == 0 {
                throw ToolError.invalidInput("'oldString' not found in \(path)")
            }
            updated = parts.joined(separator: newString)
        } else {
            let parts = original.components(separatedBy: oldString)
            switch parts.count {
            case 1:
                throw ToolError.invalidInput("'oldString' not found in \(path)")
            case 2:
                replaced = 1
                updated = parts.joined(separator: newString)
            default:
                throw ToolError.invalidInput(
                    "'oldString' matches \(parts.count - 1) times — pass " +
                    "replaceAll=true or extend the snippet to make it unique")
            }
        }
        try Data(updated.utf8).write(to: url, options: .atomic)
        return ToolOutput(
            output: "edited \(url.lastPathComponent): \(replaced) replacement(s)",
            metadata: [
                "path": url.path,
                "replacements": "\(replaced)",
            ]
        )
    }
}
