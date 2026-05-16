import Foundation

/// Lightweight TODO bag — items the model wants to remember beyond
/// the current task list. The host can choose to surface these in
/// the sidebar / a notifications panel.
public struct TodoTool: Tool {
    public let store: PlanStore

    public init(store: PlanStore) { self.store = store }

    public var schema: ToolSchema {
        ToolSchema(
            name: "todo",
            description:
                "Manage a list of TODO items that outlive the current task. " +
                "Operations: 'list', 'add' (with 'title'), 'check'/'uncheck' " +
                "(with 'id').",
            category: .planning,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "operation": SchemaBuilder.string(
                        description: "'list' | 'add' | 'check' | 'uncheck'.",
                        enumValues: ["list", "add", "check", "uncheck"]),
                    "title": SchemaBuilder.string(description: "TODO body (add)."),
                    "id": SchemaBuilder.string(description: "TODO UUID (check/uncheck)."),
                ],
                required: ["operation"]
            )
        )
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let op = try input.string("operation")
        switch op {
        case "list":
            let snap = await store.snapshot()
            if snap.todos.isEmpty {
                return ToolOutput(output: "(empty)", metadata: ["count": "0"])
            }
            let body = snap.todos.map { t in
                "\(t.done ? "[x]" : "[ ]") \(t.id.uuidString)  \(t.title)"
            }.joined(separator: "\n")
            return ToolOutput(output: body, metadata: ["count": "\(snap.todos.count)"])
        case "add":
            let title = try input.string("title")
            let entry = TodoEntry(title: title)
            await store.appendTodo(entry)
            return ToolOutput(output: "added \(entry.id.uuidString)", metadata: ["id": entry.id.uuidString])
        case "check", "uncheck":
            let raw = try input.string("id")
            guard let id = UUID(uuidString: raw) else {
                throw ToolError.invalidInput("'id' is not a UUID")
            }
            let ok = await store.markTodo(id, done: op == "check")
            if !ok { throw ToolError.notFound("todo \(raw)") }
            return ToolOutput(output: "\(op) ok")
        default:
            throw ToolError.invalidInput("unknown operation '\(op)'")
        }
    }
}
