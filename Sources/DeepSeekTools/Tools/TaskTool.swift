import Foundation

/// Ordered list of substeps the model is working through right now,
/// each with a status (`pending` / `inProgress` / `done` / `skipped`).
/// Distinct from `todo` (longer-term, possibly cross-session) and
/// from `plan` (free-form narrative).
public struct TaskTool: Tool {
    public let store: PlanStore

    public init(store: PlanStore) { self.store = store }

    public var schema: ToolSchema {
        ToolSchema(
            name: "task",
            description:
                "Manage the active task list. Operations: 'list' returns the " +
                "current items; 'set' replaces them ('titles' array); 'update' " +
                "flips one item's status by 'id'.",
            category: .planning,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "operation": SchemaBuilder.string(
                        description: "'list' | 'set' | 'update'.",
                        enumValues: ["list", "set", "update"]),
                    "titles": SchemaBuilder.array(itemsType: "string",
                                                  description: "For 'set': new task titles."),
                    "id": SchemaBuilder.string(description: "Task UUID (update)."),
                    "status": SchemaBuilder.string(
                        description: "New status (update).",
                        enumValues: ["pending", "inProgress", "done", "skipped"]),
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
            if snap.tasks.isEmpty {
                return ToolOutput(output: "(no tasks)", metadata: ["count": "0"])
            }
            let body = snap.tasks.map { t in
                "\(symbol(for: t.status)) \(t.id.uuidString)  \(t.title)"
            }.joined(separator: "\n")
            return ToolOutput(output: body, metadata: ["count": "\(snap.tasks.count)"])
        case "set":
            guard let titles = input.optionalStringArray("titles") else {
                throw ToolError.invalidInput("'titles' must be an array of strings")
            }
            let tasks = titles.map { TaskEntry(title: $0) }
            await store.setTasks(tasks)
            return ToolOutput(output: "set \(tasks.count) task(s)", metadata: ["count": "\(tasks.count)"])
        case "update":
            let raw = try input.string("id")
            guard let id = UUID(uuidString: raw) else {
                throw ToolError.invalidInput("'id' is not a UUID")
            }
            let status = try input.string("status")
            guard let parsed = TaskStatus(rawValue: status) else {
                throw ToolError.invalidInput("unknown status '\(status)'")
            }
            let ok = await store.updateTask(id, status: parsed)
            if !ok { throw ToolError.notFound("task \(raw)") }
            return ToolOutput(output: "task updated → \(status)")
        default:
            throw ToolError.invalidInput("unknown operation '\(op)'")
        }
    }

    private func symbol(for status: TaskStatus) -> String {
        switch status {
        case .pending:    return "[ ]"
        case .inProgress: return "[~]"
        case .done:       return "[x]"
        case .skipped:    return "[-]"
        }
    }
}
