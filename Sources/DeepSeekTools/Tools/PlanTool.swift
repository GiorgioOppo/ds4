import Foundation

/// High-level strategy note the model writes (and rewrites) for its
/// own benefit. The UI renders this in a sticky panel so the user
/// can see the current plan without scrolling history. Reads return
/// the current text; writes replace it wholesale.
public struct PlanTool: Tool {
    public let store: PlanStore

    public init(store: PlanStore) {
        self.store = store
    }

    public var schema: ToolSchema {
        ToolSchema(
            name: "plan",
            description:
                "Read or replace the current high-level plan for the chat. " +
                "Use operation='read' to see the current plan; 'write' to " +
                "replace it with 'text'. Use freely; cheap and side-effect-free.",
            category: .planning,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "operation": SchemaBuilder.string(
                        description: "'read' or 'write'.",
                        enumValues: ["read", "write"]),
                    "text": SchemaBuilder.string(description: "New plan body (write only)."),
                ],
                required: ["operation"]
            )
        )
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let op = try input.string("operation")
        switch op {
        case "read":
            let snap = await store.snapshot()
            return ToolOutput(
                output: snap.plan.isEmpty ? "(no plan)" : snap.plan,
                metadata: ["chars": "\(snap.plan.count)"]
            )
        case "write":
            let text = try input.string("text")
            await store.setPlan(text)
            return ToolOutput(
                output: "plan updated (\(text.count) chars)",
                metadata: ["chars": "\(text.count)"]
            )
        default:
            throw ToolError.invalidInput("operation must be 'read' or 'write'")
        }
    }
}
