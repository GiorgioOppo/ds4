import Foundation

/// Shared, actor-isolated state for the planning trio (`plan`, `task`,
/// `todo`). All three operate on the same `PlanState` so the model
/// can interleave them freely — `plan` lays out the high-level
/// strategy, `task` breaks the active step into substeps the model
/// is about to execute, `todo` tracks longer-running follow-ups.
///
/// The state lives in-process for the duration of the session; the
/// host can persist it to disk by snapshotting `PlanState`.
public actor PlanStore {
    private var state = PlanState()

    public init() {}

    public func snapshot() -> PlanState { state }
    public func restore(_ snapshot: PlanState) { state = snapshot }

    public func setPlan(_ text: String) {
        state.plan = text
        state.lastModified = .now
    }

    public func setTasks(_ tasks: [TaskEntry]) {
        state.tasks = tasks
        state.lastModified = .now
    }

    public func updateTask(_ id: UUID, status: TaskStatus) -> Bool {
        guard let idx = state.tasks.firstIndex(where: { $0.id == id }) else { return false }
        state.tasks[idx].status = status
        state.lastModified = .now
        return true
    }

    public func appendTodo(_ entry: TodoEntry) {
        state.todos.append(entry)
        state.lastModified = .now
    }

    public func markTodo(_ id: UUID, done: Bool) -> Bool {
        guard let idx = state.todos.firstIndex(where: { $0.id == id }) else { return false }
        state.todos[idx].done = done
        state.lastModified = .now
        return true
    }
}

public struct PlanState: Codable, Sendable {
    public var plan: String = ""
    public var tasks: [TaskEntry] = []
    public var todos: [TodoEntry] = []
    public var lastModified: Date = .now

    public init() {}
}

public enum TaskStatus: String, Codable, Sendable {
    case pending, inProgress, done, skipped
}

public struct TaskEntry: Codable, Sendable, Identifiable {
    public let id: UUID
    public var title: String
    public var status: TaskStatus

    public init(id: UUID = UUID(), title: String, status: TaskStatus = .pending) {
        self.id = id
        self.title = title
        self.status = status
    }
}

public struct TodoEntry: Codable, Sendable, Identifiable {
    public let id: UUID
    public var title: String
    public var done: Bool
    public var createdAt: Date

    public init(id: UUID = UUID(),
                title: String,
                done: Bool = false,
                createdAt: Date = .now) {
        self.id = id
        self.title = title
        self.done = done
        self.createdAt = createdAt
    }
}
