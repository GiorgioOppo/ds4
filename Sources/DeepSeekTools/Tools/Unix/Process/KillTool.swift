import Foundation
#if canImport(Darwin)
import Darwin
#endif

/// Send a signal to a process. Multiple safety rails:
///  - Refuses PID 1 and PID 0 unconditionally.
///  - Refuses the agent's own PID (no self-kill from a tool the
///    model just invoked — it would race the host process).
///  - Refuses PIDs the current UID doesn't own (probed via
///    `kill(pid, 0)` returning EPERM).
///
/// Category is `.dangerous`, not `.mutating`, because the effect is
/// outside the agent root and unrecoverable. That also means
/// `.plan` mode denies it at the registry level, and the
/// `AutoPermissionDelegate` denies it unless `allowDangerous=true`.
public struct KillTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "kill",
            description:
                "Send a signal to a process owned by the current user. " +
                "Default signal: TERM. Forbidden targets: PID 1, PID 0, and the agent process itself. " +
                "PIDs owned by other users are rejected.",
            category: .dangerous,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "pid": SchemaBuilder.integer(description: "Target PID.", minimum: 2),
                    "signal": SchemaBuilder.string(
                        description: "Signal name. Default TERM.",
                        enumValues: ["TERM", "INT", "HUP", "QUIT", "KILL", "USR1", "USR2"]),
                ],
                required: ["pid"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        let pid = input["pid"] as? Int ?? -1
        let sig = input["signal"] as? String ?? "TERM"
        return "kill -\(sig) \(pid)"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let pid = try input.integer("pid")
        let sigName = input.optionalString("signal") ?? "TERM"

        if pid <= 1 {
            throw ToolError.permissionDenied("refusing to signal PID \(pid)")
        }
        // pid_t is Int32 on macOS; reject anything that wouldn't fit
        // so the pid_t(pid) cast below can't trap.
        guard pid <= Int(Int32.max) else {
            throw ToolError.invalidInput("PID out of range")
        }
        let selfPid = Int(getpid())
        if pid == selfPid {
            throw ToolError.permissionDenied("refusing to signal the agent process (\(selfPid))")
        }
        // Probe with signal 0 to check existence + permission.
        if kill(pid_t(pid), 0) != 0 {
            let err = errno
            switch err {
            case ESRCH: throw ToolError.notFound("no such process: \(pid)")
            case EPERM: throw ToolError.permissionDenied("PID \(pid) is not owned by the current user")
            default:    throw ToolError.external("kill(0, \(pid)) failed: errno \(err)")
            }
        }

        let sig = Self.signalNumber(sigName) ?? SIGTERM
        if kill(pid_t(pid), sig) != 0 {
            throw ToolError.external("kill failed: errno \(errno)")
        }
        return ToolOutput(output: "signaled \(sigName) to PID \(pid)",
                          metadata: ["pid": "\(pid)", "signal": sigName])
    }

    private static func signalNumber(_ name: String) -> Int32? {
        switch name {
        case "TERM": return SIGTERM
        case "INT":  return SIGINT
        case "HUP":  return SIGHUP
        case "QUIT": return SIGQUIT
        case "KILL": return SIGKILL
        case "USR1": return SIGUSR1
        case "USR2": return SIGUSR2
        default:     return nil
        }
    }
}
