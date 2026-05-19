import Foundation
#if canImport(Darwin)
import Darwin
#endif

/// Print kernel name / hostname / release / machine. Pure Swift via
/// `uname(2)` — no shell-out.
public struct UnameTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "uname",
            description:
                "Kernel info: 'kernel' (default), 'machine' (CPU arch), 'release' (kernel version), 'all'.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "field": SchemaBuilder.string(
                        description: "Which field to print. Default 'kernel'.",
                        enumValues: ["kernel", "machine", "release", "all"]),
                ]
            )
        )
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let field = input.optionalString("field") ?? "kernel"
        var info = utsname()
        uname(&info)
        let sys = withUnsafeBytes(of: &info.sysname) {
            String(cString: $0.baseAddress!.assumingMemoryBound(to: CChar.self))
        }
        let release = withUnsafeBytes(of: &info.release) {
            String(cString: $0.baseAddress!.assumingMemoryBound(to: CChar.self))
        }
        let machine = withUnsafeBytes(of: &info.machine) {
            String(cString: $0.baseAddress!.assumingMemoryBound(to: CChar.self))
        }
        let host = ProcessInfo.processInfo.hostName

        switch field {
        case "machine": return ToolOutput(output: machine)
        case "release": return ToolOutput(output: release)
        case "all":     return ToolOutput(output: "\(sys) \(host) \(release) \(machine)")
        default:        return ToolOutput(output: sys)
        }
    }
}
