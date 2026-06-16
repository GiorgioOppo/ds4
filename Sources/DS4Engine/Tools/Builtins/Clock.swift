import Foundation
import DS4Core

extension ToolRegistry {
    /// Current date/time in ISO-8601 (no parameters).
    static let clock = BuiltinTool(
        spec: ToolSpec(name: "now",
                       description: "Return the current local date and time in ISO-8601 format.",
                       parametersJSON: #"{"type":"object","properties":{}}"#),
        run: { _ in
            let f = ISO8601DateFormatter()
            return #"{"datetime":"\#(f.string(from: Date()))"}"#
        })
}
