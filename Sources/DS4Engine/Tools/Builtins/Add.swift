import Foundation
import DS4Core

extension ToolRegistry {
    /// Sum of two numbers.
    static let add = binaryTool(name: "add", verb: "Add", symbol: "+") { $0 + $1 }
}
