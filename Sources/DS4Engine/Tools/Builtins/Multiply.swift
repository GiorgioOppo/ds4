import Foundation
import DS4Core

extension ToolRegistry {
    /// Product of two numbers.
    static let multiply = binaryTool(name: "multiply", verb: "Multiply", symbol: "×") { $0 * $1 }
}
