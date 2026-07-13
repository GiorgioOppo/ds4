import Foundation
import DS4Core

extension ToolRegistry {
    /// Difference of two numbers (a − b).
    static let subtract = binaryTool(name: "subtract", verb: "Subtract", symbol: "−") { $0 - $1 }
}
