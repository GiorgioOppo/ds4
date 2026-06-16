import Foundation
import DS4Core

extension ToolRegistry {
    /// Evaluate a basic arithmetic expression (+ - * / parentheses).
    static let calculator = BuiltinTool(
        spec: ToolSpec(name: "calculator",
                       description: "Evaluate a basic arithmetic expression with + - * / and parentheses.",
                       parametersJSON: #"{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}"#),
        run: { argsJSON in
            guard let data = argsJSON.data(using: .utf8),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let expr = obj["expression"] as? String else {
                return #"{"error":"missing 'expression' argument"}"#
            }
            return evaluateArithmetic(expr)
        })
}
