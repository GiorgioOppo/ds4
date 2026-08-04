import Foundation

enum SwiftTestParser {
    /// Supports XCTest (`Test Case ... passed`) and Swift Testing's checkmark
    /// output. Expected identifiers are matched literally and fail closed.
    static func parse(_ output: String, expected: [String]) -> [String: SWETestStatus] {
        var result: [String: SWETestStatus] = [:]
        let lines = output.components(separatedBy: .newlines)
        for test in expected {
            let matching = lines.filter { $0.localizedCaseInsensitiveContains(test) }
            if matching.contains(where: isPassingLine) { result[test] = .passed }
            else if matching.contains(where: isFailingLine) { result[test] = .failed }
        }
        return result
    }

    private static func isPassingLine(_ line: String) -> Bool {
        let lower = line.lowercased()
        return lower.contains(" passed") || lower.contains("✔") || lower.contains("✓")
    }

    private static func isFailingLine(_ line: String) -> Bool {
        let lower = line.lowercased()
        return lower.contains(" failed") || lower.contains("error:") || lower.contains("✘") || lower.contains("✗")
    }
}
