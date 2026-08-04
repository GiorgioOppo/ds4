import Foundation

enum JSONValue: Codable, Equatable, Sendable {
    case null, bool(Bool), number(Double), string(String)
    case array([JSONValue]), object([String: JSONValue])

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null }
        else if let value = try? c.decode(Bool.self) { self = .bool(value) }
        else if let value = try? c.decode(Double.self) { self = .number(value) }
        else if let value = try? c.decode(String.self) { self = .string(value) }
        else if let value = try? c.decode([JSONValue].self) { self = .array(value) }
        else { self = .object(try c.decode([String: JSONValue].self)) }
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .null: try c.encodeNil()
        case .bool(let value): try c.encode(value)
        case .number(let value): try c.encode(value)
        case .string(let value): try c.encode(value)
        case .array(let value): try c.encode(value)
        case .object(let value): try c.encode(value)
        }
    }
}

struct JSONTestCase: Codable, Sendable {
    let id: String
    let value: JSONValue?
    let error: String?
}

struct JSONTestDocument: Codable, Sendable {
    let instanceID: String
    let cases: [JSONTestCase]
    enum CodingKeys: String, CodingKey { case instanceID = "instance_id", cases }
}

struct JSONTestComparison: Codable, Sendable {
    let instanceID: String
    let passed: Bool
    let passedCases: [String]
    let failedCases: [String]
    let missingCases: [String]
    let unexpectedCases: [String]
    enum CodingKeys: String, CodingKey {
        case instanceID = "instance_id", passed
        case passedCases = "passed_cases", failedCases = "failed_cases"
        case missingCases = "missing_cases", unexpectedCases = "unexpected_cases"
    }
}

enum JSONTestComparator {
    static func compare(expected: JSONTestDocument, actual: JSONTestDocument) throws -> JSONTestComparison {
        guard expected.instanceID == actual.instanceID else {
            throw SWELocalError.invalid("instance_id diverso tra risultati expected e actual")
        }
        let expectedByID = try uniqueCases(expected.cases, label: "expected")
        let actualByID = try uniqueCases(actual.cases, label: "actual")
        var passed: [String] = [], failed: [String] = [], missing: [String] = []
        for id in expectedByID.keys.sorted() {
            guard let actualCase = actualByID[id] else { missing.append(id); continue }
            let expectedCase = expectedByID[id]!
            if expectedCase.value == actualCase.value && expectedCase.error == actualCase.error {
                passed.append(id)
            } else { failed.append(id) }
        }
        let unexpected = actualByID.keys.filter { expectedByID[$0] == nil }.sorted()
        return JSONTestComparison(instanceID: expected.instanceID,
                                  passed: failed.isEmpty && missing.isEmpty && unexpected.isEmpty,
                                  passedCases: passed, failedCases: failed,
                                  missingCases: missing, unexpectedCases: unexpected)
    }

    private static func uniqueCases(_ cases: [JSONTestCase], label: String) throws -> [String: JSONTestCase] {
        var result: [String: JSONTestCase] = [:]
        for testCase in cases {
            guard !testCase.id.isEmpty else { throw SWELocalError.invalid("Caso senza id nel file \(label)") }
            guard result.updateValue(testCase, forKey: testCase.id) == nil else {
                throw SWELocalError.invalid("Caso duplicato nel file \(label): \(testCase.id)")
            }
        }
        return result
    }
}
