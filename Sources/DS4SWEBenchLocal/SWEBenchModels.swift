import Foundation

struct SWETask: Decodable, Sendable {
    let instanceID: String
    let repo: String
    let baseCommit: String
    let problemStatement: String
    let testPatch: String
    let failToPass: [String]
    let passToPass: [String]

    enum CodingKeys: String, CodingKey {
        case instanceID = "instance_id"
        case repo, baseCommit = "base_commit"
        case problemStatement = "problem_statement"
        case testPatch = "test_patch"
        case failToPass = "FAIL_TO_PASS"
        case passToPass = "PASS_TO_PASS"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        instanceID = try c.decode(String.self, forKey: .instanceID)
        repo = try c.decode(String.self, forKey: .repo)
        baseCommit = try c.decode(String.self, forKey: .baseCommit)
        problemStatement = try c.decodeIfPresent(String.self, forKey: .problemStatement) ?? ""
        testPatch = try c.decodeIfPresent(String.self, forKey: .testPatch) ?? ""
        failToPass = try Self.decodeTestList(c, key: .failToPass)
        passToPass = try Self.decodeTestList(c, key: .passToPass)
    }

    private static func decodeTestList(_ c: KeyedDecodingContainer<CodingKeys>, key: CodingKeys) throws -> [String] {
        if let values = try? c.decode([String].self, forKey: key) { return values }
        if let json = try? c.decode(String.self, forKey: key),
           let data = json.data(using: .utf8),
           let values = try? JSONDecoder().decode([String].self, from: data) { return values }
        return []
    }
}

struct SWEPrediction: Decodable, Sendable {
    let instanceID: String
    let modelName: String
    let modelPatch: String?

    enum CodingKeys: String, CodingKey {
        case instanceID = "instance_id"
        case modelName = "model_name_or_path"
        case modelPatch = "model_patch"
    }
}

enum SWETestStatus: String, Codable, Sendable {
    case passed, failed, error, xfail, skipped, unknown
    var isPassing: Bool { self == .passed || self == .xfail }
}

struct SWEGrade: Codable, Sendable {
    let instanceID: String
    let patchApplied: Bool
    let resolved: Bool
    let failToPassSuccess: [String]
    let failToPassFailure: [String]
    let passToPassSuccess: [String]
    let passToPassFailure: [String]
    let exitCode: Int32
    let timedOut: Bool
    let durationSeconds: Double
    let verificationOnly: Bool

    enum CodingKeys: String, CodingKey {
        case instanceID = "instance_id", patchApplied = "patch_applied", resolved
        case failToPassSuccess = "fail_to_pass_success"
        case failToPassFailure = "fail_to_pass_failure"
        case passToPassSuccess = "pass_to_pass_success"
        case passToPassFailure = "pass_to_pass_failure"
        case exitCode = "exit_code", timedOut = "timed_out", durationSeconds = "duration_seconds"
        case verificationOnly = "verification_only"
    }
}

enum SWEGrader {
    static func grade(task: SWETask, statuses: [String: SWETestStatus], patchApplied: Bool,
                      exitCode: Int32, timedOut: Bool, duration: Double,
                      verificationOnly: Bool = false) -> SWEGrade {
        func split(_ tests: [String]) -> ([String], [String]) {
            var success: [String] = [], failure: [String] = []
            for test in tests {
                if statuses[test]?.isPassing == true { success.append(test) }
                else { failure.append(test) }
            }
            return (success, failure)
        }
        let f2p = split(task.failToPass), p2p = split(task.passToPass)
        return SWEGrade(instanceID: task.instanceID, patchApplied: patchApplied,
                        resolved: patchApplied && !timedOut && f2p.1.isEmpty && p2p.1.isEmpty,
                        failToPassSuccess: f2p.0, failToPassFailure: f2p.1,
                        passToPassSuccess: p2p.0, passToPassFailure: p2p.1,
                        exitCode: exitCode, timedOut: timedOut, durationSeconds: duration,
                        verificationOnly: verificationOnly)
    }
}
