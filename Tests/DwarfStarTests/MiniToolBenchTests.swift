import Foundation
import XCTest
@testable import DwarfStar

final class MiniToolBenchTests: XCTestCase {
    private let manifestHash = String(repeating: "a", count: 64)
    private let profileHash = String(repeating: "b", count: 64)

    private func configuration() -> MiniToolBenchConfiguration {
        var value = MiniToolBenchConfiguration()
        value.modelName = "DeepSeek-V4-Flash-0731"
        value.endpoint = "http://192.0.2.10:8080/v1"
        value.platform = "apple-m1-pro"
        return value
    }

    private func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mini-tool-bench-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    private func write(_ object: [String: Any], name: String, in directory: URL) throws {
        try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
            .write(to: directory.appendingPathComponent(name))
    }

    private var suite: [String: Any] {
        ["schema_version": 1, "id": "core19", "version": "1.0.0", "manifest_hash": manifestHash,
         "task_hash_algorithm": "terminal-bench-local-task-tree-v1"]
    }

    private func summary(_ names: [String] = ["results-git-leak-recovery.json"],
                         passed: Int = 1) -> [String: Any] {
        ["schema_version": 4, "generated_at": "2026-09-06T12:00:00+00:00",
         "model": ["id": "/models/flash.gguf", "name": "DeepSeek-V4-Flash-0731"],
         "platform": ["id": "apple-m1-pro", "name": "Apple M1 Pro"],
         "suite": suite, "profile_hash": profileHash, "engine": "DwarfStar", "backend": "metal",
         "quant": "IQ2XXS-w2Q2K", "total_tasks": names.count, "passed_tasks": passed,
         "pass_rate": names.isEmpty ? 0.0 : Double(passed) / Double(names.count),
         "total_duration_ms": 2_500, "tokens": ["input": 100, "cached": 20, "output": 50],
         "results": names]
    }

    private func task(_ id: String = "git-leak-recovery", passed: Bool = true) -> [String: Any] {
        let attempts: [[String: Any]] = [
            ["attempt": 1, "trial_name": "\(id)__first", "passed": false, "reward": 0.0,
             "duration_ms": 1_000, "transcript": "transcript-\(id)-attempt1.json",
             "exception": NSNull(), "tokens": ["input": 40, "cached": 0, "output": 20]],
            ["attempt": 2, "trial_name": "\(id)__second", "passed": passed,
             "reward": passed ? 1.0 : 0.0, "duration_ms": 1_500,
             "transcript": "transcript-\(id)-attempt2.json", "exception": NSNull(),
             "tokens": ["input": 60, "cached": 20, "output": 30]],
        ]
        return ["schema_version": 4, "task": id, "completed": true, "passed": passed,
                "reward": passed ? 1.0 : 0.0, "duration_ms": 2_500,
                "tokens": ["input": 100, "cached": 20, "output": 50], "agent_steps": 3,
                "attempts": attempts, "succeeded_at_attempt": passed ? 2 : NSNull(),
                "transcript": "transcript-\(id)-attempt2.json",
                "model": ["id": "/models/flash.gguf", "name": "DeepSeek-V4-Flash-0731"],
                "platform": ["id": "apple-m1-pro", "name": "Apple M1 Pro"],
                "suite": suite, "profile_hash": profileHash,
                "evaluation_profile": ["benchmark": "Terminal-Bench-Local", "suite": suite],
                "engine": "DwarfStar", "backend": "metal"]
    }

    private func completeReport(in directory: URL) throws {
        try write(summary(), name: "summary.json", in: directory)
        try write(task(), name: "results-git-leak-recovery.json", in: directory)
        try write(["schema_version": 4, "max_attempts": 2, "attempt_policy": "stop_on_pass",
                   "profile_hash": profileHash, "tier": "smoke", "suite": suite],
                  name: "run-meta.json", in: directory)
    }

    func testCommandsMatchOfficialSuiteTiersAndAttemptBudget() throws {
        for tier in MiniToolBenchTier.allCases {
            for attempts in 1...2 {
                var value = configuration()
                value.tier = tier; value.attempts = attempts
                let command = try MiniToolBenchCommandBuilder.command(value)
                XCTAssertTrue(command.contains("'--suite=core19'"))
                XCTAssertTrue(command.contains("'--tier=\(tier.rawValue)'"))
                XCTAssertTrue(command.contains("'--attempts=\(attempts)'"))
                XCTAssertTrue(command.contains("'--concurrency=1'"))
                XCTAssertTrue(command.contains("'terminal_bench.py' 'doctor'"))
                XCTAssertTrue(command.contains("'terminal_bench.py' 'run'"))
                XCTAssertTrue(command.contains(MiniToolBenchCommandBuilder.pinnedRevision))
                XCTAssertFalse(command.contains("--context-length"))
                XCTAssertFalse(command.contains("--max-tokens"))
                XCTAssertFalse(command.contains("--skip-endpoint-check"))
            }
        }
        XCTAssertEqual(MiniToolBenchCommandBuilder.pinnedRevision,
                       "4a84b3dad49750a2db9f2e96d23a9bd8dafe7b66")
        XCTAssertTrue(MiniToolBenchCommandBuilder.setupCommand.contains("checkout --detach"))
    }

    func testExplicitModelAndContextAreSharedWithDoctor() throws {
        var value = configuration()
        value.modelID = "/models/Flash Q2.gguf"
        value.contextLength = "131072"
        value.quant = "IQ2XXS-w2Q2K"
        value.inferenceProfile = "DSpark"
        let command = try MiniToolBenchCommandBuilder.command(value)
        XCTAssertEqual(command.components(separatedBy: "'--model=/models/Flash Q2.gguf'").count - 1, 2)
        XCTAssertEqual(command.components(separatedBy: "'--context-length=131072'").count - 1, 2)
        XCTAssertEqual(command.components(separatedBy: "'--quant=IQ2XXS-w2Q2K'").count - 1, 1)
        XCTAssertEqual(command.components(separatedBy: "'--inference-profile=DSpark'").count - 1, 1)
    }

    func testLegacySuiteRequiresExplicitSelection() throws {
        var value = configuration()
        XCTAssertFalse(try MiniToolBenchCommandBuilder.command(value).contains("legacy-mini20"))
        value.suite = .legacyMini20
        XCTAssertTrue(try MiniToolBenchCommandBuilder.command(value).contains("'--suite=legacy-mini20'"))
    }

    func testCustomPythonExecutableIsQuotedAndUsedForEveryInvocation() throws {
        var value = configuration()
        value.pythonExecutable = "/opt/Python 3.14'stable/$(printf literal)/python3"
        let quoted = MiniToolBenchCommandBuilder.shellQuote(value.pythonExecutable)
        let command = try MiniToolBenchCommandBuilder.command(value)
        let pythonLines = command.split(separator: "\n").filter { $0.hasPrefix(quoted + " ") }
        XCTAssertEqual(pythonLines.count, 3)
        XCTAssertTrue(pythonLines[0].hasPrefix(quoted + " '-c' "))
        XCTAssertTrue(pythonLines[1].hasPrefix(quoted + " 'terminal_bench.py' 'doctor' "))
        XCTAssertTrue(pythonLines[2].hasPrefix(quoted + " 'terminal_bench.py' 'run' "))
        XCTAssertFalse(command.contains("'python3' 'terminal_bench.py'"))
        let encoded = try JSONEncoder().encode(value)
        XCTAssertEqual(try JSONDecoder().decode(MiniToolBenchConfiguration.self, from: encoded), value)
    }

    func testLegacyPreferencesKeepSelectionsWhenPythonFieldIsMissing() throws {
        var previous = configuration()
        previous.runnerDirectory = "/home/bench/saved runner"
        previous.endpoint = "https://example.test:443/v1"
        previous.suite = .legacyMini20
        previous.tier = .smoke
        previous.attempts = 1
        previous.quant = "IQ2XXS-w2Q2K"
        previous.inferenceProfile = "DSpark"
        previous.modelID = "/models/saved.gguf"
        previous.contextLength = "131072"
        var object = try JSONSerialization.jsonObject(with: JSONEncoder().encode(previous)) as! [String: Any]
        object.removeValue(forKey: "pythonExecutable")
        let restored = try JSONDecoder().decode(MiniToolBenchConfiguration.self,
                                                from: JSONSerialization.data(withJSONObject: object))
        XCTAssertEqual(restored.pythonExecutable, "python3")
        XCTAssertEqual(restored, previous)
        XCTAssertEqual(try JSONDecoder().decode(MiniToolBenchConfiguration.self, from: Data("{}".utf8)),
                       MiniToolBenchConfiguration())
    }

    func testPythonAndLinuxChecksPrecedeDoctorAndRejectEmptyInterpreter() throws {
        let command = try MiniToolBenchCommandBuilder.command(configuration())
        let lines = command.split(separator: "\n")
        let check = lines.firstIndex { $0.hasPrefix("'python3' '-c' ") }
        let doctor = lines.firstIndex { $0.contains("'terminal_bench.py' 'doctor'") }
        XCTAssertTrue(check != nil && doctor != nil && check! < doctor!)
        XCTAssertTrue(command.contains("sys.version_info < (3, 11)"))
        XCTAssertTrue(command.contains("sys.platform != "))
        XCTAssertTrue(command.contains("linux"))
        XCTAssertTrue(command.contains("shell macOS"))
        for interpreter in ["", "   ", "python3\nnext command", "python3\0tail"] {
            var value = configuration(); value.pythonExecutable = interpreter
            XCTAssertThrowsError(try MiniToolBenchCommandBuilder.command(value), interpreter)
        }
    }

    func testShellQuotingPreservesMetacharactersWithoutEvaluatingThem() throws {
        let values = ["plain", "", "single'quote", "spaces and tabs", "$HOME", "$(printf injected)",
                      "`printf injected`", "; printf injected", "--option", "\\backslash", "città 🪐"]
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/sh")
        // The shell runs only printf; every prospective expansion is a literal argument.
        process.arguments = ["-c", "printf '%s\\0' " + values.map(MiniToolBenchCommandBuilder.shellQuote).joined(separator: " ")]
        let pipe = Pipe(); process.standardOutput = pipe
        try process.run()
        let output = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus, 0)
        // Process may normalize Unicode on macOS; compare Swift strings, whose
        // equality preserves canonical equivalence while retaining every NUL.
        XCTAssertEqual(String(decoding: output, as: UTF8.self), values.joined(separator: "\0") + "\0")
    }

    func testCommandTreatsOptionLookingValuesAndDirectoriesAsData() throws {
        var value = configuration()
        value.runnerDirectory = "-runner with 'quotes'"
        value.modelName = "--help"
        value.quant = "$(printf literal)"
        let command = try MiniToolBenchCommandBuilder.command(value)
        XCTAssertTrue(command.hasPrefix("cd './-runner with '\\''quotes'\\''' &&"))
        XCTAssertTrue(command.contains("'--model-name=--help'"))
        XCTAssertTrue(command.contains("'--quant=$(printf literal)'"))
        value.runnerDirectory = "~/a'b"
        XCTAssertTrue(try MiniToolBenchCommandBuilder.command(value).hasPrefix("cd \"$HOME\"/'a'\\''b' &&"))
    }

    func testInvalidInputCannotGenerateRunnableCommands() throws {
        let invalidEndpoints = ["file:///v1", "http://example.com", "http://user:secret@example.com/v1",
                                "http://example.com/v1?key=secret", "http://example.com/v1#fragment",
                                "http://0.0.0.0:8080/v1"]
        for endpoint in invalidEndpoints {
            var value = configuration(); value.endpoint = endpoint
            XCTAssertThrowsError(try MiniToolBenchCommandBuilder.command(value), endpoint)
        }
        for context in ["0", "-1", "1.5", "NaN", "999999999999999999999999999"] {
            var value = configuration(); value.contextLength = context
            XCTAssertThrowsError(try MiniToolBenchCommandBuilder.command(value), context)
        }
        for attempts in [0, 3] {
            var value = configuration(); value.attempts = attempts
            XCTAssertThrowsError(try MiniToolBenchCommandBuilder.command(value))
        }
        for model in ["", " \n", "Flash.gguf", "model\nnext command", "model\0tail"] {
            var value = configuration(); value.modelName = model
            XCTAssertThrowsError(try MiniToolBenchCommandBuilder.command(value))
        }
    }

    func testSchemaFourImportPreservesAttemptsAndOfficialAggregate() throws {
        let folder = try directory(); try completeReport(in: folder)
        let report = try MiniToolBenchReport.load(from: folder)
        XCTAssertEqual(report.summary.schemaVersion, 4)
        XCTAssertEqual(report.summary.suite?.manifestHash, manifestHash)
        XCTAssertNil(report.summary.suite?.name)
        XCTAssertEqual(report.summary.passedTasks, 1)
        XCTAssertEqual(report.summary.passRate, 1)
        XCTAssertEqual(report.summary.tokens?.cached, 20)
        XCTAssertEqual(report.attemptBudget, 2)
        XCTAssertEqual(report.tasks.first?.attempts?.map(\.passed), [false, true])
        XCTAssertEqual(report.tasks.first?.attempts?.map(\.attempt), [1, 2])
        XCTAssertTrue(report.warnings.contains { $0.contains("non certifica") })
    }

    func testMissingDetailsAreReportedWithoutInventingAttemptBudget() throws {
        let folder = try directory()
        try write(summary(), name: "summary.json", in: folder)
        let report = try MiniToolBenchReport.load(from: folder.appendingPathComponent("summary.json"))
        XCTAssertEqual(report.summary.totalTasks, 1)
        XCTAssertTrue(report.tasks.isEmpty)
        XCTAssertNil(report.attemptBudget)
        XCTAssertTrue(report.warnings.contains { $0.contains("Mancano 1") })
    }

    func testExceptionFailuresStillCountInTheDenominator() throws {
        let folder = try directory()
        try write(summary(passed: 0), name: "summary.json", in: folder)
        var row = task(passed: false); row["completed"] = false
        try write(row, name: "results-git-leak-recovery.json", in: folder)
        let report = try MiniToolBenchReport.load(from: folder)
        XCTAssertEqual(report.summary.totalTasks, 1)
        XCTAssertEqual(report.tasks.count, 1)
        XCTAssertEqual(report.tasks.first?.completed, false)
        XCTAssertEqual(report.summary.passRate, 0)
    }

    func testMalformedSummariesAreRejected() throws {
        let folder = try directory()
        let mutations: [(String, Any)] = [("schema_version", 5), ("total_tasks", -1),
                                        ("passed_tasks", 2), ("pass_rate", 0.5),
                                        ("results", []), ("total_tasks", 1_001)]
        for (key, value) in mutations {
            var document = summary(); document[key] = value
            try write(document, name: "summary.json", in: folder)
            XCTAssertThrowsError(try MiniToolBenchReport.load(from: folder), key)
        }
        try Data("not JSON".utf8).write(to: folder.appendingPathComponent("summary.json"))
        XCTAssertThrowsError(try MiniToolBenchReport.load(from: folder))
    }

    func testDuplicateReferencesAndDuplicateTaskIDsAreRejected() throws {
        let folder = try directory()
        try write(summary(["results-a.json", "results-a.json"], passed: 2), name: "summary.json", in: folder)
        XCTAssertThrowsError(try MiniToolBenchReport.load(from: folder))
        try write(summary(["results-a.json", "results-b.json"], passed: 2), name: "summary.json", in: folder)
        try write(task(), name: "results-a.json", in: folder)
        try write(task(), name: "results-b.json", in: folder)
        XCTAssertThrowsError(try MiniToolBenchReport.load(from: folder))
    }

    func testMixedModelSuiteAndProfileAreRejected() throws {
        let folder = try directory()
        try write(summary(), name: "summary.json", in: folder)
        let mutations: [(String, Any)] = [("model", ["id": "other-model"]),
                                        ("suite", ["id": "core19", "manifest_hash": String(repeating: "c", count: 64)]),
                                        ("profile_hash", String(repeating: "d", count: 64))]
        for (key, value) in mutations {
            var row = task(); row[key] = value
            try write(row, name: "results-git-leak-recovery.json", in: folder)
            XCTAssertThrowsError(try MiniToolBenchReport.load(from: folder), key)
        }
    }

    func testTaskOutcomesMustMatchTheSummary() throws {
        let folder = try directory()
        try write(summary(), name: "summary.json", in: folder)
        try write(task(passed: false), name: "results-git-leak-recovery.json", in: folder)
        XCTAssertThrowsError(try MiniToolBenchReport.load(from: folder))
    }

    func testRunMetadataFromAnotherProfileCannotSupplyAttemptBudget() throws {
        let folder = try directory(); try completeReport(in: folder)
        try write(["max_attempts": 2, "profile_hash": "different-profile"], name: "run-meta.json", in: folder)
        let report = try MiniToolBenchReport.load(from: folder)
        XCTAssertNil(report.attemptBudget)
        XCTAssertTrue(report.warnings.contains { $0.contains("budget") })
    }

    func testArtifactTraversalIsRejectedEvenWhenTargetIsMissing() throws {
        let folder = try directory()
        for name in ["../outside.json", "/tmp/outside.json", "sub/result.json", "sub\\result.json", "..", "a\n.json"] {
            try write(summary([name]), name: "summary.json", in: folder)
            XCTAssertThrowsError(try MiniToolBenchReport.load(from: folder), name)
        }
    }

    func testTaskAndMetadataSymlinksCannotEscapeTheSelectedDirectory() throws {
        let folder = try directory(), outside = try directory()
        try write(summary(), name: "summary.json", in: folder)
        try write(task(), name: "outside.json", in: outside)
        let result = folder.appendingPathComponent("results-git-leak-recovery.json")
        try FileManager.default.createSymbolicLink(at: result, withDestinationURL: outside.appendingPathComponent("outside.json"))
        XCTAssertThrowsError(try MiniToolBenchReport.load(from: folder))
        try FileManager.default.removeItem(at: result)
        try write(task(), name: result.lastPathComponent, in: folder)
        try write(["max_attempts": 2, "profile_hash": profileHash], name: "meta.json", in: outside)
        try FileManager.default.createSymbolicLink(at: folder.appendingPathComponent("run-meta.json"),
                                                 withDestinationURL: outside.appendingPathComponent("meta.json"))
        XCTAssertThrowsError(try MiniToolBenchReport.load(from: folder))
    }

    func testTranscriptLinksRemainInsideTheSelectedDirectory() throws {
        let folder = try directory(), outside = try directory()
        try completeReport(in: folder)
        let name = "transcript-git-leak-recovery-attempt2.json"
        try write(["steps": []], name: name, in: folder)
        var report = try MiniToolBenchReport.load(from: folder)
        XCTAssertEqual(report.transcriptURL(for: report.tasks[0])?.lastPathComponent, name)
        try FileManager.default.removeItem(at: folder.appendingPathComponent(name))
        try write(["steps": []], name: name, in: outside)
        try FileManager.default.createSymbolicLink(at: folder.appendingPathComponent(name),
                                                 withDestinationURL: outside.appendingPathComponent(name))
        XCTAssertNil(report.transcriptURL(for: report.tasks[0]))
        var row = task(); row["transcript"] = "../outside.json"
        try write(row, name: "results-git-leak-recovery.json", in: folder)
        report = try MiniToolBenchReport.load(from: folder)
        XCTAssertNil(report.transcriptURL(for: report.tasks[0]))
    }

    func testTranscriptCannotOpenExecutableCommandsOrDirectories() throws {
        let folder = try directory(); try completeReport(in: folder)
        for name in ["transcript.command", "transcript.app", "transcript-directory.json"] {
            var row = task(); row["transcript"] = name
            try write(row, name: "results-git-leak-recovery.json", in: folder)
            let artifact = folder.appendingPathComponent(name)
            if name.hasSuffix(".command") {
                try Data("#!/bin/sh\nexit 0\n".utf8).write(to: artifact)
            } else {
                try FileManager.default.createDirectory(at: artifact, withIntermediateDirectories: false)
            }
            let report = try MiniToolBenchReport.load(from: folder)
            XCTAssertNil(report.transcriptURL(for: report.tasks[0]), name)
        }
    }
}
