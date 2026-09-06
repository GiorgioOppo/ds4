import Foundation
import XCTest
@testable import DwarfStar

final class MiniToolBenchRunnerTests: XCTestCase {
    func testPodmanMachineInventoryUsesOfficialFieldNames() throws {
        let inventory = Data(#"[{"Name":"podman-machine-default","Running":true,"Default":true,"VMType":"applehv","CPUs":4,"Memory":"4GiB"},{"Name":"dwarfstar-bench","Running":false}]"#.utf8)
        let machines = try JSONDecoder().decode([MiniToolBenchMachine].self, from: inventory)
        XCTAssertEqual(machines.map(\.name), ["podman-machine-default", "dwarfstar-bench"])
        XCTAssertEqual(machines.map(\.running), [true, false])
    }

    func testPreferredMachineReusesDedicatedRunningMachine() throws {
        let machines = [MiniToolBenchMachine(name: "other-running", running: true),
                        MiniToolBenchMachine(name: "dwarfstar-bench", running: true),
                        MiniToolBenchMachine(name: "other-stopped", running: false)]
        XCTAssertEqual(MiniToolBenchMachine.preferred(in: machines)?.name, "dwarfstar-bench")
    }

    func testPreferredMachineUsesAnExistingRunningOrStoppedMachineBeforeCreation() throws {
        let stopped = MiniToolBenchMachine(name: "existing-stopped", running: false)
        let running = MiniToolBenchMachine(name: "existing-running", running: true)
        XCTAssertEqual(MiniToolBenchMachine.preferred(in: [stopped, running])?.name, running.name)
        XCTAssertEqual(MiniToolBenchMachine.preferred(in: [stopped])?.name, stopped.name)
        XCTAssertNil(MiniToolBenchMachine.preferred(in: []))
    }

    func testMalformedInventoryCannotImplyAStoppedMachine() throws {
        for document in [#"[{"Name":"vm"}]"#, #"[{"Name":"vm","Running":"false"}]"#,
                         #"[{"Running":true}]"#, #"{"Name":"vm","Running":true}"#] {
            XCTAssertThrowsError(try JSONDecoder().decode([MiniToolBenchMachine].self, from: Data(document.utf8)))
        }
    }

    @MainActor
    func testRemoteShellReceivesBootstrapAsOneLiteralArgument() throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/sh")
        // Tokenize the command into positional arguments. Do not invoke Python,
        // Podman, the bootstrap, or any of its installation/benchmark actions.
        process.arguments = ["-c", "set -- " + MiniToolBenchRunner.remoteCommand + "; printf '%s\\0' \"$@\""]
        let pipe = Pipe(); process.standardOutput = pipe
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus, 0)
        let arguments = String(decoding: data, as: UTF8.self).components(separatedBy: "\0")
        XCTAssertEqual(arguments.count, 5)
        XCTAssertEqual(Array(arguments.prefix(3)), ["python3", "-u", "-c"])
        XCTAssertEqual(arguments[3], MiniToolBenchBootstrap.script)
        XCTAssertEqual(arguments[4], "")
    }

    func testJSONLinesEventsPreserveUnicodeAndEscapedLineBreaks() throws {
        let document = #"{"event":"phase","message":"Preparazione dell’ambiente Linux…\nSeconda riga","extra":{"ignored":true}}"#
        let event = try JSONDecoder().decode(MiniToolBenchRunEvent.self, from: Data(document.utf8))
        XCTAssertEqual(event.event, "phase")
        XCTAssertEqual(event.message, "Preparazione dell’ambiente Linux…\nSeconda riga")
        XCTAssertNil(event.directory)
    }

    func testResultAndCompletionEventsDoNotRequireHumanMessages() throws {
        let result = try JSONDecoder().decode(MiniToolBenchRunEvent.self,
            from: Data(#"{"event":"result","directory":"/Users/example/Library/Application Support/DwarfStar/MiniToolBench/runs/dwarfstar-id","summary":{"schema_version":4,"total_tasks":1},"partial":false}"#.utf8))
        XCTAssertEqual(result.event, "result")
        XCTAssertTrue(result.directory?.hasSuffix("/dwarfstar-id") == true)
        XCTAssertNil(result.message)
        let completed = try JSONDecoder().decode(MiniToolBenchRunEvent.self,
            from: Data(#"{"event":"completed","action":"cancel","status":"requested","run_id":"dwarfstar-id"}"#.utf8))
        XCTAssertEqual(completed.event, "completed")
        XCTAssertNil(completed.message)
        XCTAssertNil(completed.directory)
    }

    func testMalformedEventsAreRejectedForPlainLogFallback() throws {
        for document in ["ordinary Podman progress", #"{"message":"missing event"}"#,
                         #"{"event":1}"#, #"{"event":"result","directory":42}"#,
                         #"{"event":"log","message":{"nested":"object"}}"#] {
            XCTAssertThrowsError(try JSONDecoder().decode(MiniToolBenchRunEvent.self, from: Data(document.utf8)))
        }
        let event = try JSONDecoder().decode(MiniToolBenchRunEvent.self,
            from: Data(#"{"event":"future-event","message":null}"#.utf8))
        XCTAssertEqual(event.event, "future-event")
        XCTAssertNil(event.message)
    }
}
