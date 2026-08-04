import Foundation

struct CLIOptions {
    var task = "", prediction = "", workspace = "", report = ""
    var expectedResults = "", actualResults = ""
    var timeout: TimeInterval = 1_800
    var allowUnsafe = false
    var patchOnly = false
}

func parseArguments() throws -> CLIOptions {
    var o = CLIOptions(), args = Array(CommandLine.arguments.dropFirst()), i = 0
    func value(_ flag: String) throws -> String {
        guard i + 1 < args.count else { throw SWELocalError.invalid("Valore mancante per \(flag)") }
        i += 1; return args[i]
    }
    while i < args.count {
        switch args[i] {
        case "--task": o.task = try value(args[i])
        case "--prediction": o.prediction = try value(args[i])
        case "--workspace": o.workspace = try value(args[i])
        case "--report": o.report = try value(args[i])
        case "--expected-results": o.expectedResults = try value(args[i])
        case "--actual-results": o.actualResults = try value(args[i])
        case "--timeout": o.timeout = TimeInterval(try value(args[i])) ?? o.timeout
        case "--allow-unsafe-host-execution": o.allowUnsafe = true
        case "--patch-only": o.patchOnly = true
        case "--help", "-h":
            print("""
            ds4-swe-local --task task.json --prediction prediction.json \\
              --workspace DIR --report report.json \\
              --allow-unsafe-host-execution [--timeout 1800]

            WARNING: repository tests execute untrusted code directly on this Mac.

            ds4-swe-local --expected-results expected.json \
              --actual-results actual.json --report comparison.json
            """); exit(0)
        default: throw SWELocalError.invalid("Argomento sconosciuto: \(args[i])")
        }
        i += 1
    }
    let comparisonMode = !o.expectedResults.isEmpty || !o.actualResults.isEmpty
    if comparisonMode {
        guard !o.expectedResults.isEmpty, !o.actualResults.isEmpty, !o.report.isEmpty else {
            throw SWELocalError.invalid("Il confronto richiede --expected-results, --actual-results e --report")
        }
        return o
    }
    guard !o.task.isEmpty, !o.prediction.isEmpty, !o.workspace.isEmpty,
          !o.report.isEmpty else { throw SWELocalError.invalid("Argomenti obbligatori mancanti; usa --help") }
    guard o.patchOnly || o.allowUnsafe else { throw SWELocalError.invalid("Esecuzione rifiutata: aggiungi --allow-unsafe-host-execution dopo aver compreso il rischio") }
    return o
}

do {
    let options = try parseArguments()
    let decoder = JSONDecoder()
    if !options.expectedResults.isEmpty {
        let expected = try decoder.decode(JSONTestDocument.self,
                                          from: Data(contentsOf: URL(fileURLWithPath: options.expectedResults)))
        let actual = try decoder.decode(JSONTestDocument.self,
                                        from: Data(contentsOf: URL(fileURLWithPath: options.actualResults)))
        let comparison = try JSONTestComparator.compare(expected: expected, actual: actual)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(comparison).write(to: URL(fileURLWithPath: options.report), options: .atomic)
        print("tests_passed=\(comparison.passed) passed=\(comparison.passedCases.count) failed=\(comparison.failedCases.count) report=\(options.report)")
        exit(comparison.passed ? 0 : 1)
    }
    let task = try decoder.decode(SWETask.self, from: Data(contentsOf: URL(fileURLWithPath: options.task)))
    let prediction = try decoder.decode(SWEPrediction.self, from: Data(contentsOf: URL(fileURLWithPath: options.prediction)))
    let (grade, log) = try SWELocalRunner(workspace: URL(fileURLWithPath: options.workspace, isDirectory: true),
                                          timeout: options.timeout)
        .run(task: task, prediction: prediction, patchOnly: options.patchOnly)
    let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(grade).write(to: URL(fileURLWithPath: options.report), options: .atomic)
    print(log)
    if grade.verificationOnly {
        print("patch_applied=\(grade.patchApplied) verification_only=true report=\(options.report)")
        exit(grade.patchApplied ? 0 : 1)
    }
    print("resolved=\(grade.resolved) report=\(options.report)")
    exit(grade.resolved ? 0 : 1)
} catch {
    FileHandle.standardError.write(Data("errore: \(error.localizedDescription)\n".utf8))
    exit(2)
}
