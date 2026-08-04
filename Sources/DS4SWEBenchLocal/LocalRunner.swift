import Foundation

enum SWELocalError: LocalizedError {
    case invalid(String), command(String)
    var errorDescription: String? {
        switch self { case .invalid(let s), .command(let s): return s }
    }
}

struct SWELocalRunner {
    let workspace: URL
    let timeout: TimeInterval

    func run(task: SWETask, prediction: SWEPrediction, patchOnly: Bool) throws -> (SWEGrade, String) {
        guard task.instanceID == prediction.instanceID else {
            throw SWELocalError.invalid("Task e prediction hanno instance_id diversi")
        }
        let fm = FileManager.default
        try fm.createDirectory(at: workspace, withIntermediateDirectories: true)
        let repoDir = workspace.appendingPathComponent(task.instanceID, isDirectory: true)
        if !fm.fileExists(atPath: repoDir.path) {
            let repositoryURL = task.repo.contains("://") || task.repo.hasPrefix("/")
                ? task.repo : "https://github.com/\(task.repo).git"
            let clone = try LocalProcess.run("/usr/bin/git", ["clone", "--no-checkout", repositoryURL, repoDir.path],
                                             directory: workspace, timeout: timeout)
            guard clone.exitCode == 0 else { throw SWELocalError.command("git clone fallito:\n\(clone.output)") }
        }
        let checkout = try LocalProcess.run("/usr/bin/git", ["checkout", "--force", task.baseCommit],
                                            directory: repoDir, timeout: timeout)
        guard checkout.exitCode == 0 else { throw SWELocalError.command("checkout fallito:\n\(checkout.output)") }
        _ = try LocalProcess.run("/usr/bin/git", ["clean", "-fdx"], directory: repoDir, timeout: timeout)

        let patch = prediction.modelPatch ?? ""
        guard !patch.isEmpty else {
            return (SWEGrader.grade(task: task, statuses: [:], patchApplied: false,
                                    exitCode: 1, timedOut: false, duration: 0,
                                    verificationOnly: patchOnly), "Patch vuota\n")
        }
        let apply = try LocalProcess.run("/usr/bin/git", ["apply", "--verbose", "-"], directory: repoDir,
                                         timeout: timeout, input: Data(patch.utf8))
        guard apply.exitCode == 0 else {
            return (SWEGrader.grade(task: task, statuses: [:], patchApplied: false,
                                    exitCode: apply.exitCode, timedOut: apply.timedOut, duration: apply.duration,
                                    verificationOnly: patchOnly), apply.output)
        }
        if patchOnly {
            return (SWEGrader.grade(task: task, statuses: [:], patchApplied: true,
                                    exitCode: 0, timedOut: false, duration: apply.duration,
                                    verificationOnly: true),
                    "Patch applicata correttamente a \(task.baseCommit). Test non eseguiti.\n")
        }
        if !task.testPatch.isEmpty {
            let tests = try LocalProcess.run("/usr/bin/git", ["apply", "--verbose", "-"], directory: repoDir,
                                             timeout: timeout, input: Data(task.testPatch.utf8))
            guard tests.exitCode == 0 else { throw SWELocalError.command("test_patch non applicabile:\n\(tests.output)") }
        }

        let test = try LocalProcess.run("/usr/bin/swift", ["test"], directory: repoDir, timeout: timeout)
        let statuses = SwiftTestParser.parse(test.output, expected: task.failToPass + task.passToPass)
        let combinedLog = "=== swift test ===\n" + test.output
        return (SWEGrader.grade(task: task, statuses: statuses, patchApplied: true,
                                exitCode: test.exitCode, timedOut: test.timedOut, duration: test.duration), combinedLog)
    }
}
