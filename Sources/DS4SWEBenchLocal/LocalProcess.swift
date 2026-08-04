import Foundation

struct LocalProcessResult: Sendable {
    let exitCode: Int32
    let timedOut: Bool
    let output: String
    let duration: Double
}

enum LocalProcess {
    static func run(_ executable: String, _ arguments: [String], directory: URL,
                    timeout: TimeInterval, input: Data? = nil) throws -> LocalProcessResult {
        let fm = FileManager.default
        let capture = directory.appendingPathComponent(".ds4-process-\(UUID().uuidString).log")
        fm.createFile(atPath: capture.path, contents: nil)
        defer { try? fm.removeItem(at: capture) }
        let outputHandle = try FileHandle(forWritingTo: capture)
        defer { try? outputHandle.close() }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        process.currentDirectoryURL = directory
        process.standardOutput = outputHandle
        process.standardError = outputHandle
        if let input {
            let pipe = Pipe(); pipe.fileHandleForWriting.write(input); try? pipe.fileHandleForWriting.close()
            process.standardInput = pipe
        }

        let start = Date()
        try process.run()
        let deadline = Date().addingTimeInterval(timeout)
        while process.isRunning && Date() < deadline { Thread.sleep(forTimeInterval: 0.05) }
        let timedOut = process.isRunning
        if timedOut { process.terminate() }
        process.waitUntilExit()
        try outputHandle.synchronize()
        let data = (try? Data(contentsOf: capture)) ?? Data()
        return LocalProcessResult(exitCode: process.terminationStatus, timedOut: timedOut,
                                  output: String(decoding: data, as: UTF8.self),
                                  duration: Date().timeIntervalSince(start))
    }
}
