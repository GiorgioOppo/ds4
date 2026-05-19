import XCTest
@testable import DeepSeekTools

/// Smoke tests for the Unix-toolbox wrappers that spawn a subprocess
/// (sed, awk, file, tar/gzip/zip/unzip, ps, lsof, kill, jq, git_*).
/// Each test guards against the binary being absent via
/// `XCTSkipUnless`, so a clean macOS or Linux CI without Homebrew
/// won't spuriously fail.
final class UnixBinaryWrappersTests: XCTestCase {

    private func makeTempRoot() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("UnixWrappersTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func ctx(_ root: URL) -> ToolContext {
        ToolContext(rootDirectory: root,
                    permission: AutoPermissionDelegate(allowDangerous: true))
    }

    private func write(_ content: String, to rel: String, in root: URL) throws {
        try content.write(to: root.appendingPathComponent(rel),
                          atomically: true, encoding: .utf8)
    }

    // MARK: - sed / awk / file

    func testSedSubstitution() async throws {
        try XCTSkipUnless(FileManager.default.isExecutableFile(atPath: "/usr/bin/sed"))
        let root = try makeTempRoot()
        try write("hello world\n", to: "s.txt", in: root)
        let out = try await SedTool().run(
            input: ["pattern": "world", "replacement": "swift", "path": "s.txt"],
            context: ctx(root))
        XCTAssertTrue(out.output.contains("hello swift"))
    }

    func testAwkSumColumn() async throws {
        try XCTSkipUnless(FileManager.default.isExecutableFile(atPath: "/usr/bin/awk"))
        let root = try makeTempRoot()
        try write("1\n2\n3\n", to: "n.txt", in: root)
        let out = try await AwkTool().run(
            input: ["program": "{s+=$1} END {print s}", "path": "n.txt"],
            context: ctx(root))
        XCTAssertTrue(out.output.contains("6"))
    }

    func testFileDetectsText() async throws {
        try XCTSkipUnless(FileManager.default.isExecutableFile(atPath: "/usr/bin/file"))
        let root = try makeTempRoot()
        try write("hello", to: "t.txt", in: root)
        let out = try await FileTool().run(
            input: ["path": "t.txt"], context: ctx(root))
        // /usr/bin/file emits "ASCII text" / "Unicode text" / "UTF-8 text"
        // depending on platform — match the common substring.
        XCTAssertTrue(out.output.lowercased().contains("text"))
    }

    // MARK: - Archive

    func testTarListAfterCreate() async throws {
        try XCTSkipUnless(FileManager.default.isExecutableFile(atPath: "/usr/bin/tar"))
        let root = try makeTempRoot()
        try write("payload", to: "a.txt", in: root)
        // Build a tar via /usr/bin/tar directly — TarTool only supports
        // list/extract in v1, not create. We exercise the list path.
        let archive = root.appendingPathComponent("a.tar")
        let task = Process()
        task.launchPath = "/usr/bin/tar"
        task.arguments = ["-cf", archive.path, "-C", root.path, "a.txt"]
        try task.run()
        task.waitUntilExit()
        XCTAssertEqual(task.terminationStatus, 0)

        let out = try await TarTool().run(
            input: ["operation": "list", "archive": "a.tar"],
            context: ctx(root))
        XCTAssertTrue(out.output.contains("a.txt"))
    }

    func testGzipRoundTrip() async throws {
        try XCTSkipUnless(FileManager.default.isExecutableFile(atPath: "/usr/bin/gzip"))
        try XCTSkipUnless(FileManager.default.isExecutableFile(atPath: "/usr/bin/gunzip"))
        let root = try makeTempRoot()
        try write("payload", to: "p.txt", in: root)
        _ = try await GzipTool().run(
            input: ["path": "p.txt"], context: ctx(root))
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: root.appendingPathComponent("p.txt.gz").path))
        _ = try await GunzipTool().run(
            input: ["path": "p.txt.gz"], context: ctx(root))
        let restored = try String(contentsOf: root.appendingPathComponent("p.txt"))
        XCTAssertEqual(restored, "payload")
    }

    // MARK: - ps / lsof / kill

    func testPsRuns() async throws {
        try XCTSkipUnless(FileManager.default.isExecutableFile(atPath: "/bin/ps"))
        let root = try makeTempRoot()
        let out = try await PsTool().run(input: [:], context: ctx(root))
        // PS prints a header line — exit 0 + non-empty body is enough.
        XCTAssertFalse(out.isError)
        XCTAssertFalse(out.output.isEmpty)
    }

    func testKillRejectsPid1() async throws {
        let root = try makeTempRoot()
        do {
            _ = try await KillTool().run(
                input: ["pid": 1], context: ctx(root))
            XCTFail("expected permissionDenied for PID 1")
        } catch let err as ToolError {
            if case .invalidInput = err { return }
            if case .permissionDenied = err { return }
            XCTFail("expected denied for PID 1, got \(err)")
        }
    }

    func testKillRejectsSelfPid() async throws {
        let root = try makeTempRoot()
        do {
            _ = try await KillTool().run(
                input: ["pid": Int(getpid())], context: ctx(root))
            XCTFail("expected permissionDenied for self PID")
        } catch let err as ToolError {
            if case .permissionDenied = err { return }
            XCTFail("expected .permissionDenied, got \(err)")
        }
    }

    // MARK: - jq (optional)

    func testJqExtractsField() async throws {
        let candidates = ["/opt/homebrew/bin/jq", "/usr/local/bin/jq", "/opt/local/bin/jq"]
        try XCTSkipUnless(candidates.contains { FileManager.default.isExecutableFile(atPath: $0) })
        let root = try makeTempRoot()
        try write(#"{"name":"swift"}"#, to: "j.json", in: root)
        let out = try await JqTool().run(
            input: ["filter": ".name", "path": "j.json", "raw": true],
            context: ctx(root))
        XCTAssertTrue(out.output.contains("swift"))
    }

    // MARK: - git_* (require a repo)

    func testGitStatusOnRepo() async throws {
        try XCTSkipUnless(FileManager.default.isExecutableFile(atPath: "/usr/bin/git"))
        let root = try makeTempRoot()
        // Initialize a small repo so git_status has something to inspect.
        for args in [["init"], ["config", "user.email", "test@example.com"],
                     ["config", "user.name", "test"]] {
            let t = Process()
            t.launchPath = "/usr/bin/git"
            t.currentDirectoryURL = root
            t.arguments = args
            try t.run(); t.waitUntilExit()
        }
        try write("hi", to: "a.txt", in: root)
        let out = try await GitStatusTool().run(input: [:], context: ctx(root))
        XCTAssertTrue(out.output.contains("a.txt"))
    }
}
