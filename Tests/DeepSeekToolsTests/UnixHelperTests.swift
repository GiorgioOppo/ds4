import XCTest
@testable import DeepSeekTools

/// Tests for `UnixBinary` — the shared subprocess runner used by every
/// Unix-toolbox wrapper. Covers the drain/cap/timeout/cancel/SIGKILL
/// paths against well-known stable binaries.
final class UnixHelperTests: XCTestCase {

    private func makeTempRoot() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("UnixHelperTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func ctx(_ root: URL,
                     env: [String: String]? = nil,
                     isCancelled: @escaping @Sendable () -> Bool = { false }) -> ToolContext
    {
        ToolContext(rootDirectory: root,
                    permission: AutoPermissionDelegate(allowDangerous: true),
                    environment: env,
                    isCancelled: isCancelled)
    }

    func testEchoRoundTrip() async throws {
        try XCTSkipUnless(FileManager.default.isExecutableFile(atPath: "/bin/echo"))
        let root = try makeTempRoot()
        let out = try await UnixBinary.runBinary(
            launchPath: "/bin/echo",
            arguments: ["hello", "world"],
            context: ctx(root))
        XCTAssertFalse(out.isError)
        XCTAssertTrue(out.output.contains("hello world"))
        XCTAssertEqual(out.metadata["exit"], "0")
    }

    func testFalseReturnsNonZeroExit() async throws {
        try XCTSkipUnless(FileManager.default.isExecutableFile(atPath: "/usr/bin/false"))
        let root = try makeTempRoot()
        let out = try await UnixBinary.runBinary(
            launchPath: "/usr/bin/false",
            arguments: [],
            context: ctx(root))
        XCTAssertTrue(out.isError)
        XCTAssertEqual(out.metadata["exit"], "1")
    }

    func testMissingBinaryThrowsNotFound() async throws {
        let root = try makeTempRoot()
        do {
            _ = try await UnixBinary.runBinary(
                launchPath: "/no/such/binary/exists",
                arguments: [],
                context: ctx(root))
            XCTFail("expected ToolError.notFound")
        } catch let err as ToolError {
            if case .notFound = err { return }
            XCTFail("expected .notFound, got \(err)")
        }
    }

    func testTimeoutFires() async throws {
        try XCTSkipUnless(FileManager.default.isExecutableFile(atPath: "/bin/sleep"))
        let root = try makeTempRoot()
        do {
            _ = try await UnixBinary.runBinary(
                launchPath: "/bin/sleep",
                arguments: ["10"],
                context: ctx(root),
                timeout: 1)
            XCTFail("expected timeout")
        } catch let err as ToolError {
            if case .timeout = err { return }
            XCTFail("expected .timeout, got \(err)")
        }
    }

    func testCancellationStopsLongRunner() async throws {
        try XCTSkipUnless(FileManager.default.isExecutableFile(atPath: "/bin/sleep"))
        let root = try makeTempRoot()
        let cancelled = CancelBox()
        Task {
            try? await Task.sleep(nanoseconds: 200_000_000)
            cancelled.flag = true
        }
        do {
            _ = try await UnixBinary.runBinary(
                launchPath: "/bin/sleep",
                arguments: ["30"],
                context: ctx(root, isCancelled: { cancelled.flag }),
                timeout: 30)
            XCTFail("expected cancellation to surface as timeout")
        } catch let err as ToolError {
            if case .timeout = err { return }
            XCTFail("expected .timeout from cancellation, got \(err)")
        }
    }

    func testOutputCapTruncates() async throws {
        try XCTSkipUnless(FileManager.default.isExecutableFile(atPath: "/usr/bin/yes"))
        let root = try makeTempRoot()
        // `yes` would loop forever; cap at 4 KB and 1 s timeout. The
        // child gets SIGTERM via timeout but the drained data should
        // already be at the cap.
        do {
            let out = try await UnixBinary.runBinary(
                launchPath: "/usr/bin/yes",
                arguments: ["x"],
                context: ctx(root),
                timeout: 1,
                outputCap: 4 * 1024)
            // Either timeout fired (our path) or output was capped — both
            // exit conditions prove the helper didn't block on the pipe.
            XCTAssertLessThanOrEqual(out.output.utf8.count, 6 * 1024)
        } catch let err as ToolError {
            if case .timeout = err { return }
            XCTFail("expected timeout or capped output, got \(err)")
        }
    }

    func testResolveBinaryFindsFirstExisting() throws {
        let candidates = ["/no/such/path", "/bin/sh", "/usr/bin/true"]
        let resolved = UnixBinary.resolveBinary(candidates: candidates)
        XCTAssertEqual(resolved, "/bin/sh")
    }

    func testResolveBinaryReturnsNilOnMiss() throws {
        let resolved = UnixBinary.resolveBinary(
            candidates: ["/no/such/path", "/also/missing"])
        XCTAssertNil(resolved)
    }
}

/// Tiny mutable box used to flip the cancellation flag across the
/// closure boundary. Avoids capturing a `var` (which would require an
/// inout / reference type) and keeps the test code terse.
private final class CancelBox: @unchecked Sendable {
    var flag: Bool = false
}
