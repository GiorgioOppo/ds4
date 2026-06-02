import XCTest
@testable import DeepSeekTools

/// Smoke tests for the pure-Swift Unix tools (Files / Text / Hash /
/// System / Mutating). Wrappers around system binaries live in
/// `UnixBinaryWrappersTests.swift`. The two split lets the wrapper
/// suite skip cleanly when binaries aren't installed without poisoning
/// coverage of the native tools.
final class UnixToolsTests: XCTestCase {

    private func makeTempRoot() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("UnixToolsTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func ctx(_ root: URL, mode: AgentMode = .build) -> ToolContext {
        ToolContext(rootDirectory: root,
                    mode: mode,
                    permission: AutoPermissionDelegate(allowDangerous: true))
    }

    private func write(_ content: String, to rel: String, in root: URL) throws {
        try content.write(to: root.appendingPathComponent(rel),
                          atomically: true, encoding: .utf8)
    }

    // MARK: - Files family

    func testLsListsEntries() async throws {
        let root = try makeTempRoot()
        try write("a", to: "alpha.txt", in: root)
        try write("b", to: "beta.txt", in: root)
        let out = try await LsTool().run(input: [:], context: ctx(root))
        XCTAssertTrue(out.output.contains("alpha.txt"))
        XCTAssertTrue(out.output.contains("beta.txt"))
    }

    func testLsLongFormatIncludesPerms() async throws {
        let root = try makeTempRoot()
        try write("x", to: "f.txt", in: root)
        let out = try await LsTool().run(input: ["long": true], context: ctx(root))
        XCTAssertTrue(out.output.contains("f.txt"))
        XCTAssertTrue(out.output.contains("rw"))
    }

    func testHeadAndTail() async throws {
        let root = try makeTempRoot()
        try write("1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n11\n12\n", to: "n.txt", in: root)
        let h = try await HeadTool().run(
            input: ["path": "n.txt", "lines": 3], context: ctx(root))
        XCTAssertEqual(h.output, "1\n2\n3")
        let t = try await TailTool().run(
            input: ["path": "n.txt", "lines": 3], context: ctx(root))
        XCTAssertEqual(t.output, "10\n11\n12")
    }

    func testWcAllAndModes() async throws {
        let root = try makeTempRoot()
        try write("one two\nthree\n", to: "x.txt", in: root)
        let all = try await WcTool().run(
            input: ["paths": ["x.txt"]], context: ctx(root))
        XCTAssertTrue(all.output.contains("2 3 14 x.txt") ||
                       all.output.contains("2 3 13 x.txt"))
        let lines = try await WcTool().run(
            input: ["paths": ["x.txt"], "mode": "lines"], context: ctx(root))
        XCTAssertEqual(lines.output, "2 x.txt")
    }

    func testStatBasic() async throws {
        let root = try makeTempRoot()
        try write("hi", to: "s.txt", in: root)
        let out = try await StatTool().run(
            input: ["path": "s.txt"], context: ctx(root))
        XCTAssertTrue(out.output.contains("type=file"))
        XCTAssertTrue(out.output.contains("size=2"))
    }

    func testDuSummarize() async throws {
        let root = try makeTempRoot()
        try write(String(repeating: "x", count: 100), to: "f.txt", in: root)
        let out = try await DuTool().run(
            input: ["summarize": true], context: ctx(root))
        XCTAssertTrue(out.output.contains("100"))
    }

    func testBasenameAndDirname() async throws {
        let root = try makeTempRoot()
        let b = try await BasenameTool().run(
            input: ["path": "a/b/c.swift", "suffix": ".swift"], context: ctx(root))
        XCTAssertEqual(b.output, "c")
        let d = try await DirnameTool().run(
            input: ["path": "a/b/c.swift"], context: ctx(root))
        XCTAssertEqual(d.output, "a/b")
    }

    func testFindByName() async throws {
        let root = try makeTempRoot()
        try FileManager.default.createDirectory(
            at: root.appendingPathComponent("sub"),
            withIntermediateDirectories: true)
        try write("a", to: "alpha.txt", in: root)
        try write("b", to: "sub/alpha.txt", in: root)
        let out = try await FindTool().run(
            input: ["name": "alpha.txt"], context: ctx(root))
        let matches = out.output.split(separator: "\n").count
        XCTAssertEqual(matches, 2)
    }

    func testWhichFindsBinSh() async throws {
        let root = try makeTempRoot()
        let out = try await WhichTool().run(
            input: ["command": "sh"], context: ctx(root))
        XCTAssertTrue(out.output.hasSuffix("/sh"))
    }

    func testWhichRejectsPath() async throws {
        let root = try makeTempRoot()
        do {
            _ = try await WhichTool().run(
                input: ["command": "/bin/sh"], context: ctx(root))
            XCTFail("expected invalidInput")
        } catch let err as ToolError {
            if case .invalidInput = err { return }
            XCTFail("expected .invalidInput, got \(err)")
        }
    }

    // MARK: - Text family

    func testSortAscendingAndNumeric() async throws {
        let root = try makeTempRoot()
        try write("3\n10\n1\n", to: "n.txt", in: root)
        let lex = try await SortTool().run(
            input: ["path": "n.txt"], context: ctx(root))
        XCTAssertEqual(lex.output, "1\n10\n3")
        let num = try await SortTool().run(
            input: ["path": "n.txt", "numeric": true], context: ctx(root))
        XCTAssertEqual(num.output, "1\n3\n10")
    }

    func testUniqDropsAdjacentDupes() async throws {
        let root = try makeTempRoot()
        try write("a\na\nb\na\n", to: "u.txt", in: root)
        let out = try await UniqTool().run(
            input: ["path": "u.txt"], context: ctx(root))
        XCTAssertEqual(out.output, "a\nb\na")
    }

    func testCutFields() async throws {
        let root = try makeTempRoot()
        try write("a:b:c\nd:e:f\n", to: "c.txt", in: root)
        let out = try await CutTool().run(
            input: ["path": "c.txt", "delimiter": ":", "fields": [1, 3]],
            context: ctx(root))
        XCTAssertEqual(out.output, "a:c\nd:f")
    }

    func testTrTranslateAndDelete() async throws {
        let root = try makeTempRoot()
        let trans = try await TrTool().run(
            input: ["input": "abc", "from": "ab", "to": "AB"],
            context: ctx(root))
        XCTAssertEqual(trans.output, "ABc")
        let del = try await TrTool().run(
            input: ["input": "hello world", "delete": "lo"],
            context: ctx(root))
        XCTAssertEqual(del.output, "he wrd")
    }

    func testPasteMergesColumns() async throws {
        let root = try makeTempRoot()
        try write("1\n2\n", to: "a.txt", in: root)
        try write("A\nB\n", to: "b.txt", in: root)
        let out = try await PasteTool().run(
            input: ["paths": ["a.txt", "b.txt"], "delimiter": ","],
            context: ctx(root))
        XCTAssertEqual(out.output, "1,A\n2,B")
    }

    func testCommThreeColumn() async throws {
        let root = try makeTempRoot()
        try write("a\nb\nc\n", to: "a.txt", in: root)
        try write("b\nc\nd\n", to: "b.txt", in: root)
        let out = try await CommTool().run(
            input: ["pathA": "a.txt", "pathB": "b.txt"],
            context: ctx(root))
        XCTAssertTrue(out.output.contains("a"))
        XCTAssertTrue(out.output.contains("d"))
    }

    func testXxdHex() async throws {
        let root = try makeTempRoot()
        try write("Hello!", to: "x.bin", in: root)
        let out = try await XxdTool().run(
            input: ["path": "x.bin"], context: ctx(root))
        XCTAssertTrue(out.output.hasPrefix("00000000:"))
        // 'H' 'e' = 4865, 'l' 'l' = 6c6c, then space between pairs.
        XCTAssertTrue(out.output.contains("4865 6c6c"))
        // ASCII gutter shows the printable representation.
        XCTAssertTrue(out.output.contains("Hello!"))
    }

    func testHashFamily() async throws {
        let root = try makeTempRoot()
        try write("abc", to: "h.txt", in: root)
        let md5 = try await Md5Tool().run(
            input: ["path": "h.txt"], context: ctx(root))
        XCTAssertTrue(md5.output.hasPrefix("900150983cd24fb0d6963f7d28e17f72"))
        let sha1 = try await Sha1Tool().run(
            input: ["path": "h.txt"], context: ctx(root))
        XCTAssertTrue(sha1.output.hasPrefix("a9993e364706816aba3e25717850c26c9cd0d89d"))
        let sha256 = try await Sha256Tool().run(
            input: ["path": "h.txt"], context: ctx(root))
        XCTAssertTrue(sha256.output.hasPrefix(
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"))
    }

    func testBase64RoundTrip() async throws {
        let root = try makeTempRoot()
        let enc = try await Base64Tool().run(
            input: ["input": "hello"], context: ctx(root))
        XCTAssertEqual(enc.output, "aGVsbG8=")
        let dec = try await Base64Tool().run(
            input: ["input": "aGVsbG8=", "decode": true], context: ctx(root))
        XCTAssertEqual(dec.output, "hello")
    }

    // MARK: - System family

    func testUnameAndId() async throws {
        let root = try makeTempRoot()
        let u = try await UnameTool().run(input: [:], context: ctx(root))
        XCTAssertFalse(u.output.isEmpty)
        let i = try await IdTool().run(input: [:], context: ctx(root))
        XCTAssertTrue(i.output.contains("uid="))
        XCTAssertTrue(i.output.contains("gid="))
    }

    func testEnvRedactsSecrets() async throws {
        let root = try makeTempRoot()
        let env: [String: String] = [
            "PATH": "/usr/bin",
            "OPENAI_API_KEY": "sk-real-secret",
            "GITHUB_TOKEN": "ghp_real",
            "PUBLIC_VAR": "ok",
        ]
        let out = try await EnvTool().run(
            input: [:],
            context: ToolContext(rootDirectory: root,
                                 permission: AutoPermissionDelegate(),
                                 environment: env))
        XCTAssertTrue(out.output.contains("PATH=/usr/bin"))
        XCTAssertTrue(out.output.contains("OPENAI_API_KEY=<redacted>"))
        XCTAssertTrue(out.output.contains("GITHUB_TOKEN=<redacted>"))
        XCTAssertTrue(out.output.contains("PUBLIC_VAR=ok"))
        XCTAssertFalse(out.output.contains("sk-real-secret"))
    }

    func testDateIso8601() async throws {
        let root = try makeTempRoot()
        let out = try await DateTool().run(input: [:], context: ctx(root))
        XCTAssertTrue(out.output.contains("T"))
        XCTAssertTrue(out.output.contains("-"))
    }

    // MARK: - Mutating family

    func testTouchCreatesFile() async throws {
        let root = try makeTempRoot()
        let out = try await TouchTool().run(
            input: ["path": "new.txt"], context: ctx(root))
        XCTAssertFalse(out.isError)
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: root.appendingPathComponent("new.txt").path))
    }

    func testMkdirAndCpAndMvAndRm() async throws {
        let root = try makeTempRoot()
        // mkdir
        _ = try await MkdirTool().run(
            input: ["path": "sub"], context: ctx(root))
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: root.appendingPathComponent("sub").path))
        // cp
        try write("payload", to: "a.txt", in: root)
        _ = try await CpTool().run(
            input: ["src": "a.txt", "dst": "b.txt"], context: ctx(root))
        XCTAssertEqual(try String(contentsOf: root.appendingPathComponent("b.txt")),
                       "payload")
        // mv
        _ = try await MvTool().run(
            input: ["src": "b.txt", "dst": "sub/b.txt"], context: ctx(root))
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: root.appendingPathComponent("sub/b.txt").path))
        // rm (file, no confirm needed)
        _ = try await RmTool().run(
            input: ["path": "sub/b.txt"], context: ctx(root))
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: root.appendingPathComponent("sub/b.txt").path))
    }

    func testRmRequiresConfirmForRecursive() async throws {
        let root = try makeTempRoot()
        _ = try await MkdirTool().run(
            input: ["path": "doomed"], context: ctx(root))
        do {
            _ = try await RmTool().run(
                input: ["path": "doomed"], context: ctx(root))
            XCTFail("expected invalidInput (directory without confirm)")
        } catch let err as ToolError {
            if case .invalidInput = err {} else {
                XCTFail("expected .invalidInput, got \(err)")
            }
        }
        do {
            _ = try await RmTool().run(
                input: ["path": "doomed", "recursive": true],
                context: ctx(root))
            XCTFail("expected invalidInput (recursive without confirm)")
        } catch let err as ToolError {
            if case .invalidInput = err {} else {
                XCTFail("expected .invalidInput, got \(err)")
            }
        }
        let ok = try await RmTool().run(
            input: ["path": "doomed", "recursive": true, "confirm": true],
            context: ctx(root))
        XCTAssertFalse(ok.isError)
    }

    func testRmRefusesAgentRoot() async throws {
        let root = try makeTempRoot()
        do {
            _ = try await RmTool().run(
                input: ["path": ".", "recursive": true, "confirm": true],
                context: ctx(root))
            XCTFail("expected permissionDenied")
        } catch let err as ToolError {
            if case .permissionDenied = err { return }
            XCTFail("expected .permissionDenied, got \(err)")
        }
    }

    func testLnCreatesSymlinkInsideRoot() async throws {
        let root = try makeTempRoot()
        try write("hi", to: "target.txt", in: root)
        _ = try await LnTool().run(
            input: ["target": "target.txt", "linkPath": "link.txt"],
            context: ctx(root))
        let linkURL = root.appendingPathComponent("link.txt")
        var attrs = try FileManager.default.attributesOfItem(atPath: linkURL.path)
        XCTAssertEqual(attrs[.type] as? FileAttributeType, .typeSymbolicLink)
        _ = attrs.removeValue(forKey: .type)
    }

    // MARK: - symlink farm traversal

    private func ctxWithExtras(_ root: URL, extras: [URL]) -> ToolContext {
        ToolContext(rootDirectory: root,
                    additionalReadRoots: extras,
                    permission: AutoPermissionDelegate(allowDangerous: true))
    }

    /// `find` must surface farm-style symlinks (link inside root, target
    /// inside an `additionalReadRoots` entry) — without this, every
    /// project chat using the symlink farm would see an empty find.
    func testFindFollowsTrustedSymlink() async throws {
        let farm = try makeTempRoot()
        let realSource = try makeTempRoot()
        try write("hello", to: "main.swift", in: realSource)
        try FileManager.default.createSymbolicLink(
            at: farm.appendingPathComponent("main.swift"),
            withDestinationURL: realSource.appendingPathComponent("main.swift"))
        let out = try await FindTool().run(
            input: ["name": "main.swift"],
            context: ctxWithExtras(farm, extras: [realSource]))
        XCTAssertTrue(out.output.contains("main.swift"))
    }

    /// A symlink pointing OUTSIDE the trust boundary stays skipped:
    /// the farm-strategy concession is scoped to additionalReadRoots,
    /// not a blanket "follow everything".
    func testFindSkipsUntrustedSymlink() async throws {
        let farm = try makeTempRoot()
        let untrusted = try makeTempRoot()
        try write("secret", to: "leak.txt", in: untrusted)
        try FileManager.default.createSymbolicLink(
            at: farm.appendingPathComponent("leak.txt"),
            withDestinationURL: untrusted.appendingPathComponent("leak.txt"))
        let out = try await FindTool().run(
            input: ["name": "leak.txt"],
            context: ctxWithExtras(farm, extras: []))
        XCTAssertFalse(out.output.contains("leak.txt"))
    }

    /// `du` must count bytes through farm symlinks; without trusted-
    /// root awareness the farm would always report 0.
    func testDuChargesTrustedSymlinkSize() async throws {
        let farm = try makeTempRoot()
        let realSource = try makeTempRoot()
        let payload = String(repeating: "a", count: 1000)
        try write(payload, to: "big.txt", in: realSource)
        try FileManager.default.createSymbolicLink(
            at: farm.appendingPathComponent("big.txt"),
            withDestinationURL: realSource.appendingPathComponent("big.txt"))
        let out = try await DuTool().run(
            input: ["summarize": true],
            context: ctxWithExtras(farm, extras: [realSource]))
        XCTAssertTrue(out.output.contains("1000"))
    }

    /// `write` must refuse a path whose resolved target sneaks outside
    /// the trust boundary via a symlink in the agent root.
    func testWriteRefusesSymlinkEscape() async throws {
        let root = try makeTempRoot()
        let outside = try makeTempRoot()
        try write("orig", to: "victim.txt", in: outside)
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent("sneak.txt"),
            withDestinationURL: outside.appendingPathComponent("victim.txt"))
        do {
            _ = try await WriteTool().run(
                input: ["path": "sneak.txt", "content": "owned"],
                context: ctx(root))
            XCTFail("expected permissionDenied")
        } catch let err as ToolError {
            guard case .permissionDenied = err else {
                XCTFail("expected permissionDenied, got \(err)")
                return
            }
        }
        let after = try String(
            contentsOf: outside.appendingPathComponent("victim.txt"),
            encoding: .utf8)
        XCTAssertEqual(after, "orig")
    }

    /// `write` through a farm symlink whose target IS in
    /// additionalReadRoots must succeed — the boundary widens for the
    /// project's real source folders.
    func testWriteThroughFarmSymlinkSucceeds() async throws {
        let farm = try makeTempRoot()
        let realSource = try makeTempRoot()
        try write("orig", to: "file.txt", in: realSource)
        try FileManager.default.createSymbolicLink(
            at: farm.appendingPathComponent("file.txt"),
            withDestinationURL: realSource.appendingPathComponent("file.txt"))
        _ = try await WriteTool().run(
            input: ["path": "file.txt", "content": "updated"],
            context: ctxWithExtras(farm, extras: [realSource]))
        let after = try String(
            contentsOf: realSource.appendingPathComponent("file.txt"),
            encoding: .utf8)
        XCTAssertEqual(after, "updated")
    }

    /// `sandboxBlockedSymlinkTarget` returns the resolved parent
    /// when a sandbox EPERM lands on a symlinked path. The host
    /// uses this to push the parent dir into the project's
    /// "Grant access" list.
    func testSandboxBlockedSymlinkTargetIdentifiesParent() throws {
        let farm = try makeTempRoot()
        let outside = try makeTempRoot()
        try write("hi", to: "leaf.txt", in: outside)
        let linkURL = farm.appendingPathComponent("leaf.txt")
        try FileManager.default.createSymbolicLink(
            at: linkURL,
            withDestinationURL: outside.appendingPathComponent("leaf.txt"))
        let err = NSError(domain: NSCocoaErrorDomain,
                          code: NSFileReadNoPermissionError,
                          userInfo: nil)
        let parent = sandboxBlockedSymlinkTarget(
            from: err, accessedFrom: linkURL)
        XCTAssertEqual(
            parent?.path,
            (outside.path as NSString).resolvingSymlinksInPath)
    }

    /// EPERM on a path that isn't a symlink resolves to itself →
    /// the helper returns nil so callers fall through to their
    /// existing "real" permission-denied path. Otherwise we'd
    /// nudge the user toward granting access to a directory they
    /// already own and the sheet would look broken.
    func testSandboxBlockedSymlinkTargetIgnoresNonLinkPaths() throws {
        let root = try makeTempRoot()
        try write("hi", to: "regular.txt", in: root)
        let url = root.appendingPathComponent("regular.txt")
        let err = NSError(domain: NSCocoaErrorDomain,
                          code: NSFileReadNoPermissionError,
                          userInfo: nil)
        XCTAssertNil(sandboxBlockedSymlinkTarget(
            from: err, accessedFrom: url))
    }

    /// A non-EPERM error must fall through — we don't want to
    /// rewrite e.g. a UTF-8 decode failure into a permission
    /// banner.
    func testSandboxBlockedSymlinkTargetIgnoresOtherErrors() throws {
        let root = try makeTempRoot()
        let url = root.appendingPathComponent("anything.txt")
        let err = NSError(domain: NSCocoaErrorDomain,
                          code: NSFileNoSuchFileError,
                          userInfo: nil)
        XCTAssertNil(sandboxBlockedSymlinkTarget(
            from: err, accessedFrom: url))
    }

    func testChmodOctal() async throws {
        let root = try makeTempRoot()
        try write("x", to: "f.txt", in: root)
        _ = try await ChmodTool().run(
            input: ["path": "f.txt", "mode": "600"], context: ctx(root))
        let attrs = try FileManager.default.attributesOfItem(
            atPath: root.appendingPathComponent("f.txt").path)
        let perms = (attrs[.posixPermissions] as? NSNumber)?.intValue ?? 0
        XCTAssertEqual(perms & 0o777, 0o600)
    }
}
