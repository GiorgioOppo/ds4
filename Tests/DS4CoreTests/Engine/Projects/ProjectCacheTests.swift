import XCTest
@testable import DS4Engine

/// ProjectCache: index, list, paginated read and search over a temp project.
final class ProjectCacheTests: XCTestCase {
    var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ds4-project-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root.appendingPathComponent("Sources"),
                                                withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: root.appendingPathComponent(".git"),
                                                withIntermediateDirectories: true)
        try "let answer = 42\nfunc main() {}\n".write(
            to: root.appendingPathComponent("Sources/Main.swift"), atomically: true, encoding: .utf8)
        let long = (1...300).map { "line \($0)" }.joined(separator: "\n")
        try long.write(to: root.appendingPathComponent("Sources/Long.txt"),
                       atomically: true, encoding: .utf8)
        try "junk".write(to: root.appendingPathComponent(".git/config"),
                         atomically: true, encoding: .utf8)
        ProjectCache.shared.load(root: root)
    }

    override func tearDownWithError() throws {
        ProjectCache.shared.clear()
        try? FileManager.default.removeItem(at: root)
    }

    func testIndexSkipsGit() {
        let info = ProjectCache.shared.info()
        XCTAssertEqual(info?.fileCount, 2)            // .git/config excluded
    }

    func testActiveProjectContextIsInjectedIntoProjectAgents() throws {
        let context = try XCTUnwrap(ProjectCache.shared.agentContext())
        XCTAssertEqual(context.name, root.lastPathComponent)
        XCTAssertEqual(context.rootPath, root.standardizedFileURL.path)
        XCTAssertEqual(context.fileCount, 2)
        XCTAssertTrue(context.systemInstruction.contains("ACTIVE PROJECT SELECTED IN THE GUI"))
        XCTAssertTrue(context.systemInstruction.contains("Do not ask which project"))

        let coding = try XCTUnwrap(AgentProfile.defaults.first { $0.id == "coding" })
        let scoped = coding.withActiveProjectContext(context)
        XCTAssertTrue(scoped.systemPrompt.contains(context.rootPath))
        XCTAssertTrue(scoped.systemPrompt.contains("project_inspect"))

        let general = try XCTUnwrap(AgentProfile.defaults.first { $0.id == "generale" })
        XCTAssertEqual(general.withActiveProjectContext(context), general)

        ProjectCache.shared.clear()
        XCTAssertNil(ProjectCache.shared.agentContext())
    }

    func testList() {
        let rootList = ProjectCache.shared.listTool(path: "")
        XCTAssertTrue(rootList.contains("Sources/"))
        let sub = ProjectCache.shared.listTool(path: "Sources")
        XCTAssertTrue(sub.contains("Main.swift"))
        XCTAssertTrue(sub.contains("Long.txt"))
        XCTAssertTrue(ProjectCache.shared.listTool(path: "../etc").contains("Invalid path"))
    }

    func testReadPaginated() {
        let first = ProjectCache.shared.readTool(path: "Sources/Long.txt", fromLine: 1)
        XCTAssertTrue(first.contains("1\tline 1"))
        XCTAssertTrue(first.contains("from_line=121"))      // pagination hint
        let second = ProjectCache.shared.readTool(path: "Sources/Long.txt", fromLine: 121)
        XCTAssertTrue(second.contains("121\tline 121"))
        XCTAssertTrue(ProjectCache.shared.readTool(path: "nope.swift", fromLine: 1)
            .contains("not found"))
    }

    func testSearch() {
        let hits = ProjectCache.shared.searchTool(query: "ANSWER")
        XCTAssertTrue(hits.contains("Sources/Main.swift:1"))
        XCTAssertTrue(ProjectCache.shared.searchTool(query: "zzz_not_there").contains("No results"))
    }

    /// project_read 'lines': larger chunks on demand, hard-capped at maxReadLines.
    func testReadCustomChunk() {
        let big = ProjectCache.shared.readTool(path: "Sources/Long.txt", fromLine: 1, maxLines: 250)
        XCTAssertTrue(big.contains("250\tline 250"))
        XCTAssertTrue(big.contains("from_line=251"))
        let all = ProjectCache.shared.readTool(path: "Sources/Long.txt", fromLine: 1, maxLines: 100_000)
        XCTAssertTrue(all.contains("300\tline 300"))     // cap 400 ≥ 300 lines → whole file
        XCTAssertFalse(all.contains("continues"))
        let one = ProjectCache.shared.readTool(path: "Sources/Long.txt", fromLine: 5, maxLines: 1)
        XCTAssertTrue(one.contains("5\tline 5"))
        XCTAssertTrue(one.contains("from_line=6"))
    }

    /// Scoped search: a path prefix restricts the files considered.
    func testSearchWithPathFilter() {
        XCTAssertTrue(ProjectCache.shared.searchTool(query: "answer", pathPrefix: "Sources")
            .contains("Sources/Main.swift:1"))
        XCTAssertTrue(ProjectCache.shared.searchTool(query: "answer", pathPrefix: "Sources/Main.swift")
            .contains("Sources/Main.swift:1"))
        XCTAssertTrue(ProjectCache.shared.searchTool(query: "answer", pathPrefix: "Elsewhere")
            .contains("No indexed files"))
        XCTAssertTrue(ProjectCache.shared.searchTool(query: "answer", pathPrefix: "../etc")
            .contains("Invalid path"))
    }

    /// project_tree: whole-project overview with per-directory file counts.
    func testTree() {
        let tree = ProjectCache.shared.treeTool()
        XCTAssertTrue(tree.contains("Sources/ (2)"), "got: \(tree)")
        XCTAssertTrue(ProjectCache.shared.treeTool(maxDepth: 1).contains("Sources/ (2)"))
    }

    /// project_find: match by path substring and by '*' wildcard.
    func testFind() {
        XCTAssertTrue(ProjectCache.shared.findTool(pattern: "main").contains("Sources/Main.swift"))
        XCTAssertTrue(ProjectCache.shared.findTool(pattern: "*.txt").contains("Sources/Long.txt"))
        XCTAssertTrue(ProjectCache.shared.findTool(pattern: "Sources/*.swift").contains("Sources/Main.swift"))
        XCTAssertTrue(ProjectCache.shared.findTool(pattern: "zzz_nope").contains("No files match"))
        XCTAssertTrue(ProjectCache.shared.findTool(pattern: "x").contains("too short"))
    }

    /// file_delete: removes the file and de-indexes it; directories and
    /// traversal are refused.
    func testDeleteFile() {
        _ = ProjectCache.shared.writeFileTool(path: "tmp/Doomed.txt", content: "bye")
        XCTAssertTrue(ProjectCache.shared.findTool(pattern: "Doomed").contains("tmp/Doomed.txt"))
        XCTAssertTrue(ProjectCache.shared.deleteFileTool(path: "tmp/Doomed.txt").contains("Deleted"))
        XCTAssertTrue(ProjectCache.shared.findTool(pattern: "Doomed").contains("No files match"))
        XCTAssertTrue(ProjectCache.shared.deleteFileTool(path: "tmp/Doomed.txt").contains("not found"))
        XCTAssertTrue(ProjectCache.shared.deleteFileTool(path: "Sources").contains("directory"))
        XCTAssertTrue(ProjectCache.shared.deleteFileTool(path: "../evil").contains("invalid path"))
    }

    func testToolsAreRegistered() {
        let names = Set(ToolRegistry.builtins.map(\.spec.name))
        XCTAssertTrue(names.isSuperset(of: ["project_tree", "project_list", "project_find",
                                            "project_read", "project_search", "project_inspect", "project_reload",
                                            "file_delete"]))
    }

    /// One model round can carry independent orientation, search, and source
    /// evidence. The sectioned response remains easy to cite during a review.
    func testProjectInspectBatchesIndependentOperations() throws {
        let arguments = #"""
        {
          "tree_depth": 2,
          "find": ["*.swift"],
          "search": [{"query": "answer", "path": "Sources"}],
          "read": [
            {"path": "Sources/Main.swift", "from_line": 1, "lines": 2},
            {"path": "Sources/Long.txt", "from_line": 121, "lines": 2}
          ]
        }
        """#
        let tool = try XCTUnwrap(ToolRegistry.builtin(named: "project_inspect"))
        let output = tool.run(arguments)
        XCTAssertTrue(output.contains("## Read · Sources/Main.swift:1+2"), output)
        XCTAssertTrue(output.contains("1\tlet answer = 42"), output)
        XCTAssertTrue(output.contains("121\tline 121"), output)
        XCTAssertTrue(output.contains("## Search · answer in Sources"), output)
        XCTAssertTrue(output.contains("Sources/Main.swift:1"), output)
        XCTAssertTrue(output.contains("## Find · *.swift"), output)
        XCTAssertTrue(output.contains("## Tree · depth 2"), output)
    }

    func testProjectInspectRejectsEmptyAndCapsOperations() throws {
        let tool = try XCTUnwrap(ToolRegistry.builtin(named: "project_inspect"))
        XCTAssertTrue(tool.run("{}").contains("No inspection requested"))
        let searches = (1...14).map { _ in #"{"query":"answer"}"# }.joined(separator: ",")
        let output = tool.run(#"{"search":["# + searches + "]}")
        XCTAssertTrue(output.contains("operation(s) omitted"), output)
    }

    /// project_reload picks up files created OUTSIDE the tools (scripts, git):
    /// the index is an import-time snapshot until re-indexed.
    func testReloadPicksUpExternalChanges() throws {
        try "let late = true\n".write(to: root.appendingPathComponent("Sources/Late.swift"),
                                      atomically: true, encoding: .utf8)
        XCTAssertTrue(ProjectCache.shared.findTool(pattern: "Late").contains("No files match"))
        let info = ProjectCache.shared.reload()
        XCTAssertEqual(info?.fileCount, 3)
        XCTAssertTrue(ProjectCache.shared.findTool(pattern: "Late").contains("Sources/Late.swift"))
        ProjectCache.shared.clear()
        XCTAssertNil(ProjectCache.shared.reload())
    }

    /// project_edit must base the replacement on the CURRENT file, not on the
    /// content cache: an external change (git, the user's editor) between the
    /// caching read and the edit must survive, not be silently reverted.
    func testEditReadsFreshDiskContent() throws {
        _ = ProjectCache.shared.readTool(path: "Sources/Main.swift", fromLine: 1)   // warm the cache
        try "let answer = 43\nfunc main() {}\n".write(
            to: root.appendingPathComponent("Sources/Main.swift"), atomically: true, encoding: .utf8)
        // "43" exists only on disk: a stale-cache edit would report "not found".
        let out = ProjectCache.shared.editTool(path: "Sources/Main.swift", find: "43", replace: "44")
        XCTAssertTrue(out.contains("Edited"), "got: \(out)")
        let disk = try String(contentsOf: root.appendingPathComponent("Sources/Main.swift"), encoding: .utf8)
        XCTAssertTrue(disk.contains("let answer = 44"))
    }

    /// file_add append on a newline-terminated file must not manufacture a
    /// blank line before the added content (and keeps the trailing newline).
    func testAddLinesAppendKeepsShape() throws {
        _ = ProjectCache.shared.writeFileTool(path: "tmp/nl.txt", content: "a\nb\n")
        let out = ProjectCache.shared.addLinesTool(path: "tmp/nl.txt", content: "c")
        XCTAssertTrue(out.contains("at the end"), "got: \(out)")
        let disk = try String(contentsOf: root.appendingPathComponent("tmp/nl.txt"), encoding: .utf8)
        XCTAssertEqual(disk, "a\nb\nc\n")
    }

    /// file_modify enforces the same resulting-size cap as file_add.
    func testModifyLinesResultCapped() throws {
        try FileManager.default.createDirectory(at: root.appendingPathComponent("tmp"),
                                                withIntermediateDirectories: true)
        let half = String(repeating: "a", count: 450_000)
        try (half + "\n" + half).write(to: root.appendingPathComponent("tmp/big.txt"),
                                       atomically: true, encoding: .utf8)
        let out = ProjectCache.shared.modifyLinesTool(path: "tmp/big.txt",
                                                      content: String(repeating: "b", count: 600_000),
                                                      fromLine: 1)
        XCTAssertTrue(out.contains("too large"), "got: \(out)")
    }

    /// file_read whole-file mode: a cap landing inside a multibyte character
    /// must back off to a character boundary, not report "non-text".
    func testWholeFileReadBacksOffUTF8Boundary() throws {
        try FileManager.default.createDirectory(at: root.appendingPathComponent("tmp"),
                                                withIntermediateDirectories: true)
        let content = String(repeating: "a", count: 24 * 1024 - 1) + "é" + String(repeating: "b", count: 64)
        try content.write(to: root.appendingPathComponent("tmp/utf8.txt"),
                          atomically: true, encoding: .utf8)
        let out = ProjectCache.shared.readFileTool(path: "tmp/utf8.txt")
        XCTAssertFalse(out.contains("Non-text"), "got: \(out.prefix(200))")
        XCTAssertTrue(out.contains("truncated"))
    }

    /// Symlinks must be second-class citizens: never indexed, never readable,
    /// never a write path — a link in an imported/cloned repo can point
    /// anywhere on disk.
    func testSymlinksAreRefused() throws {
        let outside = FileManager.default.temporaryDirectory
            .appendingPathComponent("ds4-outside-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: outside) }
        let sentinel = "OUTSIDE_SENTINEL_CONTENT"
        try sentinel.write(to: outside.appendingPathComponent("secret.txt"),
                           atomically: true, encoding: .utf8)
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent("link.txt"),
            withDestinationURL: outside.appendingPathComponent("secret.txt"))
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent("linkdir"),
            withDestinationURL: outside)
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent("internal-link"),
            withDestinationURL: root.appendingPathComponent("Sources"))
        let brokenTarget = outside.appendingPathComponent("missing.txt")
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent("broken-link.txt"),
            withDestinationURL: brokenTarget)
        let brokenDirectoryTarget = outside.appendingPathComponent("missing-directory")
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent("broken-linkdir"),
            withDestinationURL: brokenDirectoryTarget)
        ProjectCache.shared.load(root: root)                       // re-index with the links present
        XCTAssertTrue(ProjectCache.shared.findTool(pattern: "link").contains("No files match"))
        XCTAssertFalse(ProjectCache.shared.readFileTool(path: "link.txt").contains(sentinel))
        XCTAssertFalse(ProjectCache.shared.readFileTool(path: "linkdir/secret.txt").contains(sentinel))
        XCTAssertTrue(ProjectCache.shared.lineCountTool(path: "linkdir/secret.txt").contains("invalid path"))
        XCTAssertTrue(ProjectCache.shared.writeFileTool(path: "linkdir/evil.txt", content: "x")
            .contains("invalid path"))
        XCTAssertTrue(ProjectCache.shared.writeFileTool(path: "link.txt", content: "replaced")
            .contains("invalid path"))
        XCTAssertTrue(ProjectCache.shared.addLinesTool(path: "linkdir/secret.txt", content: "x")
            .contains("invalid path"))
        XCTAssertTrue(ProjectCache.shared.modifyLinesTool(path: "linkdir/secret.txt", content: "x", fromLine: 1)
            .contains("invalid path"))
        XCTAssertTrue(ProjectCache.shared.deleteFileTool(path: "linkdir/secret.txt")
            .contains("invalid path"))
        XCTAssertTrue(ProjectCache.shared.writeTool(path: "linkdir/evil.swift", content: "x")
            .contains("Invalid path"))
        XCTAssertTrue(ProjectCache.shared.writeFileTool(path: "internal-link/evil.txt", content: "x")
            .contains("invalid path"))
        XCTAssertTrue(ProjectCache.shared.writeFileTool(path: "broken-link.txt", content: "x")
            .contains("invalid path"))
        XCTAssertTrue(ProjectCache.shared.writeFileTool(path: "broken-linkdir/evil.txt", content: "x")
            .contains("invalid path"))
        XCTAssertTrue(ProjectCache.shared.editTool(path: "linkdir/secret.txt", find: "secret", replace: "x")
            .contains("Invalid path"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: outside.appendingPathComponent("evil.txt").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: brokenTarget.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: brokenDirectoryTarget.path))
        XCTAssertEqual(try String(contentsOf: outside.appendingPathComponent("secret.txt"), encoding: .utf8),
                       sentinel)
    }

    /// A missing directory hierarchy is not a symlink and remains a valid
    /// creation target; once created, every component is revalidated.
    func testWriteCreatesNewNestedDirectories() throws {
        let out = ProjectCache.shared.writeFileTool(path: "fresh/nested/ok.txt", content: "safe")
        XCTAssertTrue(out.contains("Created"), "got: \(out)")
        XCTAssertEqual(
            try String(contentsOf: root.appendingPathComponent("fresh/nested/ok.txt"), encoding: .utf8),
            "safe"
        )
    }

    /// project_read without any project says so, instead of a misleading
    /// "file not found in the index".
    func testReadWithoutProjectSaysNoProject() {
        ProjectCache.shared.clear()
        XCTAssertTrue(ProjectCache.shared.readTool(path: "x.swift", fromLine: 1)
            .contains("No project imported"))
    }
}
