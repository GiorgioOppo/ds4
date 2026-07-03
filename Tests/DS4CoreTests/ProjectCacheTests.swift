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
                                            "project_read", "project_search", "file_delete"]))
    }
}
