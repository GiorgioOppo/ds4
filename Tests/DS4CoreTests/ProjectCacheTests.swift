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
        XCTAssertTrue(ProjectCache.shared.listTool(path: "../etc").contains("non valido"))
    }

    func testReadPaginated() {
        let first = ProjectCache.shared.readTool(path: "Sources/Long.txt", fromLine: 1)
        XCTAssertTrue(first.contains("1\tline 1"))
        XCTAssertTrue(first.contains("from_line=121"))      // pagination hint
        let second = ProjectCache.shared.readTool(path: "Sources/Long.txt", fromLine: 121)
        XCTAssertTrue(second.contains("121\tline 121"))
        XCTAssertTrue(ProjectCache.shared.readTool(path: "nope.swift", fromLine: 1)
            .contains("non trovato"))
    }

    func testSearch() {
        let hits = ProjectCache.shared.searchTool(query: "ANSWER")
        XCTAssertTrue(hits.contains("Sources/Main.swift:1"))
        XCTAssertTrue(ProjectCache.shared.searchTool(query: "zzz_not_there").contains("Nessun risultato"))
    }

    func testToolsAreRegistered() {
        let names = Set(ToolRegistry.builtins.map(\.spec.name))
        XCTAssertTrue(names.isSuperset(of: ["project_list", "project_read", "project_search"]))
    }
}
