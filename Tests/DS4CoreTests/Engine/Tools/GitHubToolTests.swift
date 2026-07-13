import XCTest
@testable import DS4Engine

/// github_clone argument handling: repo parsing/validation, the pinned archive
/// URL, and the documentation shortlist. Everything here is pure — no network,
/// no processes, no filesystem.
final class GitHubToolTests: XCTestCase {

    // MARK: Repo parsing

    func testParseOwnerSlashName() {
        XCTAssertEqual(GitHubTool.parseRepo("apple/swift"),
                       GitHubTool.Repo(owner: "apple", name: "swift", ref: nil))
    }

    func testParseHTTPSURLVariants() {
        for raw in ["https://github.com/apple/swift",
                    "https://github.com/apple/swift.git",
                    "https://github.com/apple/swift/",
                    "https://www.github.com/apple/swift",
                    "github.com/apple/swift"] {
            XCTAssertEqual(GitHubTool.parseRepo(raw),
                           GitHubTool.Repo(owner: "apple", name: "swift", ref: nil), raw)
        }
    }

    func testParseSSHForm() {
        XCTAssertEqual(GitHubTool.parseRepo("git@github.com:apple/swift.git"),
                       GitHubTool.Repo(owner: "apple", name: "swift", ref: nil))
    }

    func testParseTreeURLTakesRefIncludingSlashes() {
        XCTAssertEqual(GitHubTool.parseRepo("https://github.com/apple/swift/tree/main"),
                       GitHubTool.Repo(owner: "apple", name: "swift", ref: "main"))
        XCTAssertEqual(GitHubTool.parseRepo("https://github.com/apple/swift/tree/release/6.0"),
                       GitHubTool.Repo(owner: "apple", name: "swift", ref: "release/6.0"))
    }

    func testExplicitRefWinsOverURLRef() {
        XCTAssertEqual(GitHubTool.parseRepo("https://github.com/apple/swift/tree/main", ref: "dev"),
                       GitHubTool.Repo(owner: "apple", name: "swift", ref: "dev"))
    }

    func testQueryAndFragmentAreStripped() {
        XCTAssertEqual(GitHubTool.parseRepo("https://github.com/apple/swift?tab=readme#usage"),
                       GitHubTool.Repo(owner: "apple", name: "swift", ref: nil))
    }

    func testRejectsOtherHostsAndEmbeddings() {
        XCTAssertNil(GitHubTool.parseRepo("https://gitlab.com/apple/swift"))
        XCTAssertNil(GitHubTool.parseRepo("https://evil.com/github.com/apple/swift"))
        XCTAssertNil(GitHubTool.parseRepo("user@host.com/apple/swift"))
    }

    func testRejectsBadComponents() {
        XCTAssertNil(GitHubTool.parseRepo("apple"))                 // no name
        XCTAssertNil(GitHubTool.parseRepo("apple/.."))              // traversal
        XCTAssertNil(GitHubTool.parseRepo("app le/swift"))          // space in owner
        XCTAssertNil(GitHubTool.parseRepo("apple/swi;ft"))          // shell-ish char
        XCTAssertNil(GitHubTool.parseRepo("-apple/swift"))          // leading hyphen
        XCTAssertNil(GitHubTool.parseRepo(""))
    }

    // MARK: Ref validation

    func testValidRef() {
        XCTAssertTrue(GitHubTool.validRef("main"))
        XCTAssertTrue(GitHubTool.validRef("feature/kv-reuse"))
        XCTAssertTrue(GitHubTool.validRef("v1.2.3"))
        XCTAssertTrue(GitHubTool.validRef("0a1b2c3d"))
        XCTAssertFalse(GitHubTool.validRef(".."))
        XCTAssertFalse(GitHubTool.validRef("a..b"))
        XCTAssertFalse(GitHubTool.validRef("-flag"))
        XCTAssertFalse(GitHubTool.validRef("/abs"))
        XCTAssertFalse(GitHubTool.validRef("trailing/"))
        XCTAssertFalse(GitHubTool.validRef("spa ce"))
        XCTAssertFalse(GitHubTool.validRef(""))
    }

    // MARK: Archive URL (host pinning)

    func testArchiveURLDefaultsToHEAD() {
        let repo = GitHubTool.Repo(owner: "apple", name: "swift", ref: nil)
        XCTAssertEqual(GitHubTool.archiveURL(for: repo).absoluteString,
                       "https://codeload.github.com/apple/swift/tar.gz/HEAD")
    }

    func testArchiveURLWithRef() {
        let repo = GitHubTool.Repo(owner: "apple", name: "swift", ref: "release/6.0")
        let url = GitHubTool.archiveURL(for: repo)
        XCTAssertEqual(url.host, "codeload.github.com")
        XCTAssertEqual(url.absoluteString,
                       "https://codeload.github.com/apple/swift/tar.gz/release/6.0")
    }

    // MARK: Documentation shortlist

    func testDocCandidatesOrderAndDedup() {
        let files = ["CHANGELOG.md", "Makefile", "README.md",
                     "Sources/README.md", "Sources/main.swift",
                     "docs/guide.md", "doc/notes.rst", "docs/img.png"]
        let docs = GitHubTool.docCandidates(in: files)
        XCTAssertEqual(docs, ["README.md",                    // root README first
                              "CHANGELOG.md",                 // other root .md
                              "docs/guide.md", "doc/notes.rst", // docs folders (no binaries)
                              "Sources/README.md"])           // nested READMEs last
    }

    func testDocCandidatesEmptyWhenNoDocs() {
        XCTAssertTrue(GitHubTool.docCandidates(in: ["Sources/a.swift", "Makefile"]).isEmpty)
    }
}
