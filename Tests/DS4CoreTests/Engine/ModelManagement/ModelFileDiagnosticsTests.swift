import XCTest
@testable import DS4Engine

/// The pre-flight model-path diagnostics must name the actual cause and the
/// fix: missing file, orphan `.part` download, container/legacy sandbox split.
final class ModelFileDiagnosticsTests: XCTestCase {
    private func temporaryPath(_ name: String) -> String {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("ds4-diag-\(UUID().uuidString)-\(name)").path
    }

    func testReadableFileHasNoIssue() throws {
        let path = temporaryPath("model.gguf")
        try Data([1, 2, 3]).write(to: URL(fileURLWithPath: path))
        addTeardownBlock { try? FileManager.default.removeItem(atPath: path) }

        XCTAssertNil(ModelFileDiagnostics.openabilityIssue(path: path))
    }

    func testMissingFileIsNamed() throws {
        let issue = ModelFileDiagnostics.openabilityIssue(
            path: temporaryPath("assente.gguf"))
        XCTAssertTrue(try XCTUnwrap(issue).contains("file non trovato"))
    }

    func testEmptyPathIsNamed() {
        XCTAssertEqual(ModelFileDiagnostics.openabilityIssue(path: ""),
                       "nessun modello selezionato")
    }

    func testOrphanPartFileSuggestsResumingTheDownload() throws {
        let path = temporaryPath("model.gguf")
        try Data(repeating: 0, count: 128).write(
            to: URL(fileURLWithPath: path + ".part"))
        addTeardownBlock {
            try? FileManager.default.removeItem(atPath: path + ".part")
        }

        let issue = try XCTUnwrap(
            ModelFileDiagnostics.openabilityIssue(path: path))
        XCTAssertTrue(issue.contains("download incompleto"))
        XCTAssertTrue(issue.contains("128 byte"))
    }

    func testContainerPathYieldsLegacyLocationAndHint() throws {
        let container = "/Users/oppog/Library/Containers/com.dwarfstar.app/Data"
            + "/Library/Application Support/DwarfStar/models/m.gguf"
        XCTAssertEqual(
            ModelFileDiagnostics.legacyLocation(forContainerPath: container),
            "/Users/oppog/Library/Application Support/DwarfStar/models/m.gguf")

        let issue = ModelFileDiagnostics.openabilityIssue(path: container)
        XCTAssertTrue(try XCTUnwrap(issue).contains("mv \""))
    }

    func testNonContainerPathHasNoLegacyLocation() throws {
        XCTAssertNil(ModelFileDiagnostics.legacyLocation(
            forContainerPath: "/Users/oppog/models/m.gguf"))
        let issue = ModelFileDiagnostics.openabilityIssue(
            path: temporaryPath("plain.gguf"))
        XCTAssertFalse(try XCTUnwrap(issue).contains("legacy"))
    }
}
