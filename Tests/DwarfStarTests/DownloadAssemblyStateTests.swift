import CryptoKit
import DS4Engine
import Foundation
import XCTest
@testable import DwarfStar

final class DownloadAssemblyStateTests: XCTestCase {
    private func fixture() throws -> (URL, ModelCatalogEntry, Data) {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("assembly-ui-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let a = Data("piece-A".utf8), b = Data("piece-B".utf8), joined = a + b
        func sha(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
        let parts = [ModelTarget(id: "p1", file: "one.part1", approxGB: 1, note: "", sha256: sha(a), expectedSizeBytes: Int64(a.count), role: .splitFragment),
                     ModelTarget(id: "p2", file: "one.part2", approxGB: 1, note: "", sha256: sha(b), expectedSizeBytes: Int64(b.count), role: .splitFragment)]
        let output = ModelTarget(id: "final", file: "one.gguf", approxGB: 1, note: "", sha256: sha(joined), expectedSizeBytes: Int64(joined.count))
        let entry = ModelCatalogEntry(id: .deepSeek41Q4, displayName: "fixture", profile: .deepSeek41,
            summary: "", artifacts: parts, runtimeAvailability: .runnable, assemblyOutput: output)
        try a.write(to: directory.appendingPathComponent(parts[0].file))
        try b.write(to: directory.appendingPathComponent(parts[1].file))
        return (directory, entry, joined)
    }

    @MainActor
    func testFragmentsAloneRemainPartialUntilFinalGGUFExists() throws {
        let (dir, entry, _) = try fixture()
        defer { try? FileManager.default.removeItem(at: dir) }
        let runner = DownloadRunner(); runner.configure(searchDirectories: [dir], destination: dir)
        let state = runner.inspect(entry)
        XCTAssertEqual(state.state, .partial)
        XCTAssertEqual(state.installedArtifacts, 2)
        XCTAssertEqual(state.artifactCount, 3)
        XCTAssertNil(state.pathsByTargetID["final"])
    }

    @MainActor
    func testFinalRemainsSelectableAfterFragmentsAreRemovedWithoutNetworkOrMerge() throws {
        let (dir, entry, joined) = try fixture()
        defer { try? FileManager.default.removeItem(at: dir) }
        let output = dir.appendingPathComponent("one.gguf")
        try joined.write(to: output)
        for part in entry.artifacts { try FileManager.default.removeItem(at: dir.appendingPathComponent(part.file)) }
        let runner = DownloadRunner(); runner.configure(searchDirectories: [dir], destination: dir)
        XCTAssertEqual(runner.inspect(entry).state, .installed)
        var selected: String?
        runner.acquire(entry) { selected = $0; return true }
        XCTAssertEqual(selected, output.path)
        XCTAssertFalse(runner.isRunning)
    }

    @MainActor
    func testInvalidUnusedFragmentDoesNotHideCompleteFinal() throws {
        let (dir, entry, joined) = try fixture()
        defer { try? FileManager.default.removeItem(at: dir) }
        try joined.write(to: dir.appendingPathComponent("one.gguf"))
        try Data("wrong-size-leftover".utf8).write(to: dir.appendingPathComponent(entry.artifacts[0].file))
        let runner = DownloadRunner(); runner.configure(searchDirectories: [dir], destination: dir)
        let state = runner.inspect(entry)
        XCTAssertEqual(state.state, .installed)
        XCTAssertTrue(state.invalidArtifacts.isEmpty)
        XCTAssertEqual(state.pathsByTargetID["final"], dir.appendingPathComponent("one.gguf").path)
    }
}
