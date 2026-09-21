import CryptoKit
import DS4Core
import Foundation
import XCTest
@testable import DS4Engine

final class ModelAssemblyTests: XCTestCase {
    private func digest(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    private func fixture() throws -> (URL, ModelCatalogEntry, [URL], Data) {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("model-assembly-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let pieces = [Data("GGUF-first-piece".utf8), Data("second-piece-tail".utf8)]
        let targets = pieces.enumerated().map { index, data in
            ModelTarget(id: "part\(index)", file: "fixture.gguf.part\(index + 1)", approxGB: 1,
                note: "test", sha256: digest(data), expectedSizeBytes: Int64(data.count), role: .splitFragment)
        }
        let joined = pieces.reduce(into: Data()) { $0.append($1) }
        let output = ModelTarget(id: "output", file: "fixture.gguf", approxGB: 1, note: "test",
                                 sha256: digest(joined), expectedSizeBytes: Int64(joined.count))
        let entry = ModelCatalogEntry(id: .deepSeek41Q4, displayName: "fixture", profile: .deepSeek41,
            summary: "test", artifacts: targets, runtimeAvailability: .runnable, assemblyOutput: output)
        let urls = targets.map { directory.appendingPathComponent($0.file) }
        for (url, data) in zip(urls, pieces) { try data.write(to: url) }
        return (directory, entry, urls, joined)
    }
    func testAssemblesOrderedFragmentsAndOnlyFinalGGUFIsSelectable() async throws {
        let (dir, entry, sources, expected) = try fixture()
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertTrue(entry.isSelectable)
        XCTAssertEqual(entry.primaryArtifact?.file, "fixture.gguf")
        XCTAssertEqual(entry.assemblyPeakBytes, Int64(expected.count * 2))
        let result = try await ModelDownloader.assemble(entry: entry, fragmentURLs: sources, in: dir, chunkBytes: 3)
        XCTAssertEqual(result.disposition, .downloaded)
        XCTAssertEqual(try Data(contentsOf: result.fileURL), expected)
        XCTAssertTrue(sources.allSatisfy { FileManager.default.fileExists(atPath: $0.path) })
        XCTAssertFalse(FileManager.default.fileExists(atPath: result.fileURL.path + ".assembling.part"))
        let reuse = try await ModelDownloader.assemble(entry: entry, fragmentURLs: sources, in: dir)
        XCTAssertEqual(reuse.disposition, .alreadyPresent)
    }
    func testCancellationPreservesPartialAndResumeValidatesWholePrefix() async throws {
        let (dir, entry, sources, expected) = try fixture()
        defer { try? FileManager.default.removeItem(at: dir) }
        do {
            _ = try await ModelDownloader.assemble(entry: entry, fragmentURLs: sources, in: dir, chunkBytes: 3,
                onProgress: { progress in
                    if progress.completedBytes >= 6 { withUnsafeCurrentTask { $0?.cancel() } }
                })
            XCTFail("cancelled assembly published a result")
        } catch is CancellationError { } catch { XCTFail("unexpected \(error)") }
        let partial = dir.appendingPathComponent("fixture.gguf.assembling.part")
        XCTAssertEqual(try Data(contentsOf: partial), expected.prefix(6))
        XCTAssertFalse(FileManager.default.fileExists(atPath: dir.appendingPathComponent("fixture.gguf").path))
        let result = try await ModelDownloader.assemble(entry: entry, fragmentURLs: sources, in: dir, chunkBytes: 5)
        XCTAssertEqual(try Data(contentsOf: result.fileURL), expected)
    }
    func testWrongPrefixOrCorruptFragmentNeverPublishesFinal() async throws {
        let (dir, entry, sources, _) = try fixture()
        defer { try? FileManager.default.removeItem(at: dir) }
        let partial = dir.appendingPathComponent("fixture.gguf.assembling.part")
        let wrong = Data("bad".utf8)
        try wrong.write(to: partial)
        do {
            _ = try await ModelDownloader.assemble(entry: entry, fragmentURLs: sources, in: dir, chunkBytes: 4)
            XCTFail("wrong prefix accepted")
        } catch ModelDownloader.AssemblyError.incompatiblePrefix { } catch { XCTFail("unexpected \(error)") }
        XCTAssertEqual(try Data(contentsOf: partial), wrong)
        try FileManager.default.removeItem(at: partial)
        var corrupt = try Data(contentsOf: sources[1]); corrupt[0] ^= 1
        try corrupt.write(to: sources[1])
        do {
            _ = try await ModelDownloader.assemble(entry: entry, fragmentURLs: sources, in: dir, chunkBytes: 3)
            XCTFail("corrupt source accepted")
        } catch ModelDownloader.AssemblyError.digestMismatch { } catch { XCTFail("unexpected \(error)") }
        XCTAssertFalse(FileManager.default.fileExists(atPath: dir.appendingPathComponent("fixture.gguf").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: partial.path))
    }
    func testFinalDigestAndExistingFinalAreAlwaysVerified() async throws {
        let (dir, entry, sources, expected) = try fixture()
        defer { try? FileManager.default.removeItem(at: dir) }
        let bad = ModelTarget(id: "bad", file: "bad.gguf", approxGB: 1, note: "test",
                              sha256: String(repeating: "0", count: 64), expectedSizeBytes: Int64(expected.count))
        let recipe = ModelCatalogEntry(id: .deepSeek41Q4, displayName: "bad", profile: .deepSeek41,
            summary: "", artifacts: entry.artifacts, runtimeAvailability: .runnable, assemblyOutput: bad)
        do {
            _ = try await ModelDownloader.assemble(entry: recipe, fragmentURLs: sources, in: dir)
            XCTFail("bad final hash accepted")
        } catch ModelDownloader.AssemblyError.digestMismatch { } catch { XCTFail("unexpected \(error)") }
        XCTAssertFalse(FileManager.default.fileExists(atPath: dir.appendingPathComponent("bad.gguf").path))
        var wrong = expected; wrong[0] ^= 1
        try wrong.write(to: dir.appendingPathComponent("fixture.gguf"))
        do {
            _ = try await ModelDownloader.assemble(entry: entry, fragmentURLs: sources, in: dir)
            XCTFail("existing corrupt final accepted")
        } catch ModelDownloader.AssemblyError.digestMismatch { } catch { XCTFail("unexpected \(error)") }
        XCTAssertEqual(try Data(contentsOf: dir.appendingPathComponent("fixture.gguf")), wrong)
    }
    func testSymlinkStagingCannotModifyAnotherFile() async throws {
        let (dir, entry, sources, _) = try fixture()
        defer { try? FileManager.default.removeItem(at: dir) }
        let victim = dir.appendingPathComponent("user-file")
        let original = Data("keep this".utf8); try original.write(to: victim)
        try FileManager.default.createSymbolicLink(at: dir.appendingPathComponent("fixture.gguf.assembling.part"), withDestinationURL: victim)
        do {
            _ = try await ModelDownloader.assemble(entry: entry, fragmentURLs: sources, in: dir)
            XCTFail("symlink accepted")
        } catch { }
        XCTAssertEqual(try Data(contentsOf: victim), original)
    }
    func testPinnedDS41Q4RecipeAndDownloadOnlyCompanions() throws {
        let entry = try XCTUnwrap(ModelCatalogRegistry.entry(.deepSeek41Q4))
        XCTAssertTrue(entry.isSelectable)
        XCTAssertEqual(entry.primaryArtifact?.file, "DeepSeek-V4.1-Flash-Q4.gguf")
        XCTAssertEqual(entry.expectedSizeBytes, 518_596_067_328)
        XCTAssertEqual(entry.assemblyPeakBytes, 1_037_192_134_656)
        XCTAssertEqual(entry.assemblyOutput?.sha256, "a5e2e2c3ada4b2e98d9f9e4b50f6d9c2a12c2c96f5da165c07e13aff9264984e")
        XCTAssertTrue(entry.artifacts.allSatisfy { $0.role == .splitFragment && $0.file.contains(".part") })
        XCTAssertEqual(NativeModelAccessoryCatalog.entries.count, 8)
        for accessory in NativeModelAccessoryCatalog.entries {
            XCTAssertFalse(accessory.isSelectable); XCTAssertNil(accessory.primaryArtifact)
            XCTAssertNil(ModelCatalogRegistry.entry(accessory.id))
            let target = try XCTUnwrap(accessory.artifacts.first)
            XCTAssertEqual(target.sha256?.count, 64)
            XCTAssertGreaterThan(target.expectedSizeBytes ?? 0, 0)
            XCTAssertNotEqual(target.source.revision, "main")
            XCTAssertEqual(ModelDownloader.target(target.id), target)
        }
    }
}
