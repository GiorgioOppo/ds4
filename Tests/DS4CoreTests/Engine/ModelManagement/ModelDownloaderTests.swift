import XCTest
import Foundation
import CryptoKit
@testable import DS4Engine

/// Pure/synthetic coverage for the native model catalog and downloader. These
/// tests never contact Hugging Face.
final class ModelDownloaderTests: XCTestCase {
    func testTargetMapAndURL() {
        XCTAssertFalse(ModelDownloader.targets.isEmpty)
        guard let q4 = ModelDownloader.target("q4-imatrix") else {
            return XCTFail("missing q4-imatrix")
        }
        XCTAssertEqual(
            q4.file,
            "DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
        )
        XCTAssertEqual(
            ModelDownloader.resolveURL(q4).absoluteString,
            "https://huggingface.co/antirez/deepseek-v4-gguf/resolve/main/\(q4.file)"
        )
        XCTAssertNil(ModelDownloader.target("does-not-exist"))

        for target in ModelCatalogRegistry.allArtifacts {
            XCTAssertEqual(ModelDownloader.target(target.id), target)
        }
        XCTAssertEqual(ModelDownloader.target("mtp")?.role, .optionalComponent)
    }

    func testTypedCatalogSelectsFlashAndSingleFileProQ2() {
        XCTAssertEqual(DeepSeekV4ModelCatalog.entries.count, 5)
        XCTAssertEqual(GLM52ModelCatalog.entries.count, 3)
        XCTAssertEqual(ModelCatalogRegistry.entries.count, 8)
        XCTAssertEqual(
            Set(ModelCatalogRegistry.selectableEntries.map(\.id)),
            Set([.flashQ2Imatrix, .flashQ2Q4Imatrix, .flashQ4Imatrix, .proQ2Imatrix])
        )

        let proQ2 = DeepSeekV4ModelCatalog.entry(.proQ2Imatrix)
        XCTAssertEqual(proQ2?.profile, .deepSeekV4(.pro))
        XCTAssertTrue(proQ2?.isSelectable ?? false)
        XCTAssertTrue(proQ2?.runtimeAvailability.isRunnable ?? false)
        XCTAssertNil(proQ2?.runtimeAvailability.unavailableReason)
        XCTAssertEqual(proQ2?.artifacts.count, 1)
        XCTAssertEqual(proQ2?.primaryArtifact?.role, .mainModel)

        let split = DeepSeekV4ModelCatalog.entry(.proQ4Split)
        XCTAssertFalse(split?.isSelectable ?? true)
        XCTAssertFalse(split?.runtimeAvailability.isRunnable ?? true)
        XCTAssertNotNil(split?.runtimeAvailability.unavailableReason)
        XCTAssertEqual(split?.artifacts.count, 2)
        XCTAssertTrue(split?.artifacts.allSatisfy { $0.role == .distributedShard } ?? false)
        XCTAssertFalse(
            DeepSeekV4ModelCatalog.entries
                .flatMap(\.artifacts)
                .contains { $0.id == DeepSeekV4AccessoryCatalog.mtp.id },
            "MTP is an accessory, not a main-model catalog entry"
        )

        let artifacts = ModelCatalogRegistry.allArtifacts
        XCTAssertEqual(Set(artifacts.map(\.id)).count, artifacts.count)
        XCTAssertEqual(Set(artifacts.map(\.file)).count, artifacts.count)
        XCTAssertTrue(artifacts.allSatisfy { target in
            guard let digest = target.sha256, digest.count == 64 else { return false }
            return digest.allSatisfy { $0.isHexDigit && !$0.isUppercase }
        }, "every downloadable main-model artifact must have a pinned lowercase SHA-256")
    }

    func testGLM52CatalogIsPinnedDownloadOnlyAndUsesItsOwnRepository() throws {
        let expected: [(ModelCatalogID, String, Int64, String)] = [
            (
                .glm52IQ2XXS,
                "GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf",
                211_075_856_448,
                "a49de64c5020432bdae23de36a423a9660a5621bc0db8d12b66bd8814b07fea0"
            ),
            (
                .glm52Q2K,
                "GLM-5.2-UD-Q2_K_RoutedQ2K.gguf",
                262_036_650_048,
                "b9fa49d010dad35b96418c45831c212a746715b0646c1121ccfc414455bd6fe5"
            ),
            (
                .glm52Q4K,
                "GLM-5.2-UD-Q4_K_RoutedQ4K.gguf",
                434_170_886_208,
                "7160879c87756236eea16ec6bfeb19288d16fa94dcfcef3a5ed5f38b1383d3a5"
            ),
        ]

        for (id, file, byteCount, digest) in expected {
            let entry = try XCTUnwrap(ModelCatalogRegistry.entry(id))
            let target = try XCTUnwrap(entry.artifacts.first)
            XCTAssertEqual(entry.profile, .glm52)
            XCTAssertFalse(entry.isSelectable)
            XCTAssertFalse(entry.runtimeAvailability.isRunnable)
            XCTAssertEqual(entry.expectedSizeBytes, byteCount)
            XCTAssertEqual(target.file, file)
            XCTAssertEqual(target.expectedSizeBytes, byteCount)
            XCTAssertEqual(target.sha256, digest)
            XCTAssertEqual(target.source, .glm52)
            XCTAssertEqual(
                ModelDownloader.resolveURL(target).absoluteString,
                "https://huggingface.co/antirez/glm-5.2-gguf/resolve/" +
                    "2638b3b878f5c6cc3ae7334b8dbea1275025f52e/\(file)"
            )
            XCTAssertEqual(
                ModelDownloader.resolveURL(file),
                ModelDownloader.resolveURL(target),
                "the legacy filename resolver must retain the target repository"
            )
        }

        XCTAssertTrue(GLM52ModelCatalog.entries.allSatisfy { !$0.isSelectable })
        XCTAssertTrue(
            Set(GLM52ModelCatalog.entries.map(\.id))
                .isDisjoint(with: Set(DeepSeekV4ModelCatalog.entries.map(\.id)))
        )
    }

    func testTokenResolution() {
        XCTAssertEqual(ModelDownloader.resolveToken("abc123"), "abc123")
        XCTAssertEqual(ModelDownloader.resolveToken("  abc123\n"), "abc123")
    }

    func testExistingNonEmptyFileReturnsAlreadyPresentWithoutNetwork() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let target = ModelTarget(
            id: "synthetic",
            file: "synthetic.gguf",
            approxGB: 1,
            note: "test only",
            // Deliberately wrong: already-installed GGUFs are not re-read, which
            // keeps GUI startup instant for files of hundreds of gigabytes.
            sha256: String(repeating: "0", count: 64)
        )
        let final = directory.appendingPathComponent(target.file)
        let bytes = Data("already installed".utf8)
        try bytes.write(to: final)
        let recorder = DownloadStateRecorder()

        let result = try await ModelDownloader.acquire(
            target: target,
            ggufDirectory: directory,
            onProgress: { recorder.record(progress: $0) },
            onState: { recorder.record(state: $0) }
        )

        XCTAssertEqual(result.disposition, .alreadyPresent)
        XCTAssertEqual(result.fileURL, final.standardizedFileURL)
        XCTAssertEqual(result.byteCount, Int64(bytes.count))
        XCTAssertEqual(try Data(contentsOf: final), bytes)
        XCTAssertFalse(FileManager.default.fileExists(atPath: final.path + ".part"))
        XCTAssertEqual(recorder.states, [.checkingLocalFile, .completed(.alreadyPresent)])
        XCTAssertEqual(recorder.progress.last?.fractionCompleted, 1)
    }

    func testExistingPinnedFileWithWrongSizeIsRejectedWithoutNetwork() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let target = ModelTarget(
            id: "pinned-size",
            file: "pinned-size.gguf",
            approxGB: 1,
            note: "test only",
            expectedSizeBytes: 4
        )
        try Data([1, 2, 3]).write(to: directory.appendingPathComponent(target.file))

        do {
            _ = try await ModelDownloader.acquire(target: target, ggufDirectory: directory)
            XCTFail("a wrong-size final file must not be reused")
        } catch let error as ModelDownloader.DownloadError {
            guard case .existingFileSizeMismatch(let expected, let actual) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(expected, 4)
            XCTAssertEqual(actual, 3)
        }
    }

    func testExistingPinnedFileWithExactSizeReturnsAlreadyPresentWithoutNetwork() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let bytes = Data([1, 2, 3, 4])
        let target = ModelTarget(
            id: "pinned-size-match",
            file: "pinned-size-match.gguf",
            approxGB: 1,
            note: "test only",
            expectedSizeBytes: Int64(bytes.count),
            source: .init(repository: "invalid.example/repository", revision: "missing")
        )
        let final = directory.appendingPathComponent(target.file)
        try bytes.write(to: final)
        let recorder = DownloadStateRecorder()

        let result = try await ModelDownloader.acquire(
            target: target,
            ggufDirectory: directory,
            onProgress: { recorder.record(progress: $0) },
            onState: { recorder.record(state: $0) }
        )

        XCTAssertEqual(result.disposition, .alreadyPresent)
        XCTAssertEqual(result.fileURL, final.standardizedFileURL)
        XCTAssertEqual(result.byteCount, Int64(bytes.count))
        XCTAssertEqual(recorder.states, [.checkingLocalFile, .completed(.alreadyPresent)])
        XCTAssertFalse(FileManager.default.fileExists(atPath: final.path + ".part"))
    }

    func testLocalStateDistinguishesMissingEmptyAndPresent() throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let target = ModelTarget(
            id: "state", file: "state.gguf", approxGB: 0, note: "test only"
        )
        let file = directory.appendingPathComponent(target.file)

        XCTAssertEqual(try ModelDownloader.localState(target: target, in: directory), .missing)
        XCTAssertTrue(FileManager.default.createFile(atPath: file.path, contents: nil))
        XCTAssertEqual(try ModelDownloader.localState(target: target, in: directory), .empty)
        try Data([1, 2, 3]).write(to: file)
        XCTAssertEqual(
            try ModelDownloader.localState(target: target, in: directory),
            .present(byteCount: 3)
        )
    }

    func testUnsafeCatalogPathIsRejectedBeforeFilesystemOrNetworkAccess() throws {
        let directory = temporaryDirectory()
        for unsafe in ["../escape.gguf", "nested/model.gguf", "nested\\model.gguf", ".", ".."] {
            let target = ModelTarget(id: "unsafe", file: unsafe, approxGB: 0, note: "test")
            XCTAssertThrowsError(try ModelDownloader.destinationURL(for: target, in: directory)) {
                guard case ModelDownloader.DownloadError.unsafeFileName = $0 else {
                    return XCTFail("unexpected error for \(unsafe): \($0)")
                }
            }
        }
    }

    func testLocalStateRejectsSymbolicLinkEvenWhenItsTargetExists() throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let outside = directory.appendingPathComponent("outside.bin")
        try Data([1]).write(to: outside)
        let target = ModelTarget(
            id: "symlink", file: "linked.gguf", approxGB: 0, note: "test"
        )
        try FileManager.default.createSymbolicLink(
            at: directory.appendingPathComponent(target.file),
            withDestinationURL: outside
        )

        XCTAssertThrowsError(try ModelDownloader.localState(target: target, in: directory)) {
            guard case ModelDownloader.DownloadError.existingItemNotRegular = $0 else {
                return XCTFail("unexpected error: \($0)")
            }
        }
    }

    func testRequiredFreeSpaceAccountsForPartAndMargin() {
        XCTAssertEqual(
            ModelDownloader.requiredFreeSpace(
                totalBytes: 100,
                existingPartialBytes: 40,
                minimumMarginBytes: 10
            ),
            70
        )
        XCTAssertEqual(
            ModelDownloader.requiredFreeSpace(
                totalBytes: 100,
                existingPartialBytes: 150,
                minimumMarginBytes: 10
            ),
            10
        )
    }

    func testSpacePreflightDoesNotBlockACompletePinnedPartial() {
        XCTAssertTrue(
            ModelDownloader.shouldPreflightSpace(
                partialBytes: 0,
                expectedSizeBytes: 100
            )
        )
        XCTAssertTrue(
            ModelDownloader.shouldPreflightSpace(
                partialBytes: 99,
                expectedSizeBytes: 100
            )
        )
        XCTAssertFalse(
            ModelDownloader.shouldPreflightSpace(
                partialBytes: 100,
                expectedSizeBytes: 100
            )
        )
        XCTAssertFalse(
            ModelDownloader.shouldPreflightSpace(
                partialBytes: 101,
                expectedSizeBytes: 100
            )
        )
        XCTAssertFalse(
            ModelDownloader.shouldPreflightSpace(
                partialBytes: 50,
                expectedSizeBytes: nil
            )
        )
    }

    func testLocalizedErrorsRemainReadableToSwiftUI() {
        let error = ModelDownloader.DownloadError.http(401)
        XCTAssertEqual(error.localizedDescription, error.description)
        XCTAssertTrue(error.localizedDescription.contains("token"))
    }

    /// SHA-256("abc") is a fixed known-answer vector.
    func testSHA256KnownAnswer() throws {
        let file = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ds4-sha-\(UUID().uuidString).bin")
        try Data("abc".utf8).write(to: file)
        defer { try? FileManager.default.removeItem(at: file) }
        XCTAssertEqual(
            try ModelDownloader.sha256Hex(of: file),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        )
    }

    func testSHA256MultiChunkMatchesOneShot() throws {
        let blob = Data((0..<5000).map { UInt8($0 & 0xff) })
        let file = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ds4-sha-\(UUID().uuidString).bin")
        try blob.write(to: file)
        defer { try? FileManager.default.removeItem(at: file) }
        let oneShot = SHA256.hash(data: blob).map { String(format: "%02x", $0) }.joined()
        XCTAssertEqual(try ModelDownloader.sha256Hex(of: file, chunk: 64), oneShot)
    }

    func testSHA256CapsAnOversizedRequestedChunk() throws {
        let file = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ds4-sha-cap-\(UUID().uuidString).bin")
        XCTAssertTrue(FileManager.default.createFile(atPath: file.path, contents: nil))
        defer { try? FileManager.default.removeItem(at: file) }

        let handle = try FileHandle(forWritingTo: file)
        let oneMiB = Data(repeating: 0xa5, count: 1 << 20)
        for _ in 0..<9 { try handle.write(contentsOf: oneMiB) }
        try handle.close()

        var checkpoints: [Int64] = []
        _ = try ModelDownloader.sha256Hex(of: file, chunk: .max) {
            checkpoints.append($0)
        }

        XCTAssertEqual(checkpoints.last, 9 << 20)
        var previous: Int64 = 0
        for checkpoint in checkpoints {
            XCTAssertLessThanOrEqual(
                checkpoint - previous,
                Int64(ModelDownloader.maximumSHA256ChunkBytes)
            )
            previous = checkpoint
        }
    }

    func testDownloadTransportDisablesCachesAndSerializesBodyCallbacks() {
        let configuration = HTTPRangeFileTransfer.makeSessionConfiguration()
        XCTAssertNil(configuration.urlCache)
        XCTAssertNil(configuration.urlCredentialStorage)
        XCTAssertNil(configuration.httpCookieStorage)
        XCTAssertFalse(configuration.httpShouldSetCookies)
        XCTAssertEqual(configuration.requestCachePolicy, .reloadIgnoringLocalCacheData)
        XCTAssertEqual(configuration.httpMaximumConnectionsPerHost, 1)

        let queue = HTTPRangeFileTransfer.makeDelegateQueue()
        XCTAssertEqual(queue.maxConcurrentOperationCount, 1)
        XCTAssertNotNil(queue.underlyingQueue)
    }

    func testResumeMetadataRejectsOversizedFileWithoutDecodingIt() throws {
        let file = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ds4-resume-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: file) }
        try Data(repeating: 0x20, count: Int(ResumeMetadata.maximumEncodedBytes) + 1)
            .write(to: file)
        XCTAssertNil(ResumeMetadata.load(from: file))
    }

    func testResumeMetadataStillLoadsWithinBound() throws {
        let file = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ds4-resume-valid-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: file) }

        ResumeMetadata(validator: "test-etag", totalBytes: 123).save(to: file)
        let loaded = ResumeMetadata.load(from: file)
        XCTAssertEqual(loaded?.validator, "test-etag")
        XCTAssertEqual(loaded?.totalBytes, 123)
    }

    func testResumeMetadataDoesNotWriteOversizedValidator() {
        let file = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ds4-resume-write-cap-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: file) }

        ResumeMetadata(
            validator: String(repeating: "x", count: ResumeMetadata.maximumEncodedBytes),
            totalBytes: 123
        ).save(to: file)
        XCTAssertFalse(FileManager.default.fileExists(atPath: file.path))
    }

    func testDownloadFileDescriptorCanBypassUnifiedCache() throws {
        let file = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ds4-nocache-\(UUID().uuidString).bin")
        XCTAssertTrue(FileManager.default.createFile(atPath: file.path, contents: nil))
        defer { try? FileManager.default.removeItem(at: file) }

        let handle = try FileHandle(forWritingTo: file)
        defer { try? handle.close() }
        XCTAssertTrue(ModelDownloader.enableUncachedIO(for: handle))
    }

    private func temporaryDirectory() -> URL {
        URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("ds4-download-\(UUID().uuidString)", isDirectory: true)
    }
}

private final class DownloadStateRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var storedStates: [ModelDownloadState] = []
    private var storedProgress: [ModelDownloadProgress] = []

    var states: [ModelDownloadState] { lock.withLock { storedStates } }
    var progress: [ModelDownloadProgress] { lock.withLock { storedProgress } }

    func record(state: ModelDownloadState) {
        lock.withLock { storedStates.append(state) }
    }

    func record(progress: ModelDownloadProgress) {
        lock.withLock { storedProgress.append(progress) }
    }
}
