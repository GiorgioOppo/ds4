import XCTest
import Foundation
import CryptoKit
import DS4Engine

/// Validates the native model downloader's target map + URL/token resolution
/// (no network). Confirms q4-imatrix maps to the exact GGUF filename on disk.
final class ModelDownloaderTests: XCTestCase {
    func testTargetMapAndURL() {
        XCTAssertFalse(ModelDownloader.targets.isEmpty)
        guard let q4 = ModelDownloader.target("q4-imatrix") else { return XCTFail("missing q4-imatrix") }
        XCTAssertEqual(q4.file, "DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf")
        XCTAssertEqual(ModelDownloader.resolveURL(q4.file).absoluteString,
                       "https://huggingface.co/antirez/deepseek-v4-gguf/resolve/main/\(q4.file)")
        XCTAssertNil(ModelDownloader.target("does-not-exist"))
        // ids the GUI catalog uses must all resolve.
        for id in ["q2-imatrix", "q2-q4-imatrix", "q4-imatrix", "pro-q2-imatrix", "mtp"] {
            XCTAssertNotNil(ModelDownloader.target(id), "GUI target \(id) not in downloader")
        }
    }

    func testTokenResolution() {
        // explicit wins; nil when none provided and no env/cache (don't assert the
        // cache path — just that explicit passes through).
        XCTAssertEqual(ModelDownloader.resolveToken("abc123"), "abc123")
    }

    /// Content-integrity primitive: the streaming SHA-256 must match the canonical
    /// digest. SHA-256("abc") is a fixed known-answer vector.
    func testSHA256KnownAnswer() throws {
        let f = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ds4-sha-\(UUID().uuidString).bin")
        try Data("abc".utf8).write(to: f)
        defer { try? FileManager.default.removeItem(at: f) }
        XCTAssertEqual(try ModelDownloader.sha256Hex(of: f),
                       "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    }

    /// The chunked read must produce the same digest as a one-shot hash, even when
    /// the data spans many chunk boundaries (tiny chunk forces the loop).
    func testSHA256MultiChunkMatchesOneShot() throws {
        let blob = Data((0..<5000).map { UInt8($0 & 0xff) })
        let f = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ds4-sha-\(UUID().uuidString).bin")
        try blob.write(to: f)
        defer { try? FileManager.default.removeItem(at: f) }
        let oneShot = SHA256.hash(data: blob).map { String(format: "%02x", $0) }.joined()
        XCTAssertEqual(try ModelDownloader.sha256Hex(of: f, chunk: 64), oneShot)
    }
}
