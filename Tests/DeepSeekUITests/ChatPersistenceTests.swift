import XCTest
@testable import DeepSeekUI

/// Coverage for the disk-IO surface of `ChatPersistence` that doesn't
/// depend on the user's Application Support directory: the static
/// `atomicWrite` helper and the binary `chat.tokens` codec. The
/// schedule/flush + per-chat-folder helpers exercise
/// `PersistencePaths`, which always rebases under the real
/// `~/Library/Application Support/...` — those are covered by an
/// integration test that creates a fresh chat id per run and tears
/// the folder down in tearDown.
final class ChatPersistenceTests: XCTestCase {

    private var tempDir: URL!

    override func setUp() async throws {
        try await super.setUp()
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ChatPersistenceTests-\(UUID().uuidString)",
                                     isDirectory: true)
        try FileManager.default.createDirectory(
            at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDown() async throws {
        if let tempDir, FileManager.default.fileExists(atPath: tempDir.path) {
            try? FileManager.default.removeItem(at: tempDir)
        }
        try await super.tearDown()
    }

    // MARK: atomicWrite

    func testAtomicWrite_createsFile() throws {
        let url = tempDir.appendingPathComponent("a.json")
        let payload = Data("hello".utf8)
        try ChatPersistence.atomicWrite(payload, to: url)
        let read = try Data(contentsOf: url)
        XCTAssertEqual(read, payload)
    }

    func testAtomicWrite_overwritesExistingFile() throws {
        let url = tempDir.appendingPathComponent("b.json")
        try Data("first".utf8).write(to: url)
        try ChatPersistence.atomicWrite(Data("second".utf8), to: url)
        let read = try Data(contentsOf: url)
        XCTAssertEqual(read, Data("second".utf8))
    }

    func testAtomicWrite_createsMissingParentDirectories() throws {
        let url = tempDir
            .appendingPathComponent("nested", isDirectory: true)
            .appendingPathComponent("deeper", isDirectory: true)
            .appendingPathComponent("c.json")
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: url.deletingLastPathComponent().path))
        try ChatPersistence.atomicWrite(Data("ok".utf8), to: url)
        let read = try Data(contentsOf: url)
        XCTAssertEqual(read, Data("ok".utf8))
    }

    func testAtomicWrite_leavesNoTempArtefacts() throws {
        let url = tempDir.appendingPathComponent("d.json")
        try ChatPersistence.atomicWrite(Data("x".utf8), to: url)
        // No `.tmp.…` siblings should remain after a successful
        // write. The atomic helper's `defer` block cleans them up.
        let siblings = try FileManager.default.contentsOfDirectory(
            at: tempDir, includingPropertiesForKeys: nil)
        let stray = siblings.filter { $0.lastPathComponent.contains(".tmp.") }
        XCTAssertTrue(stray.isEmpty, "Stray temp files: \(stray)")
    }

    // MARK: chat.tokens binary codec

    /// `chat.tokens` is a tiny binary file (magic + count + Int32
    /// little-endian payload). This test asserts the on-disk shape
    /// using the public `tokensMagic` constant — we can't go
    /// through `writeChatTokens`/`readChatTokens` here because they
    /// rebase under `PersistencePaths`, which we don't override.
    /// The integration-level read/write is covered in PR 2 once
    /// `ChatStore` wires the layer up.
    func testChatTokens_diskShape() throws {
        let url = tempDir.appendingPathComponent("chat.tokens")
        let tokens: [Int32] = [1, -2, 3_000_000, -3_000_000, 0]
        var data = Data()
        data.append(contentsOf: ChatPersistence.tokensMagic)
        var count = UInt32(tokens.count).littleEndian
        withUnsafeBytes(of: &count) { data.append(contentsOf: $0) }
        for var t in tokens {
            t = t.littleEndian
            withUnsafeBytes(of: &t) { data.append(contentsOf: $0) }
        }
        try data.write(to: url)
        let read = try Data(contentsOf: url)
        XCTAssertEqual(read.count, 8 + tokens.count * 4)
        XCTAssertEqual(Array(read.prefix(4)),
                        ChatPersistence.tokensMagic)
        let storedCount: UInt32 = read.subdata(in: 4..<8)
            .withUnsafeBytes { ptr in
                UInt32(littleEndian: ptr.load(as: UInt32.self))
            }
        XCTAssertEqual(Int(storedCount), tokens.count)
        var decoded: [Int32] = []
        var offset = 8
        for _ in 0..<tokens.count {
            let val: Int32 = read.subdata(in: offset..<(offset + 4))
                .withUnsafeBytes { ptr in
                    Int32(littleEndian: ptr.load(as: Int32.self))
                }
            decoded.append(val)
            offset += 4
        }
        XCTAssertEqual(decoded, tokens)
    }
}
