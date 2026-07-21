import XCTest
@testable import DS4Engine
import DS4Metal

/// Store-level behavior only — entries are synthetic GKV1 files (the engine
/// wire format's header + padding payload), no Metal engine involved.
final class GLM52DiskKVStoreTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("glm-kv-tests-\(UUID().uuidString)")
    }

    override func tearDownWithError() throws {
        if let directory { try? FileManager.default.removeItem(at: directory) }
    }

    /// GKV1 wire format: magic, version, gguf size, layer count, token
    /// count, token ids, payload.
    private func checkpointData(tokens: [Int32],
                                payloadBytes: Int = 64) -> Data {
        var data = Data()
        func u32(_ value: UInt32) {
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }
        u32(GLM52ResidentModel.kvMagic); u32(1)
        withUnsafeBytes(of: UInt64(42).littleEndian) { data.append(contentsOf: $0) }
        u32(78); u32(UInt32(tokens.count))
        tokens.withUnsafeBufferPointer { data.append(Data(buffer: $0)) }
        data.append(Data(repeating: 0xAB, count: payloadBytes))
        return data
    }

    @discardableResult
    private func storeEntry(_ store: GLM52DiskKVStore,
                            tokens: [Int32]) -> Bool {
        store.store(tokens: tokens) { url in
            try self.checkpointData(tokens: tokens).write(to: url)
        }
    }

    private func tokens(_ range: Range<Int>) -> [Int32] {
        range.map(Int32.init)
    }

    func testStoreAndLongestPrefixLookup() throws {
        let store = try GLM52DiskKVStore(directory: directory, budgetTokens: 0)
        XCTAssertTrue(storeEntry(store, tokens: tokens(0..<80)))
        XCTAssertTrue(storeEntry(store, tokens: tokens(0..<100)))

        let hit = store.findLongestPrefix(of: tokens(0..<120))
        XCTAssertEqual(hit?.tokens.count, 100)
        // Strict prefix: the exact-length entry is not a candidate — the
        // next-shorter stored prefix wins instead.
        XCTAssertEqual(store.findLongestPrefix(of: tokens(0..<100))?
            .tokens.count, 80)
        // Diverging ids do not match.
        var diverged = tokens(0..<120)
        diverged[10] = 9999
        XCTAssertNil(store.findLongestPrefix(of: diverged))
    }

    func testDedupAndMinTokens() throws {
        let store = try GLM52DiskKVStore(directory: directory, budgetTokens: 0)
        XCTAssertTrue(storeEntry(store, tokens: tokens(0..<80)))
        XCTAssertFalse(storeEntry(store, tokens: tokens(0..<80)),
                       "identical tokens must dedup")
        XCTAssertFalse(storeEntry(store, tokens: tokens(0..<10)),
                       "below minTokens must refuse")
    }

    func testShouldStoreInterval() throws {
        let store = try GLM52DiskKVStore(directory: directory, budgetTokens: 0)
        // No stored prefix: any conversation ≥ minTokens qualifies (anchor).
        XCTAssertTrue(store.shouldStore(tokens: tokens(0..<70)))
        XCTAssertTrue(storeEntry(store, tokens: tokens(0..<100)))
        XCTAssertFalse(store.shouldStore(tokens: tokens(0..<100)),
                       "exact entry already stored")
        XCTAssertFalse(store.shouldStore(tokens: tokens(0..<200)),
                       "within storeIntervalTokens of the stored prefix")
        XCTAssertTrue(store.shouldStore(tokens: tokens(0..<360)),
                      "≥ interval beyond the stored prefix")
        // An unrelated conversation is its own anchor.
        XCTAssertTrue(store.shouldStore(tokens: tokens(1000..<1100)))
    }

    func testSupersedeEvictionUnderTokenBudget() throws {
        let store = try GLM52DiskKVStore(directory: directory,
                                         budgetTokens: 400)
        XCTAssertTrue(storeEntry(store, tokens: tokens(0..<100)))
        // Same conversation, 256 tokens later: 100 + 356 > 400 forces an
        // eviction and the supersede rule targets the own shorter prefix.
        XCTAssertTrue(storeEntry(store, tokens: tokens(0..<356)))
        XCTAssertNil(store.findLongestPrefix(of: tokens(0..<356)),
                     "the shorter prefix of the same conversation is evicted")
        XCTAssertEqual(store.findLongestPrefix(of: tokens(0..<400))?
            .tokens.count, 356)
    }

    func testEntryLargerThanTokenBudgetRefused() throws {
        let store = try GLM52DiskKVStore(directory: directory,
                                         budgetTokens: 100)
        XCTAssertFalse(storeEntry(store, tokens: tokens(0..<200)))
    }

    func testAdoptLegacyCheckpoint() throws {
        let store = try GLM52DiskKVStore(directory: directory, budgetTokens: 0)
        let legacy = directory
            .appendingPathComponent(GLM52DiskKVStore.legacyFileName)
        try checkpointData(tokens: tokens(0..<90)).write(to: legacy)
        store.adoptLegacyCheckpoint(at: legacy)
        XCTAssertFalse(FileManager.default.fileExists(atPath: legacy.path))
        XCTAssertEqual(store.findLongestPrefix(of: tokens(0..<120))?
            .tokens.count, 90)
    }

    func testIndexRebuildAfterLoss() throws {
        let first = try GLM52DiskKVStore(directory: directory, budgetTokens: 0)
        XCTAssertTrue(storeEntry(first, tokens: tokens(0..<100)))
        try FileManager.default.removeItem(
            at: directory.appendingPathComponent("index.json"))
        let second = try GLM52DiskKVStore(directory: directory, budgetTokens: 0)
        XCTAssertEqual(second.findLongestPrefix(of: tokens(0..<150))?
            .tokens.count, 100)
    }

    func testForeignFileIgnored() throws {
        let store = try GLM52DiskKVStore(directory: directory, budgetTokens: 0)
        try Data(repeating: 0xFF, count: 128).write(
            to: directory.appendingPathComponent("garbage.glmkv"))
        XCTAssertNil(store.findLongestPrefix(of: tokens(0..<100)))
        XCTAssertTrue(storeEntry(store, tokens: tokens(0..<80)))
        XCTAssertEqual(store.findLongestPrefix(of: tokens(0..<100))?
            .tokens.count, 80)
    }
}
