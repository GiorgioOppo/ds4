import XCTest
@testable import DS4Engine
import DS4Metal

/// Store behavior over synthetic LKV1 headers; no Metal/model is required.
final class LagunaDiskKVStoreTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("laguna-kv-tests-\(UUID().uuidString)")
    }

    override func tearDownWithError() throws {
        if let directory {
            try? FileManager.default.removeItem(at: directory)
        }
    }

    private func options() -> LagunaDiskKVStore.Options {
        var options = LagunaDiskKVStore.Options()
        options.minTokens = 4
        options.storeIntervalTokens = 8
        return options
    }

    private func checkpoint(tokens: [Int32]) -> Data {
        var data = Data()
        func u32(_ value: UInt32) {
            withUnsafeBytes(of: value.littleEndian) {
                data.append(contentsOf: $0)
            }
        }
        u32(LagunaResidentModel.kvMagic); u32(1)
        withUnsafeBytes(of: UInt64(42).littleEndian) {
            data.append(contentsOf: $0)
        }
        u32(48); u32(UInt32(tokens.count))
        u32(1_024); u32(512); u32(36); u32(0)
        tokens.withUnsafeBufferPointer { data.append(Data(buffer: $0)) }
        data.append(Data(repeating: 0xA5, count: 64))
        return data
    }

    @discardableResult
    private func writeEntry(_ store: LagunaDiskKVStore,
                            _ tokens: [Int32]) -> Bool {
        store.store(tokens: tokens) { url in
            try self.checkpoint(tokens: tokens).write(to: url)
        }
    }

    private func ids(_ range: Range<Int>) -> [Int32] {
        range.map(Int32.init)
    }

    func testLongestPrefixDedupAndInterval() throws {
        let store = try LagunaDiskKVStore(
            directory: directory, budgetTokens: 0, options: options())
        XCTAssertTrue(writeEntry(store, ids(0..<8)))
        XCTAssertTrue(writeEntry(store, ids(0..<16)))
        XCTAssertFalse(writeEntry(store, ids(0..<16)),
                       "exact ids must dedup")
        XCTAssertEqual(
            store.findLongestPrefix(of: ids(0..<24))?.tokens.count, 16)
        XCTAssertFalse(store.shouldStore(tokens: ids(0..<20)))
        XCTAssertTrue(store.shouldStore(tokens: ids(0..<24)))

        var diverged = ids(0..<24)
        diverged[2] = 9_999
        XCTAssertNil(store.findLongestPrefix(of: diverged))
    }

    func testSupersededPrefixEvictionAndIndexRebuild() throws {
        let first = try LagunaDiskKVStore(
            directory: directory, budgetTokens: 24, options: options())
        XCTAssertTrue(writeEntry(first, ids(0..<8)))
        XCTAssertTrue(writeEntry(first, ids(0..<24)))
        XCTAssertNil(first.findLongestPrefix(of: ids(0..<24)),
                     "the shorter same-conversation prefix is evicted")

        try FileManager.default.removeItem(
            at: directory.appendingPathComponent("index.json"))
        let rebuilt = try LagunaDiskKVStore(
            directory: directory, budgetTokens: 24, options: options())
        XCTAssertEqual(
            rebuilt.findLongestPrefix(of: ids(0..<32))?.tokens.count, 24)
    }

    func testForeignAndOverBudgetEntriesAreRejected() throws {
        let store = try LagunaDiskKVStore(
            directory: directory, budgetTokens: 8, options: options())
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        try Data(repeating: 0xFF, count: 80).write(
            to: directory.appendingPathComponent("foreign.lagunakv"))
        XCTAssertNil(store.findLongestPrefix(of: ids(0..<16)))
        XCTAssertFalse(writeEntry(store, ids(0..<9)))
        XCTAssertTrue(writeEntry(store, ids(0..<8)))
    }
}
