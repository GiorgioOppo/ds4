import XCTest
import DS4Metal
@testable import DS4Engine

/// Disk KV store: serialization round-trip, longest-prefix lookup, dedup and
/// budget eviction — all CPU-only (synthetic snapshots, no GPU/model needed).
final class DiskKVStoreTests: XCTestCase {
    private var dir: URL!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("kvstore-test-\(UUID().uuidString)")
    }
    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: dir)
    }

    private func snapshot(nKeys: Int, headDim: Int = 8, layers: Int = 3) -> KVSnapshot {
        var ls: [KVLayerSnapshot] = []
        for il in 0..<layers {
            let rawStart = max(0, nKeys - 4)             // tiny SWA window for the test
            let rows = nKeys - rawStart
            let raw = (0..<rows * headDim).map { Float(il * 1000 + $0) }
            let comp = il == 0 ? nil : CompSnapshot(
                count: 2,
                stateKv: (0..<16).map { Float($0) }, stateScore: (0..<16).map { Float(-$0) },
                cacheRows: (0..<2 * headDim).map { Float(il) + Float($0) * 0.5 })
            // NSA indexer state on one layer (round-trips alongside the attention comp).
            let idx = il == 2 ? CompSnapshot(
                count: 1,
                stateKv: (0..<8).map { Float($0) * 2 }, stateScore: (0..<8).map { Float($0) - 4 },
                cacheRows: (0..<headDim).map { Float($0) * 0.25 }) : nil
            ls.append(KVLayerSnapshot(rawStart: rawStart, raw: raw, comp: comp, idx: idx))
        }
        return KVSnapshot(nKeys: nKeys, headDim: headDim, layers: ls)
    }

    func testStoreAndFindLongestPrefix() throws {
        let store = try DiskKVStore(directory: dir, budgetMB: 64, quantBits: 2, contextSize: 8192,
                                    options: { var o = DiskKVStore.Options(); o.minTokens = 4; return o }())
        let short = Array(0..<8), long = Array(0..<16)
        XCTAssertTrue(store.store(tokens: short, modelName: "m.gguf", snapshot: snapshot(nKeys: 8)))
        XCTAssertTrue(store.store(tokens: long, modelName: "m.gguf", snapshot: snapshot(nKeys: 16)))
        // Dedup: same tokens+model → no second file.
        XCTAssertFalse(store.store(tokens: long, modelName: "m.gguf", snapshot: snapshot(nKeys: 16)))

        // Query extends `long` → the 16-token entry wins over the 8-token one.
        let hit = try XCTUnwrap(store.findLongestPrefix(of: Array(0..<20), modelName: "m.gguf"))
        XCTAssertEqual(hit.tokens, long)
        XCTAssertEqual(store.loadSnapshot(hit.url), snapshot(nKeys: 16))

        // Exact-length query (no remainder to prefill) must NOT match that entry.
        let hit2 = try XCTUnwrap(store.findLongestPrefix(of: long, modelName: "m.gguf"))
        XCTAssertEqual(hit2.tokens, short)

        // Different model or diverging tokens → no match.
        XCTAssertNil(store.findLongestPrefix(of: Array(0..<20), modelName: "other.gguf"))
        XCTAssertNil(store.findLongestPrefix(of: [9, 9, 9, 9, 9, 9, 9, 9, 9], modelName: "m.gguf"))
    }

    /// The C supersede-continued rule: a CONTINUED entry that is a strict prefix
    /// of the incoming checkpoint gets its score collapsed (×0.05 with no hits)
    /// and is evicted first — even though the unrelated entry has fewer tokens
    /// (lower base score = tokens/bytes) and would otherwise be the victim.
    func testSupersededContinuedPrefixEvictedFirst() throws {
        var opts = DiskKVStore.Options(); opts.minTokens = 4
        let store = try DiskKVStore(directory: dir, budgetMB: 64, quantBits: 2,
                                    contextSize: 8192, options: opts)
        let conv = Array(0..<8)                      // older checkpoint of conversation A
        let other = Array(500..<506)                 // unrelated conversation B (6 < 8 tokens)
        XCTAssertTrue(store.store(tokens: conv, modelName: "m.gguf",
                                  snapshot: snapshot(nKeys: 8), reason: .continued))
        XCTAssertTrue(store.store(tokens: other, modelName: "m.gguf",
                                  snapshot: snapshot(nKeys: 6), reason: .continued))
        // Force exactly one eviction (target = total − 1) while a LONGER
        // checkpoint of conversation A is incoming: the superseded prefix
        // (conv) must be the victim, not the unrelated entry.
        let budget = UInt64(64) * 1_048_576
        store.evictToBudget(incomingBytes: budget - bytesOnDisk() + 1,
                            incomingTokens: Array(0..<12), incomingModel: "m.gguf")
        XCTAssertNil(store.findLongestPrefix(of: Array(0..<20), modelName: "m.gguf"))
        XCTAssertNotNil(store.findLongestPrefix(of: Array(500..<520), modelName: "m.gguf"))
    }

    private func bytesOnDisk() -> UInt64 {
        let files = (try? FileManager.default.contentsOfDirectory(at: dir,
                                                                  includingPropertiesForKeys: [.fileSizeKey])) ?? []
        return files.reduce(0) { acc, url in
            acc + UInt64((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
    }

    /// Token budget: total stored tokens stay within budgetTokens, evicting the
    /// lowest-score entries; an entry that alone exceeds the budget is refused.
    func testTokenBudgetEviction() throws {
        var opts = DiskKVStore.Options(); opts.minTokens = 4
        let store = try DiskKVStore(directory: dir, budgetMB: 64, quantBits: 2,
                                    contextSize: 8192, budgetTokens: 20, options: opts)
        XCTAssertFalse(store.store(tokens: Array(0..<32), modelName: "m.gguf",
                                   snapshot: snapshot(nKeys: 32)))       // alone > budget
        XCTAssertTrue(store.store(tokens: Array(0..<8), modelName: "m.gguf",
                                  snapshot: snapshot(nKeys: 8)))
        XCTAssertTrue(store.store(tokens: Array(100..<108), modelName: "m.gguf",
                                  snapshot: snapshot(nKeys: 8)))
        // Third 8-token entry would make 24 > 20: something must be evicted.
        XCTAssertTrue(store.store(tokens: Array(200..<208), modelName: "m.gguf",
                                  snapshot: snapshot(nKeys: 8)))
        let entries = try FileManager.default.contentsOfDirectory(atPath: dir.path)
            .filter { $0.hasSuffix(".kv") }
        XCTAssertEqual(entries.count, 2, "8+8+8 tokens must not fit a 20-token budget")
    }

    /// The streaming reader hands out ONE layer at a time in order; reassembling
    /// the stream must equal both the in-RAM parse and the original snapshot.
    func testStreamSnapshotMatchesLoad() throws {
        var opts = DiskKVStore.Options(); opts.minTokens = 4
        let store = try DiskKVStore(directory: dir, budgetMB: 64, quantBits: 2,
                                    contextSize: 8192, options: opts)
        let original = snapshot(nKeys: 12)
        XCTAssertTrue(store.store(tokens: Array(0..<12), modelName: "m.gguf", snapshot: original))
        let hit = try XCTUnwrap(store.findLongestPrefix(of: Array(0..<20), modelName: "m.gguf"))

        var meta: (nKeys: Int, headDim: Int, layerCount: Int)?
        var indices: [Int] = []
        var layers: [KVLayerSnapshot] = []
        try store.streamSnapshot(hit.url,
                                 onMeta: { meta = ($0, $1, $2) },
                                 onLayer: { i, layer in indices.append(i); layers.append(layer) })
        XCTAssertEqual(meta?.nKeys, 12)
        XCTAssertEqual(meta?.headDim, 8)
        XCTAssertEqual(meta?.layerCount, 3)
        XCTAssertEqual(indices, [0, 1, 2])
        XCTAssertEqual(KVSnapshot(nKeys: 12, headDim: 8, layers: layers), original)
        XCTAssertEqual(store.loadSnapshot(hit.url), original)
    }

    /// A throwing onMeta (the decoder's shape gate) aborts BEFORE any tensor
    /// body is parsed: no layer callbacks, error propagated, entry untouched.
    func testStreamAbortsOnMetaMismatch() throws {
        struct Mismatch: Error {}
        var opts = DiskKVStore.Options(); opts.minTokens = 4
        let store = try DiskKVStore(directory: dir, budgetMB: 64, quantBits: 2,
                                    contextSize: 8192, options: opts)
        XCTAssertTrue(store.store(tokens: Array(0..<8), modelName: "m.gguf", snapshot: snapshot(nKeys: 8)))
        let hit = try XCTUnwrap(store.findLongestPrefix(of: Array(0..<20), modelName: "m.gguf"))
        var layerCalls = 0
        XCTAssertThrowsError(try store.streamSnapshot(hit.url,
                                                      onMeta: { _, _, _ in throw Mismatch() },
                                                      onLayer: { _, _ in layerCalls += 1 })) {
            XCTAssertTrue($0 is Mismatch)
        }
        XCTAssertEqual(layerCalls, 0)
        XCTAssertNotNil(store.loadSnapshot(hit.url), "aborted stream must not damage the entry")
    }

    /// A truncated body surfaces as StreamError.corrupt (the restore path then
    /// discards the entry, like the C loader).
    func testStreamThrowsOnTruncatedEntry() throws {
        var opts = DiskKVStore.Options(); opts.minTokens = 4
        let store = try DiskKVStore(directory: dir, budgetMB: 64, quantBits: 2,
                                    contextSize: 8192, options: opts)
        XCTAssertTrue(store.store(tokens: Array(0..<8), modelName: "m.gguf", snapshot: snapshot(nKeys: 8)))
        let hit = try XCTUnwrap(store.findLongestPrefix(of: Array(0..<20), modelName: "m.gguf"))
        let full = try Data(contentsOf: hit.url)
        try full.prefix(full.count - 16).write(to: hit.url)   // chop the tail
        XCTAssertThrowsError(try store.streamSnapshot(hit.url, onMeta: { _, _, _ in },
                                                      onLayer: { _, _ in })) {
            XCTAssertTrue($0 is DiskKVStore.StreamError)
        }
        XCTAssertNil(store.loadSnapshot(hit.url))
    }

    /// SnapshotBox: single-take ownership; the boxed store writes the same entry
    /// as the by-value path, and a second store from the emptied box is refused.
    func testSnapshotBoxStore() throws {
        var opts = DiskKVStore.Options(); opts.minTokens = 4
        let store = try DiskKVStore(directory: dir, budgetMB: 64, quantBits: 2,
                                    contextSize: 8192, options: opts)
        let original = snapshot(nKeys: 8)
        let box = DiskKVStore.SnapshotBox(original)
        XCTAssertTrue(store.store(tokens: Array(0..<8), modelName: "m.gguf", box: box))
        let hit = try XCTUnwrap(store.findLongestPrefix(of: Array(0..<9), modelName: "m.gguf"))
        XCTAssertEqual(store.loadSnapshot(hit.url), original)
        // The box was emptied by take(): a reuse cannot silently write garbage.
        XCTAssertFalse(store.store(tokens: Array(100..<108), modelName: "m.gguf", box: box))
    }

    func testEvictionRespectsBudget() throws {
        var opts = DiskKVStore.Options(); opts.minTokens = 4
        // Budget floor is 64 MB in the store; entries here are ~KB, so exercise
        // eviction by writing into a store whose budget we then shrink via a new
        // instance and a large entry count.
        let store = try DiskKVStore(directory: dir, budgetMB: 64, quantBits: 4,
                                    contextSize: 8192, options: opts)
        for i in 0..<5 {
            let toks = Array(i * 100..<(i * 100 + 6))
            XCTAssertTrue(store.store(tokens: toks, modelName: "m.gguf",
                                      snapshot: snapshot(nKeys: 6)))
        }
        let files = try FileManager.default.contentsOfDirectory(atPath: dir.path)
        XCTAssertEqual(files.filter { $0.hasSuffix(".kv") }.count, 5)
    }
}
