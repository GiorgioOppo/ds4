import XCTest
@testable import DeepSeekUI

/// The pinning LRU that keeps recently-viewed round payloads warm.
/// Capacity is the only knob; eviction skips pinned entries and uses
/// access-time as the tiebreaker.
final class RoundLRUCacheTests: XCTestCase {

    private func key(_ i: Int) -> RoundKey {
        RoundKey(chatID: UUID(uuid: (UInt8(i), 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0)),
                  turnID: UUID(),
                  roundID: UUID())
    }

    private func round(_ i: Int) -> StoredRound {
        StoredRound(id: UUID(), roundIndex: i, content: "r\(i)")
    }

    func testEmptyCache_returnsNil() {
        let cache = RoundLRUCache(capacity: 2)
        XCTAssertFalse(cache.contains(key(0)))
    }

    func testPutAndGet_returnsValue() {
        var cache = RoundLRUCache(capacity: 2)
        let k = key(0); let v = round(0)
        cache.put(k, v)
        XCTAssertEqual(cache.get(k), v)
    }

    func testCapacityEnforced_evictsLRU() {
        var cache = RoundLRUCache(capacity: 2)
        let k0 = key(0), k1 = key(1), k2 = key(2)
        cache.put(k0, round(0))
        cache.put(k1, round(1))
        // Touch k0 so k1 becomes the LRU.
        _ = cache.get(k0)
        cache.put(k2, round(2))
        XCTAssertNotNil(cache.get(k0))
        XCTAssertNil(cache.get(k1), "k1 should have been evicted as LRU")
        XCTAssertNotNil(cache.get(k2))
        XCTAssertEqual(cache.count, 2)
    }

    func testPinnedKey_isNotEvicted() {
        var cache = RoundLRUCache(capacity: 2)
        let k0 = key(0), k1 = key(1), k2 = key(2)
        cache.put(k0, round(0))
        cache.pin(k0)
        cache.put(k1, round(1))
        cache.put(k2, round(2))
        // k0 was eligible by age (oldest) but pinned, so k1 went
        // instead.
        XCTAssertNotNil(cache.get(k0))
        XCTAssertNil(cache.get(k1))
        XCTAssertNotNil(cache.get(k2))
    }

    func testUnpin_allowsLaterEviction() {
        var cache = RoundLRUCache(capacity: 2)
        let k0 = key(0), k1 = key(1), k2 = key(2)
        cache.put(k0, round(0))
        cache.pin(k0)
        cache.put(k1, round(1))
        cache.unpin(k0)
        // Now k0 is the LRU again — putting k2 evicts it.
        cache.put(k2, round(2))
        XCTAssertNil(cache.get(k0))
        XCTAssertNotNil(cache.get(k1))
        XCTAssertNotNil(cache.get(k2))
    }

    func testRemove_dropsEntry() {
        var cache = RoundLRUCache(capacity: 4)
        let k = key(0)
        cache.put(k, round(0))
        XCTAssertEqual(cache.count, 1)
        cache.remove(k)
        XCTAssertEqual(cache.count, 0)
        XCTAssertNil(cache.get(k))
    }

    func testEveryKeyPinned_growsPastCapacity() {
        var cache = RoundLRUCache(capacity: 2)
        let k0 = key(0), k1 = key(1), k2 = key(2)
        cache.put(k0, round(0)); cache.pin(k0)
        cache.put(k1, round(1)); cache.pin(k1)
        cache.put(k2, round(2)); cache.pin(k2)
        // Documented behavior: if we cannot evict, the cache grows.
        // This guarantees the streaming round is never lost, even if
        // the user opened a wave of other disclosures.
        XCTAssertEqual(cache.count, 3)
        XCTAssertNotNil(cache.get(k0))
        XCTAssertNotNil(cache.get(k1))
        XCTAssertNotNil(cache.get(k2))
    }

    func testIsPinnedReflectsPinUnpin() {
        var cache = RoundLRUCache(capacity: 2)
        let k = key(0)
        cache.put(k, round(0))
        XCTAssertFalse(cache.isPinned(k))
        cache.pin(k)
        XCTAssertTrue(cache.isPinned(k))
        cache.unpin(k)
        XCTAssertFalse(cache.isPinned(k))
    }
}

