import Foundation
import DS4Core

// Roadmap step 1, second tranche: a slot cache between the payload reader and
// the future decode loop. Slots hold one expert's packed gate|up|down record
// (the reader's layout, so a hit and a fresh read serve byte-identical
// records — the upstream logits-invariance lesson holds by construction).
// Eviction is least-recently-used; the experts of the batch being served are
// pinned so one token's top-8 can never evict itself. Pure bookkeeping over
// pread — MetalIO fills come later beside it, with this path as the
// permanent correctness fallback.

public struct GLM52ExpertSlotCacheStats: Sendable, Equatable {
    public internal(set) var hits = 0
    public internal(set) var misses = 0
    public internal(set) var evictions = 0

    public init() {}
}

public enum GLM52ExpertSlotCacheError: Error, Sendable, Equatable,
    CustomStringConvertible {
    case invalidSlotGeometry(slotCount: Int, slotBytes: Int)
    case budgetBelowSelectionWorkingSet(slots: Int, required: Int)
    case recordLargerThanSlot(needed: Int, slotBytes: Int)
    case batchLargerThanCache(batch: Int, slots: Int)

    public var description: String {
        switch self {
        case .invalidSlotGeometry(let slots, let bytes):
            return "GLM 5.2 expert cache: invalid geometry \(slots) slots × \(bytes) bytes"
        case .budgetBelowSelectionWorkingSet(let slots, let required):
            return "GLM 5.2 expert cache: \(slots) slots cannot hold one "
                + "token's \(required)-expert selection"
        case .recordLargerThanSlot(let needed, let bytes):
            return "GLM 5.2 expert cache: record needs \(needed) bytes, slot holds \(bytes)"
        case .batchLargerThanCache(let batch, let slots):
            return "GLM 5.2 expert cache: batch of \(batch) exceeds \(slots) slots"
        }
    }
}

/// LRU slot cache of packed expert records keyed by (layer, expert).
///
/// Not thread-safe by design: the decode loop owns it like the DeepSeek slot
/// cache. Byte identity between hit and miss is the contract — a cached
/// record is exactly the bytes `GLM52PayloadReader` read for that expert.
public final class GLM52ExpertSlotCache {
    private struct Key: Hashable {
        let layer: Int
        let expert: UInt32
    }

    private struct Entry {
        let key: Key
        let byteCount: Int
        var lastUse: UInt64
    }

    private let reader: GLM52PayloadReader
    private let slotBytes: Int
    private let slotCount: Int
    private var storage: [UInt8]
    private var entries: [Entry?]
    private var slotByKey: [Key: Int] = [:]
    private var clock: UInt64 = 0
    public private(set) var stats = GLM52ExpertSlotCacheStats()

    /// `selectionWidth` is the per-token working set (the router's top-k):
    /// budgets below it cannot pin one batch and are refused at creation, the
    /// interim upstream guard kept as a hard floor here.
    public init(reader: GLM52PayloadReader,
                slotCount: Int,
                slotBytes: Int,
                selectionWidth: Int = Int(GLM52Shape.v5_2.nExpertUsed)) throws {
        guard slotCount > 0, slotBytes > 0, selectionWidth > 0 else {
            throw GLM52ExpertSlotCacheError.invalidSlotGeometry(
                slotCount: slotCount, slotBytes: slotBytes)
        }
        guard slotCount >= selectionWidth else {
            throw GLM52ExpertSlotCacheError.budgetBelowSelectionWorkingSet(
                slots: slotCount, required: selectionWidth)
        }
        self.reader = reader
        self.slotCount = slotCount
        self.slotBytes = slotBytes
        self.storage = [UInt8](repeating: 0, count: slotCount * slotBytes)
        self.entries = [Entry?](repeating: nil, count: slotCount)
    }

    /// Derive the slot count from a byte budget (floor division).
    public convenience init(reader: GLM52PayloadReader,
                            byteBudget: Int,
                            slotBytes: Int,
                            selectionWidth: Int = Int(GLM52Shape.v5_2.nExpertUsed)) throws {
        guard slotBytes > 0 else {
            throw GLM52ExpertSlotCacheError.invalidSlotGeometry(
                slotCount: 0, slotBytes: slotBytes)
        }
        try self.init(reader: reader,
                      slotCount: byteBudget / slotBytes,
                      slotBytes: slotBytes,
                      selectionWidth: selectionWidth)
    }

    /// Serve one layer's stream plan: every selected expert's packed record,
    /// from cache or freshly read, in router rank order. The batch's slots
    /// are pinned against each other; `body` receives one raw buffer per
    /// expert (valid only inside the closure) plus the shared record layout.
    public func withRecords<R>(
        plan: GLM52ExpertStreamPlan,
        _ body: ([UnsafeRawBufferPointer], GLM52ExpertPackedRecordLayout) throws -> R
    ) throws -> R {
        let layout = try reader.packedLayout(
            of: GLM52ExpertStreamPlan(
                layer: plan.layer,
                experts: [plan.experts[0]],
                totalBytes: plan.experts[0].totalBytes))
        guard layout.recordBytes <= slotBytes else {
            throw GLM52ExpertSlotCacheError.recordLargerThanSlot(
                needed: layout.recordBytes, slotBytes: slotBytes)
        }
        guard plan.experts.count <= slotCount else {
            throw GLM52ExpertSlotCacheError.batchLargerThanCache(
                batch: plan.experts.count, slots: slotCount)
        }

        // Phase 1: resolve every expert to a slot, filling misses via the
        // reader. Slots claimed by THIS batch are pinned.
        var pinned = Set<Int>()
        var slots = [Int]()
        slots.reserveCapacity(plan.experts.count)
        for expert in plan.experts {
            let key = Key(layer: plan.layer, expert: expert.expertID)
            clock += 1
            if let slot = slotByKey[key],
               let entry = entries[slot],
               entry.byteCount == layout.recordBytes {
                entries[slot]?.lastUse = clock
                stats.hits += 1
                pinned.insert(slot)
                slots.append(slot)
                continue
            }
            let slot = try claimSlot(excluding: pinned)
            let singlePlan = GLM52ExpertStreamPlan(
                layer: plan.layer,
                experts: [expert],
                totalBytes: expert.totalBytes)
            try storage.withUnsafeMutableBytes { raw in
                let base = raw.baseAddress! + slot * slotBytes
                let destination = UnsafeMutableRawBufferPointer(
                    start: base, count: layout.recordBytes)
                try reader.read(plan: singlePlan, into: destination)
            }
            entries[slot] = Entry(key: key,
                                  byteCount: layout.recordBytes,
                                  lastUse: clock)
            slotByKey[key] = slot
            stats.misses += 1
            pinned.insert(slot)
            slots.append(slot)
        }

        // Phase 2: hand the pinned records to the caller.
        let recordBytes = layout.recordBytes
        let slotStride = slotBytes
        return try storage.withUnsafeBytes { raw in
            let buffers = slots.map { slot in
                UnsafeRawBufferPointer(
                    start: raw.baseAddress! + slot * slotStride,
                    count: recordBytes)
            }
            return try body(buffers, layout)
        }
    }

    /// Least-recently-used unpinned slot, preferring empty ones.
    private func claimSlot(excluding pinned: Set<Int>) throws -> Int {
        var victim: Int?
        var oldest = UInt64.max
        for slot in 0..<slotCount where !pinned.contains(slot) {
            guard let entry = entries[slot] else { return slot }
            if entry.lastUse < oldest {
                oldest = entry.lastUse
                victim = slot
            }
        }
        guard let slot = victim else {
            throw GLM52ExpertSlotCacheError.batchLargerThanCache(
                batch: pinned.count + 1, slots: slotCount)
        }
        if let entry = entries[slot] {
            slotByKey.removeValue(forKey: entry.key)
            stats.evictions += 1
        }
        entries[slot] = nil
        return slot
    }
}
