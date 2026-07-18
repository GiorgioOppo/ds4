import DS4Core
import Foundation
import Metal

// Keyed LRU arena of routed-expert records in ONE shared MTLBuffer — the
// retention the page cache could never give this workload. Three jobs:
//
// 1. stage() (decode thread): resolve a selection to per-record buffer
//    offsets, reading ONLY the records not already resident. Repeats across
//    consecutive prefill tokens (layer-major revisits the same layer) and
//    speculative warm-ups become zero-copy, zero-I/O hits.
// 2. speculate() (background): warm slots with a guessed selection (the
//    previous token's routing) while the GPU computes attention — the read
//    happens INSIDE the compute window instead of stalling the FFN.
// 3. Retention: slots keep their record until evicted by LRU, so a hit is
//    guaranteed to still hold real bytes (fills in flight carry a
//    DispatchGroup the stager waits on).
//
// Victims are never taken from slots that are mid-fill or belong to the
// selection being reserved, so a staged record can never be overwritten
// while its command buffer is being encoded: the caller consumes it within
// the layer's synchronous commit, and eviction needs the slot to fall out
// of the LRU across LATER selections first.

final class GLM52ExpertArena {
    struct Stats: Sendable {
        var hitBytes: UInt64 = 0
        var readBytes: UInt64 = 0
        var speculativeBytes: UInt64 = 0
    }

    let buffer: MTLBuffer
    let slotBytes: Int

    private struct Slot {
        var key = UInt64.max
        var tick: UInt64 = 0
        var filling: DispatchGroup?
    }

    private struct Reservation {
        let rank: Int
        let id: UInt32
        let slot: Int
        let fresh: Bool
        let inFlight: DispatchGroup?
    }

    private let lock = NSLock()
    private var slots: [Slot]
    private var map: [UInt64: Int] = [:]
    private var tick: UInt64 = 0
    private var stats = Stats()

    /// `slotCount` is clamped to at least 24: one live selection (8) plus a
    /// speculative selection in flight (8) must NEVER exhaust the victims.
    init(device: MTLDevice, slotCount: Int, slotBytes: Int) throws {
        let count = max(24, slotCount)
        guard slotBytes > 0, let buffer = device.makeBuffer(
            length: count * slotBytes, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        self.buffer = buffer
        self.slotBytes = slotBytes
        slots = Array(repeating: Slot(), count: count)
    }

    private static func key(layer: Int, id: UInt32) -> UInt64 {
        UInt64(UInt32(layer)) << 32 | UInt64(id)
    }

    func statsSnapshot() -> Stats {
        lock.lock()
        defer { lock.unlock() }
        return stats
    }

    func resetStats() {
        lock.lock()
        stats = Stats()
        lock.unlock()
    }

    // MARK: - Reservation core

    private func reserve(layer: Int, ids: [UInt32], recordBytes: Int,
                         speculative: Bool) -> [Reservation] {
        lock.lock()
        defer { lock.unlock() }
        tick += 1
        let wanted = Set(ids.map { Self.key(layer: layer, id: $0) })
        var reservations: [Reservation] = []
        reservations.reserveCapacity(ids.count)
        for (rank, id) in ids.enumerated() {
            let key = Self.key(layer: layer, id: id)
            if let slot = map[key] {
                slots[slot].tick = tick
                if !speculative { stats.hitBytes += UInt64(recordBytes) }
                reservations.append(Reservation(
                    rank: rank, id: id, slot: slot, fresh: false,
                    inFlight: slots[slot].filling))
                continue
            }
            var victim = -1
            var oldest = UInt64.max
            for (index, slot) in slots.enumerated()
                where slot.filling == nil && !wanted.contains(slot.key)
                    && slot.tick < oldest {
                victim = index
                oldest = slot.tick
            }
            guard victim >= 0 else {
                // Sized to make this unreachable (see init); a speculative
                // caller just skips, the stager reports it loudly.
                reservations.append(Reservation(
                    rank: rank, id: id, slot: -1, fresh: true,
                    inFlight: nil))
                continue
            }
            map.removeValue(forKey: slots[victim].key)
            map[key] = victim
            slots[victim].key = key
            slots[victim].tick = tick
            let group = DispatchGroup()
            group.enter()
            slots[victim].filling = group
            if speculative {
                stats.speculativeBytes += UInt64(recordBytes)
            } else {
                stats.readBytes += UInt64(recordBytes)
            }
            reservations.append(Reservation(
                rank: rank, id: id, slot: victim, fresh: true,
                inFlight: nil))
        }
        return reservations
    }

    /// Concurrent fill of the fresh reservations; on ANY failure every
    /// fresh key of this call is scrubbed (waiters see the miss and re-read)
    /// and the first error is thrown to a throwing caller.
    private func fill(_ fresh: [Reservation], layer: Int, recordBytes: Int,
                      read: (UInt32, UnsafeMutableRawBufferPointer) throws
                          -> Void) throws {
        guard !fresh.isEmpty else { return }
        nonisolated(unsafe) var failure: Error?
        let failureLock = NSLock()
        nonisolated(unsafe) let base = buffer.contents()
        let slotBytes = self.slotBytes
        DispatchQueue.concurrentPerform(iterations: fresh.count) { i in
            let reservation = fresh[i]
            let destination = UnsafeMutableRawBufferPointer(
                start: base + reservation.slot * slotBytes,
                count: recordBytes)
            do {
                try read(reservation.id, destination)
            } catch {
                failureLock.lock()
                if failure == nil { failure = error }
                failureLock.unlock()
            }
        }
        lock.lock()
        for reservation in fresh {
            if failure != nil {
                let key = Self.key(layer: layer, id: reservation.id)
                if map[key] == reservation.slot {
                    map.removeValue(forKey: key)
                    slots[reservation.slot].key = .max
                }
            }
            slots[reservation.slot].filling?.leave()
            slots[reservation.slot].filling = nil
        }
        lock.unlock()
        if let failure { throw failure }
    }

    // MARK: - Public operations

    /// Resolve `ids` to per-rank byte offsets into `buffer`, reading only
    /// the records not already resident. Blocks until every record —
    /// including any still being speculatively filled — holds real bytes.
    func stage(layer: Int, ids: [UInt32], recordBytes: Int,
               read: (UInt32, UnsafeMutableRawBufferPointer) throws -> Void)
        throws -> [Int] {
        guard recordBytes <= slotBytes else {
            throw MetalError.unsupported(
                "GLM 5.2 expert arena: record da \(recordBytes) B oltre lo "
                + "slot da \(slotBytes) B")
        }
        var offsets = [Int](repeating: -1, count: ids.count)
        var remaining = Array(zip(ids.indices, ids))
        // A speculative fill that failed (scrubbed key) shows up as a hit
        // whose slot no longer carries the key: retry as a fresh read.
        for _ in 0..<3 where !remaining.isEmpty {
            let batch = remaining.map(\.1)
            let ranks = remaining.map(\.0)
            let reservations = reserve(layer: layer, ids: batch,
                                       recordBytes: recordBytes,
                                       speculative: false)
            guard reservations.allSatisfy({ $0.slot >= 0 }) else {
                throw MetalError.unsupported(
                    "GLM 5.2 expert arena esaurita: aumentare "
                    + "DS4_GLM_EXPERT_ARENA")
            }
            try fill(reservations.filter(\.fresh), layer: layer,
                     recordBytes: recordBytes, read: read)
            for reservation in reservations where !reservation.fresh {
                reservation.inFlight?.wait()
            }
            var retry: [(Int, UInt32)] = []
            lock.lock()
            for reservation in reservations {
                let key = Self.key(layer: layer, id: reservation.id)
                if map[key] == reservation.slot {
                    offsets[ranks[reservation.rank]] =
                        reservation.slot * slotBytes
                } else {
                    retry.append((ranks[reservation.rank], reservation.id))
                }
            }
            lock.unlock()
            remaining = retry
        }
        guard remaining.isEmpty else {
            throw MetalError.unsupported(
                "GLM 5.2 expert arena: record non stabilizzati dopo i retry")
        }
        return offsets
    }

    /// Best-effort background warm-up: reserve and fill only the missing
    /// records of a GUESSED selection; never waits, errors only scrub.
    func speculate(layer: Int, ids: [UInt32], recordBytes: Int,
                   read: (UInt32, UnsafeMutableRawBufferPointer) throws
                       -> Void) {
        guard recordBytes <= slotBytes else { return }
        let reservations = reserve(layer: layer, ids: ids,
                                   recordBytes: recordBytes,
                                   speculative: true)
        let fresh = reservations.filter { $0.fresh && $0.slot >= 0 }
        try? fill(fresh, layer: layer, recordBytes: recordBytes, read: read)
    }
}
