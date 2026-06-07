import Foundation
#if canImport(Darwin)
import Darwin
#endif

// Faithful Swift port of ds4_ssd_memory_lock_acquire / _release.
// Reserves and mlock()s a block of anonymous memory so streaming behavior can
// be measured under reduced available RAM (--simulate-used-memory).

public final class SimulatedMemoryLock {
    private var ptr: UnsafeMutableRawPointer?
    private var bytes: UInt64 = 0

    public init() {}

    public var lockedBytes: UInt64 { bytes }

    /// Reserve, touch, and mlock `bytes` of memory in 256 MiB chunks. Returns
    /// false (and prints to stderr) on failure, matching the C original.
    /// A request of 0 is a successful no-op.
    @discardableResult
    public func acquire(bytes wanted: UInt64) -> Bool {
        ptr = nil
        bytes = 0
        if wanted == 0 { return true }
        if wanted > UInt64(Int.max) {
            err("ds4: --simulate-used-memory is too large for this process")
            return false
        }

        guard let region = mmap(nil, Int(wanted),
                                PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANON,
                                -1, 0),
              region != MAP_FAILED else {
            err(String(format: "ds4: --simulate-used-memory mmap %.2f GiB failed: %@",
                       gib(wanted), String(cString: strerror(errno))))
            return false
        }

        let pageLong = sysconf(Int32(_SC_PAGESIZE))
        let page = UInt64(pageLong > 0 ? pageLong : 4096)
        let chunkBytes: UInt64 = 256 * 1024 * 1024
        let p = region.assumingMemoryBound(to: UInt8.self)

        // Touch and lock in bounded chunks: a single huge mlock() is harder to
        // diagnose and can stall on macOS VM.
        var locked: UInt64 = 0
        var off: UInt64 = 0
        while off < wanted {
            var len = wanted - off
            if len > chunkBytes { len = chunkBytes }

            var pos = off
            while pos < off + len {
                p[Int(pos)] = UInt8(truncatingIfNeeded: pos / page)
                pos += page
            }
            if len != 0 { p[Int(off + len - 1)] = 1 }

            if mlock(region + Int(off), Int(len)) != 0 {
                err(String(format: "ds4: --simulate-used-memory mlock failed after %.2f/%.2f GiB: %@",
                           gib(locked), gib(wanted), String(cString: strerror(errno))))
                if locked != 0 { munlock(region, Int(locked)) }
                munmap(region, Int(wanted))
                return false
            }
            locked += len
            off += chunkBytes
        }

        ptr = region
        bytes = wanted
        err(String(format: "ds4: simulated used memory: locked %.2f GiB before model load",
                   gib(wanted)))
        return true
    }

    public func release() {
        guard let region = ptr, bytes != 0 else { return }
        munlock(region, Int(bytes))
        munmap(region, Int(bytes))
        ptr = nil
        bytes = 0
    }

    deinit { release() }

    private func gib(_ b: UInt64) -> Double { Double(b) / Double(SSDStreaming.gib) }

    private func err(_ message: String) {
        FileHandle.standardError.write(Data((message + "\n").utf8))
    }
}
