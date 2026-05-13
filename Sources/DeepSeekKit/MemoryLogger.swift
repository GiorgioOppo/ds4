import Foundation
import Metal
#if canImport(Darwin)
import Darwin
import MachO
#endif

/// Diagnostic memory tracker. Logs to stderr at strategic forward-
/// pass boundaries when ANY of the tracked metrics moves by more
/// than `thresholdBytes` since the last sample, or unconditionally
/// when `force: true`.
///
/// Tracks five distinct numbers because on Apple Silicon the
/// "memory used by my process" question has multiple answers and
/// they don't agree:
///
///   - **rss**:   `task_info(MACH_TASK_BASIC_INFO).resident_size`.
///                What Activity Monitor's "Memory" column shows.
///                Excludes pages wired by IOKit drivers on the
///                process's behalf.
///   - **vsz**:   `task_info.virtual_size`. Virtual address space
///                reserved (e.g. all mmap'd shards count here even
///                with no pages faulted).
///   - **mtl**:   `MTLDevice.currentAllocatedSize`. Metal driver's
///                tally of MTLBuffer / MTLTexture / heap bytes
///                allocated. Includes the GPU-driver-pinned
///                portions Activity Monitor hides.
///   - **freeRAM**: `host_statistics64 HOST_VM_INFO64.free_count`
///                converted to bytes. System-wide truly-unused.
///   - **wired**: `host_statistics64.wire_count`. Pages pinned by
///                the kernel and drivers — grows with our MTL
///                allocations even when rss doesn't.
public enum MemoryLogger {
    /// Master switch. Off by default — flip to true (programmatically
    /// or via the env var below) to re-enable the trace when
    /// diagnosing memory growth.
    public static var enabled: Bool = {
        ProcessInfo.processInfo.environment["DEEPSEEK_MEM_LOG"] == "1"
    }()

    /// Skip a snapshot if none of the metrics moved more than this
    /// since the last log. Default is 0 (emit every call) — debug
    /// mode. Bump to e.g. 256 MB for a quieter trace.
    public static var thresholdBytes: UInt64 = 0

    /// Pressure ratio (1 - sysFree/physical) over which we prefix
    /// the label with ⚠️ (HIGH) and 🔥 (CRITICAL). Tuned to fire
    /// before macOS's own jetsam pressure events, so the user sees
    /// the spike in the log seconds before the kernel kills the
    /// process.
    public static var pressureHigh: Double = 0.70
    public static var pressureCritical: Double = 0.85

    /// `willAllocate` skips lines for allocations smaller than
    /// this. Default 1 MB — keeps the trace focused on the
    /// allocations that matter for OOM diagnosis (KV cache,
    /// activations, mmap shards). Set to 0 to see every single
    /// MTLBuffer creation.
    public static var allocLogThresholdBytes: UInt64 = 1024 * 1024

    /// Lazily-mutated state — single-threaded use only (we only
    /// snapshot from MainActor or the inference Task, never both).
    nonisolated(unsafe) private static var last = Sample()
    nonisolated(unsafe) private static var headerEmitted = false
    nonisolated(unsafe) private static var cachedPhysical: UInt64 = 0

    /// Emit a one-line log RIGHT BEFORE an MTLBuffer / mmap region
    /// is allocated, so if the allocation triggers a kernel panic
    /// the user sees "what was being requested" as the last
    /// surviving line in the trace.
    public static func willAllocate(bytes: Int,
                                     shape: [Int]? = nil,
                                     dtype: DType? = nil,
                                     label: String = "") {
        guard enabled, UInt64(bytes) >= allocLogThresholdBytes else { return }
        emitHeaderIfNeeded()
        var parts: [String] = []
        parts.append("[alloc \(formatBytes(UInt64(bytes)))")
        if let s = shape { parts.append("shape=\(s)") }
        if let d = dtype { parts.append("dtype=\(d.shortName)") }
        if !label.isEmpty { parts.append(label) }
        parts[parts.count - 1] += "]"
        let line = parts.joined(separator: " ") + "\n"
        FileHandle.standardError.write(Data(line.utf8))
    }

    public static func snapshot(_ label: String, force: Bool = false) {
        guard enabled else { return }
        emitHeaderIfNeeded()
        let cur = currentSample()
        let bigDelta = absDelta(cur.rss, last.rss) >= thresholdBytes
                    || absDelta(cur.mtl, last.mtl) >= thresholdBytes
                    || absDelta(cur.wired, last.wired) >= thresholdBytes
        if !force && thresholdBytes > 0 && !bigDelta { return }

        // System-wide pressure: what % of physical RAM is taken
        // (by anyone, not just us). Catches kernel_task growth from
        // the Metal driver wiring pages on our behalf — those
        // bytes never appear in `rss` but they DO reduce sysFree
        // and grow `wired`.
        let phys = cachedPhysicalRAM()
        let usedSys = phys > cur.freeRAM ? phys - cur.freeRAM : 0
        let pressure = phys > 0 ? Double(usedSys) / Double(phys) : 0
        let marker: String
        if pressure >= pressureCritical { marker = "🔥 " }
        else if pressure >= pressureHigh { marker = "⚠️ " }
        else { marker = "" }

        let line = String(format:
            "[mem %@%@] rss=%.2fGB vsz=%.2fGB mtl=%.2fGB sysFree=%.2fGB wired=%.2fGB pressure=%.0f%%\n",
            marker as CVarArg,
            label as CVarArg,
            cur.rss.gib,
            cur.vsz.gib,
            cur.mtl.gib,
            cur.freeRAM.gib,
            cur.wired.gib,
            pressure * 100)
        FileHandle.standardError.write(Data(line.utf8))
        last = cur
    }

    /// Force a final summary line. Always emits.
    public static func summary(_ label: String) {
        snapshot(label, force: true)
    }

    private static func emitHeaderIfNeeded() {
        if headerEmitted { return }
        headerEmitted = true
        let header = """
        [mem] columns: rss=process resident · vsz=virtual addr space · mtl=Metal driver allocated · sysFree=system pages free · wired=system pages wired · pressure=1-sysFree/physical
        [mem] watermarks: ⚠️ HIGH at \(Int(pressureHigh * 100))% · 🔥 CRITICAL at \(Int(pressureCritical * 100))%

        """
        FileHandle.standardError.write(Data(header.utf8))
    }

    private static func cachedPhysicalRAM() -> UInt64 {
        if cachedPhysical == 0 {
            cachedPhysical = SystemProbe.physicalRAM()
        }
        return cachedPhysical
    }

    // ---- internals ----

    private struct Sample {
        var rss: UInt64 = 0
        var vsz: UInt64 = 0
        var mtl: UInt64 = 0
        var freeRAM: UInt64 = 0
        var wired: UInt64 = 0
    }

    private static func currentSample() -> Sample {
        var s = Sample()
        s.mtl = UInt64(max(0, Device.shared.mtl.currentAllocatedSize))

        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(
            MemoryLayout<mach_task_basic_info>.stride /
            MemoryLayout<integer_t>.stride)
        let kr = withUnsafeMutablePointer(to: &info) { ptr -> kern_return_t in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_,
                           task_flavor_t(MACH_TASK_BASIC_INFO),
                           $0, &count)
            }
        }
        if kr == KERN_SUCCESS {
            s.rss = UInt64(info.resident_size)
            s.vsz = UInt64(info.virtual_size)
        }

        var vm = vm_statistics64_data_t()
        var vmCount = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.stride /
            MemoryLayout<integer_t>.stride)
        let port = mach_host_self()
        let vmkr = withUnsafeMutablePointer(to: &vm) { ptr -> kern_return_t in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(vmCount)) {
                host_statistics64(port, HOST_VM_INFO64, $0, &vmCount)
            }
        }
        if vmkr == KERN_SUCCESS {
            let page = UInt64(vm_kernel_page_size)
            s.freeRAM = UInt64(vm.free_count) * page
            s.wired = UInt64(vm.wire_count) * page
        }
        return s
    }

    private static func absDelta(_ a: UInt64, _ b: UInt64) -> UInt64 {
        a > b ? a - b : b - a
    }

    private static func formatBytes(_ b: UInt64) -> String {
        let kib = 1024.0
        let mib = kib * 1024
        let gib = mib * 1024
        let bd = Double(b)
        if bd >= gib { return String(format: "%.2fGB", bd / gib) }
        if bd >= mib { return String(format: "%.1fMB", bd / mib) }
        if bd >= kib { return String(format: "%.0fKB", bd / kib) }
        return "\(b)B"
    }
}

private extension DType {
    /// Short tag used in alloc log lines. Mirrors the safetensors
    /// dtype strings where possible.
    var shortName: String {
        switch self {
        case .f32:      return "f32"
        case .f16:      return "f16"
        case .bf16:     return "bf16"
        case .i32:      return "i32"
        case .i64:      return "i64"
        case .i8:       return "i8"
        case .i4:       return "i4"
        case .i2:       return "i2"
        case .fp8E4M3:  return "fp8"
        case .fp4E2M1:  return "fp4"
        case .e8m0:     return "e8m0"
        }
    }
}

private extension UInt64 {
    var gib: Double { Double(self) / (1024 * 1024 * 1024) }
}
