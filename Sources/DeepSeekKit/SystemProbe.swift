import Foundation
import Metal
#if canImport(Darwin)
import Darwin
#endif

/// Host-resource probes used by `LoadPlan.decide` to pick a checkpoint
/// loading strategy. All numbers are best-effort and informational —
/// the only metric the loader makes a hard decision on is
/// `processAvailableRAM()`.
public enum SystemProbe {

    /// Total installed physical RAM in bytes. Static for the lifetime
    /// of the process. Returned via `sysctlbyname("hw.memsize")`; 0 on
    /// failure (treated by callers as "unknown, skip the check").
    public static func physicalRAM() -> UInt64 {
        var v: UInt64 = 0
        var sz = MemoryLayout<UInt64>.size
        if sysctlbyname("hw.memsize", &v, &sz, nil, 0) == 0 { return v }
        return 0
    }

    /// Best-effort estimate of how many bytes are currently
    /// allocatable on this host before triggering jetsam.
    ///
    /// macOS has no per-process available-memory API exposed to
    /// userland (`os_proc_available_memory` is iOS-only, marked
    /// `API_UNAVAILABLE(macos)`), so we read the host-wide VM
    /// statistics and sum the page classes that the kernel treats as
    /// reclaimable: `free + inactive + speculative`. This matches
    /// Activity Monitor's "Memory Free + Cached Files" figure and is
    /// the same metric most pressure-aware tools (`vm_stat`, `top -l`)
    /// surface.
    public static func processAvailableRAM() -> UInt64 {
        var stats = vm_statistics64_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.stride /
            MemoryLayout<integer_t>.stride)
        let port = mach_host_self()
        let kr = withUnsafeMutablePointer(to: &stats) { ptr -> kern_return_t in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(port, HOST_VM_INFO64, $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return 0 }
        let page = UInt64(vm_kernel_page_size)
        return (UInt64(stats.free_count) +
                UInt64(stats.inactive_count) +
                UInt64(stats.speculative_count)) * page
    }

    /// Number of currently-online logical cores.
    public static func cpuCount() -> Int {
        ProcessInfo.processInfo.activeProcessorCount
    }

    /// Apple Silicon: the soft cap the GPU prefers to stay under for
    /// resident allocations. Informational — included in the log so
    /// the user can correlate jetsam events with a model that exceeds
    /// this number.
    public static func mtlRecommendedWorkingSet() -> UInt64 {
        Device.shared.mtl.recommendedMaxWorkingSetSize
    }

    /// Cap for how many shards we read in parallel when preloading. A
    /// single APFS-on-NVMe device tops out around 3-4 concurrent
    /// streams before contention drops aggregate throughput, so we
    /// don't scale with core count past 4.
    public static func preloadConcurrency() -> Int {
        min(cpuCount(), 4)
    }

    /// Effective unified-memory budget the loader's safety guards
    /// should treat as "the size we shouldn't oversubscribe past".
    ///
    /// Returns `min(processAvailableRAM, physicalRAM × cap)` with a
    /// conservative cap (default 60%): on Apple Silicon the GPU and
    /// CPU share the same physical pages, so the "available" figure
    /// from `host_statistics64` (free + inactive + speculative) is
    /// optimistic — those inactive pages are file cache and other
    /// processes' working sets that the kernel only evicts under
    /// real pressure. Once we start mmap'ing a multi-hundred-GB
    /// checkpoint the kernel is forced to evict, the GPU's working
    /// set competes for the same pages, and the whole system
    /// thrashes.
    ///
    /// Treating 60% of physical as the ceiling keeps roughly 40%
    /// of unified memory for the OS, the GUI, the GPU command
    /// queue, and any other app the user has open.
    public static func effectiveProcessBudget(
        physicalCap: Double = 0.60
    ) -> UInt64 {
        let phys = physicalRAM()
        let avail = processAvailableRAM()
        if phys == 0 { return avail }
        let physCap = UInt64(Double(phys) * physicalCap)
        if avail == 0 { return physCap }
        return min(avail, physCap)
    }

    // MARK: - GPU identification

    /// Marketing string esposta da `MTLDevice.name`. Su Apple Silicon
    /// è la forma "Apple M<N> [Pro|Max|Ultra]" o "Apple M<N>".
    public static func gpuName() -> String {
        Device.shared.mtl.name
    }

    /// Tabella interna chip → numero di GPU core, esposta `internal`
    /// per i test. Match per prefisso (vedi `gpuCoreCount()`): la
    /// prima entry il cui prefisso matcha `name` vince. Ordinata
    /// dal più specifico (Pro/Max/Ultra) al più generico (base
    /// "Apple M<N>") per evitare false partite.
    static let _internalCoreTable: [(prefix: String, cores: Int)] = [
        ("Apple M1 Ultra", 64),
        ("Apple M1 Max",   32),
        ("Apple M1 Pro",   16),
        ("Apple M1",        8),
        ("Apple M2 Ultra", 76),
        ("Apple M2 Max",   38),
        ("Apple M2 Pro",   19),
        ("Apple M2",       10),
        ("Apple M3 Ultra", 80),
        ("Apple M3 Max",   40),
        ("Apple M3 Pro",   18),
        ("Apple M3",       10),
        ("Apple M4 Max",   40),
        ("Apple M4 Pro",   20),
        ("Apple M4",       10),
    ]

    /// Numero di GPU core, in base alla tabella interna. `nil` se
    /// il chip non è riconosciuto (es. M5+ futuri, GPU non Apple).
    /// Il caller decide cosa fare con il nil — tipicamente loggare
    /// un warning e cadere su `gpuCoreCountOrFallback()`.
    ///
    /// Logica:
    ///   - prova ogni entry della tabella in ordine, restituisce
    ///     la prima il cui `prefix` è un prefisso esatto di `name`,
    ///   - per evitare match parziali (es. "Apple M3" che matcha
    ///     "Apple M3 Max"), la tabella ordina i nomi specifici
    ///     PRIMA dei generici.
    public static func gpuCoreCount() -> Int? {
        return _coreCount(forDeviceName: gpuName())
    }

    /// Variante deterministica: usa il fallback `cpuCount() * 2`
    /// quando il chip non è in tabella. Non emette warning — è
    /// pensata per i call site che non possono gestire `nil`.
    public static func gpuCoreCountOrFallback() -> Int {
        if let n = gpuCoreCount() { return n }
        return cpuCount() * 2
    }

    /// Risolve un nome arbitrario (utile nei test, dove vogliamo
    /// passare stringhe sintetiche senza dipendere dal device
    /// reale). Internal per non esporre l'API ai consumer esterni.
    static func _coreCount(forDeviceName name: String) -> Int? {
        for entry in _internalCoreTable {
            if name.hasPrefix(entry.prefix) {
                return entry.cores
            }
        }
        return nil
    }
}
