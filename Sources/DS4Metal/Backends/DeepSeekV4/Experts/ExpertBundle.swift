import Foundation
import Metal
import DS4Core

// DS4_EXPERT_BUNDLE=1: sidecar file (<gguf>.expbundle) that repacks the routed
// experts so each expert's gate|up|down slabs are CONTIGUOUS (4 KB-aligned
// records, ordered by layer then expert id).
//
// Why: a slot-cache miss reads the three slabs of ONE expert. In the GGUF they
// live in three different tensors — three ~2 MB reads scattered across the
// file. Measured on the 2-bit Flash: the gather runs at ~49% of the SSD's
// parallel ceiling. In the bundle the same bytes are adjacent: the three
// concurrent preads form one ~7 MB sequential burst per miss.
//
// Same bytes, same numerics — only the on-disk LAYOUT changes. The bundle
// duplicates the expert region of the model on disk (~ everything but the
// dense weights), so it is OPT-IN and skipped when free space is short.
// Built once next to the model (.tmp + rename, torn files impossible);
// validated by size/geometry + per-layer content fingerprints; any failure
// logs and falls back to the plain GGUF reads.
public final class ExpertBundle: @unchecked Sendable, DS4Logging {
    public static let logTag = "expbundle"

    let fd: Int32
    let layers: Range<Int>
    let nExpert: Int
    let gateBytes: Int, upBytes: Int, downBytes: Int
    let dataBase: Int
    let record: Int              // aligned gate+up+down record stride
    /// Percorso del file effettivamente aperto — la UI lo mostra all'utente
    /// ("pronto" senza dire DOVE ha generato ore di caccia al file sbagliato).
    public let path: String
    /// Runtime PROOF in the engine log that misses are actually being served
    /// from the sidecar — "caricato" only proves the file validated. Logs the
    /// first served expert, then a LOGARITHMIC heartbeat (5k, 10k, 20k, 40k…):
    /// a long prefill serves hundreds of thousands of experts and the linear
    /// every-5000 beat flooded the log with dozens of identical lines.
    let useLock = NSLock()
    var served = 0
    var nextBeat = 5000
    /// Optional Metal 3 fast-resource-loading backend. It loads bundle ranges
    /// straight into MTLBuffer resources; `pread` remains the permanent
    /// correctness fallback if creation or any submission fails.
    struct MetalIOBackend {
        let queue: any MTLIOCommandQueue
        let handle: any MTLIOFileHandle
    }
    let metalIOLock = NSLock()
    var metalIO: MetalIOBackend?
    var metalIOFailureLogged = false
    var metalIOSubmissions = 0
    var metalIOBreaker = MetalIOCircuitBreaker(minimumGBs: 1.5)


    static let magic: UInt32 = 0x4245_5344   // "DSEB" little-endian
    static let version: UInt32 = 1
    static let align = 4096

    deinit { close(fd) }

    init(fd: Int32, path: String, layers: Range<Int>, nExpert: Int,
                 gateBytes: Int, upBytes: Int, downBytes: Int, dataBase: Int, record: Int) {
        self.fd = fd; self.path = path; self.layers = layers; self.nExpert = nExpert
        self.gateBytes = gateBytes; self.upBytes = upBytes; self.downBytes = downBytes
        self.dataBase = dataBase; self.record = record
    }
}
