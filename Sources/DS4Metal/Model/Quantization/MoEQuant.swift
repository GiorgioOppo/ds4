import Foundation
import Metal

/// Routed-expert quantization formats supported by the MoE matvec. All share the
/// `kernel_mul_mv_id<...>` wrapper; they differ only in GGUF block byte-size and
/// the rows-per-threadgroup (nr0). Maps from GGUF type codes (q4_K=12, q2_K=10,
/// iq2_xxs=16).
public enum MoEQuant: Sendable {
    case q4_K, q2_K, iq2_xxs
    public var kernel: String {
        switch self {
        case .q4_K:    return "kernel_mul_mv_id_q4_K_f32"
        case .q2_K:    return "kernel_mul_mv_id_q2_K_f32"
        case .iq2_xxs: return "kernel_mul_mv_id_iq2_xxs_f32"
        }
    }
    /// GGUF block bytes per 256 elements.
    public var blockBytes: Int {
        switch self { case .q4_K: return 144; case .q2_K: return 84; case .iq2_xxs: return 66 }
    }
    /// N_R0_* — rows per threadgroup (q4_K=2; q2_K/iq2_xxs=4).
    public var nr0: Int { self == .q4_K ? 2 : 4 }
    /// Threadgroup memory the kernel needs. iq2_xxs cooperatively loads the
    /// codebook into shared memory: svalues = uint64 grid[256] (2048 B) + ssigns
    /// = uint8[128] (128 B) = 2176 B. q2_K/q4_K only use the 256 B reduction
    /// scratch. Allocating only 256 B for iq2_xxs causes out-of-bounds threadgroup
    /// writes → garbage grid → wrong matvec on every gate/up expert.
    public var threadgroupBytes: Int { self == .iq2_xxs ? (256 * 8 + 128) : 256 }
    public static func from(ggufType: UInt32) -> MoEQuant? {
        switch ggufType { case 12: return .q4_K; case 10: return .q2_K; case 16: return .iq2_xxs; default: return nil }
    }
}

// Stage C: graph composition. Composes the validated Stage-B tensor-ops on
// GPUTensors into the DSV4 graph fragments. First fragments: token embedding
// (get_rows -> HC repeat) and the output head (final RMSNorm -> vocab matmul).
// Each fragment encodes into a GraphContext (one command buffer) and is validated
// against a CPU reference; the per-layer decode/prefill body builds on these.

