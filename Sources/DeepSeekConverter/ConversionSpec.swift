import Foundation

/// Target weight dtype for a quantization conversion. Used by
/// `Converter.runQuantize`. Mirrors the existing `--target-dtype`
/// CLI flag.
public enum ConversionTarget: String, Sendable, CaseIterable {
    case bf16, f16
    case int8, int4, int2
    /// Keep the source dtype as-is (just fuse FP8/FP4 + scale into
    /// the safetensors entry); equivalent to the converter's
    /// historical `--target-dtype keep`. Useful for repacking HF
    /// shards into our naming convention without changing
    /// precision.
    case keep

    /// Dtype tag we write into the safetensors header.
    public var safetensorsTag: String {
        switch self {
        case .bf16: return "BF16"
        case .f16:  return "F16"
        case .int8: return "I8"
        case .int4: return "I4"
        case .int2: return "I2"
        case .keep: return ""  // determined per-tensor
        }
    }

    /// For BF16/F16 (the only target dtypes that emit a single
    /// tensor per weight) this is 2 bytes per element. The INT*
    /// paths size their own outputs (sub-byte packing + scale
    /// companion), so callers that hit those return values here
    /// don't read them.
    public var bytesPerElement: Int {
        switch self {
        case .bf16, .f16: return 2
        case .int8:       return 1
        case .int4:       return 1   // unused; INT4 path computes inDim/2
        case .int2:       return 1   // unused; INT2 path computes inDim/4
        case .keep:       return 2   // unused in keep mode
        }
    }
}

/// Input parameters for a HF → native conversion run.
public struct QuantizeSpec: Sendable {
    /// Path to the HuggingFace-format checkpoint directory
    /// (`*.safetensors`, `tokenizer.json`, `config.json`).
    public var hfPath: URL
    /// Output directory the converter writes shards + index +
    /// tokenizer/config sidecars to. Created if missing.
    public var savePath: URL
    /// Total number of routed experts in the model. Used to size
    /// the MoE expert tensors and pick the layer-aligned shard
    /// packing. Required because the kit needs it before reading
    /// the checkpoint; HF config.json carries it but the converter
    /// doesn't parse JSON to discover it.
    public var nExperts: Int
    /// Model-parallel sharding factor on the input side. The HF
    /// checkpoints we consume are produced with `mp=1`; leave at 1
    /// unless you're feeding a sharded source.
    public var modelParallel: Int
    /// Target dtype for `Linear` weights. Non-Linear tensors stay
    /// at BF16 regardless (RMSNorm gains, biases, attn_sink, etc.).
    public var target: ConversionTarget
    /// Maximum shard size in gigabytes (10^9 bytes). Capped at 95%
    /// of `MTLDevice.maxBufferLength` by the runtime since each
    /// shard becomes one `MTLBuffer` via `bytesNoCopy`.
    public var shardSizeGB: Double

    public init(hfPath: URL, savePath: URL, nExperts: Int,
                 modelParallel: Int = 1,
                 target: ConversionTarget,
                 shardSizeGB: Double = 5.0) {
        self.hfPath = hfPath
        self.savePath = savePath
        self.nExperts = nExperts
        self.modelParallel = modelParallel
        self.target = target
        self.shardSizeGB = shardSizeGB
    }
}

/// Input parameters for a previously-converted-by-us → BF16
/// dequantization run. Reverse of `QuantizeSpec`. Reads INT8 /
/// INT4 / INT2 `.weight` + F16 `.scale` companions, dequantizes
/// in-place per tensor, and writes a plain BF16 checkpoint with
/// the same naming convention.
public struct DequantizeSpec: Sendable {
    /// Path to a directory previously produced by `QuantizeSpec`
    /// with target = .int8 / .int4 / .int2. Must contain
    /// `model-*.safetensors` shards plus the .scale companion
    /// tensors the runtime loader pairs at load time.
    public var inputPath: URL
    /// Destination directory. Tokenizer/config sidecars are copied
    /// verbatim from `inputPath`.
    public var savePath: URL
    /// Target dtype. Only BF16/F16 make sense for a dequant; INT*
    /// targets are rejected by the runner.
    public var target: ConversionTarget
    /// Shard size cap (same constraints as QuantizeSpec).
    public var shardSizeGB: Double

    public init(inputPath: URL, savePath: URL,
                 target: ConversionTarget = .bf16,
                 shardSizeGB: Double = 5.0) {
        self.inputPath = inputPath
        self.savePath = savePath
        self.target = target
        self.shardSizeGB = shardSizeGB
    }
}
