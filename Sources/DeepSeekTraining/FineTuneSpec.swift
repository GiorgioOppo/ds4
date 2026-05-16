import Foundation

/// Optimizer choice for a fine-tuning run. Plain enum mirroring the
/// flags that the future `finetune` CLI / native backend will accept.
public enum FineTuneOptimizer: String, Sendable, CaseIterable {
    case adamW = "adamw"
    case sgd
    case lion

    public var displayName: String {
        switch self {
        case .adamW: return "AdamW"
        case .sgd:   return "SGD"
        case .lion:  return "Lion"
        }
    }
}

/// Layout of the training dataset on disk. Picks how the runner will
/// tokenize each row.
public enum DatasetFormat: String, Sendable, CaseIterable {
    /// One JSON object per line with a `messages: [{role, content}]`
    /// field — the OpenAI fine-tuning format.
    case jsonlChat = "jsonl-chat"
    /// One JSON object per line with `prompt` + `completion`.
    case jsonlPromptCompletion = "jsonl-prompt-completion"
    /// Plain UTF-8 text, split into max-seq-len chunks.
    case plainText = "text"

    public var displayName: String {
        switch self {
        case .jsonlChat:              return "JSONL chat (messages[])"
        case .jsonlPromptCompletion:  return "JSONL prompt + completion"
        case .plainText:              return "Plain text"
        }
    }
}

/// Mixed-precision flavour. Mirrors the dtype options we already
/// support on the inference path.
public enum TrainingPrecision: String, Sendable, CaseIterable {
    case bf16
    case f16
    case f32

    public var displayName: String {
        switch self {
        case .bf16: return "BF16"
        case .f16:  return "F16"
        case .f32:  return "F32"
        }
    }

    public var detail: String {
        switch self {
        case .bf16: return "Mixed BF16/F32 — recommended default. ~½ memory vs F32 with no measurable accuracy hit."
        case .f16:  return "Mixed F16/F32 — narrower dynamic range than BF16; may need loss scaling."
        case .f32:  return "Full F32 forward + backward — safest, but ~2× memory and slower throughput."
        }
    }
}

/// Input parameters for a full fine-tuning run. The "full" qualifier
/// here means *all weights are updated* — no LoRA / adapter split.
/// `FineTuner.run` consumes this spec and emits a stream of
/// `FineTuneEvent`s back to the caller.
public struct FineTuneSpec: Sendable {
    /// Path to the base model directory. Same layout the inference
    /// path expects: `model-*.safetensors` shards + `config.json` +
    /// `tokenizer.json`. The runner refuses to start if any of those
    /// is missing.
    public var baseModelPath: URL
    /// Path to the training dataset file (or directory of files when
    /// the format supports it).
    public var datasetPath: URL
    /// Optional eval dataset. When nil, a fraction of `datasetPath`
    /// is held out — see `evalSplit`.
    public var evalDatasetPath: URL?
    /// Directory the runner writes shard updates + an `epoch-N.json`
    /// training log to. Must be different from `baseModelPath` —
    /// the runner refuses to overwrite its own input.
    public var outputPath: URL

    /// Dataset layout selector.
    public var format: DatasetFormat
    /// Mixed-precision flavour for forward + backward.
    public var precision: TrainingPrecision

    // ---- Hyperparameters ----

    public var learningRate: Double
    public var epochs: Int
    public var batchSize: Int
    public var gradientAccumulationSteps: Int
    public var maxSequenceLength: Int
    public var warmupSteps: Int
    public var weightDecay: Double
    public var optimizer: FineTuneOptimizer
    /// Fraction of the training data reserved for eval when
    /// `evalDatasetPath` is nil. Ignored when an explicit eval set is
    /// provided.
    public var evalSplit: Double
    /// Snapshot the in-progress weights every N optimizer steps. 0
    /// disables intermediate snapshots (only the final checkpoint is
    /// emitted).
    public var saveEverySteps: Int
    /// RNG seed for the data shuffler + parameter init. Same seed
    /// reproduces the same training run.
    public var seed: UInt64

    public init(baseModelPath: URL,
                 datasetPath: URL,
                 evalDatasetPath: URL? = nil,
                 outputPath: URL,
                 format: DatasetFormat = .jsonlChat,
                 precision: TrainingPrecision = .bf16,
                 learningRate: Double = 5e-5,
                 epochs: Int = 3,
                 batchSize: Int = 1,
                 gradientAccumulationSteps: Int = 8,
                 maxSequenceLength: Int = 2048,
                 warmupSteps: Int = 100,
                 weightDecay: Double = 0.01,
                 optimizer: FineTuneOptimizer = .adamW,
                 evalSplit: Double = 0.05,
                 saveEverySteps: Int = 500,
                 seed: UInt64 = 42) {
        self.baseModelPath = baseModelPath
        self.datasetPath = datasetPath
        self.evalDatasetPath = evalDatasetPath
        self.outputPath = outputPath
        self.format = format
        self.precision = precision
        self.learningRate = learningRate
        self.epochs = epochs
        self.batchSize = batchSize
        self.gradientAccumulationSteps = gradientAccumulationSteps
        self.maxSequenceLength = maxSequenceLength
        self.warmupSteps = warmupSteps
        self.weightDecay = weightDecay
        self.optimizer = optimizer
        self.evalSplit = evalSplit
        self.saveEverySteps = saveEverySteps
        self.seed = seed
    }
}
