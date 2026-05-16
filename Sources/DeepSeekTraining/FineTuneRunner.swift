import Foundation

/// Stub implementation of the fine-tuning runner.
///
/// The native Metal engine in DeepSeekKit does not yet implement
/// backward kernels — see `TODO.md` §2 and §5 — so this runner
/// cannot actually train weights today. What it *does* do is:
///
///   1. Validate the spec (paths exist, output is distinct, dataset
///      is well-formed for the declared format).
///   2. Emit a `discovered` event with a rough example count by
///      scanning the dataset.
///   3. Print a short "what would happen" plan (LR schedule, total
///      steps, optimizer choice) so the user can sanity-check their
///      hyperparameters before a real backend lands.
///   4. Throw `FineTuneNotImplemented` so the UI can render a
///      friendly explanation instead of silently doing nothing.
///
/// When a real backend (native Metal training, or an external CLI
/// like a future `Sources/trainer` executable) is wired up, swap the
/// `throw` at the end for the actual training loop. The
/// `FineTuneSpec` + `FineTuneEvent` surface area is designed to be
/// stable across that swap so callers don't need to change.
public enum FineTuneRunner {

    public static func run(
        spec: FineTuneSpec,
        cancellation: TrainingCancellationToken,
        onEvent: @escaping @Sendable (FineTuneEvent) -> Void
    ) throws {
        let fm = FileManager.default

        // ---- 1. Validate ----

        guard fm.fileExists(atPath: spec.baseModelPath.path) else {
            throw NSError(
                domain: "FineTuneRunner", code: 10,
                userInfo: [NSLocalizedDescriptionKey:
                    "Base model directory does not exist: \(spec.baseModelPath.path)"])
        }
        let configURL = spec.baseModelPath.appendingPathComponent("config.json")
        guard fm.fileExists(atPath: configURL.path) else {
            throw NSError(
                domain: "FineTuneRunner", code: 11,
                userInfo: [NSLocalizedDescriptionKey:
                    "Base model directory is missing config.json — expected a converted DeepSeek checkpoint."])
        }
        guard fm.fileExists(atPath: spec.datasetPath.path) else {
            throw NSError(
                domain: "FineTuneRunner", code: 12,
                userInfo: [NSLocalizedDescriptionKey:
                    "Dataset path does not exist: \(spec.datasetPath.path)"])
        }
        if let evalPath = spec.evalDatasetPath {
            guard fm.fileExists(atPath: evalPath.path) else {
                throw NSError(
                    domain: "FineTuneRunner", code: 13,
                    userInfo: [NSLocalizedDescriptionKey:
                        "Eval dataset path does not exist: \(evalPath.path)"])
            }
        }
        guard spec.baseModelPath.standardizedFileURL
                != spec.outputPath.standardizedFileURL else {
            throw NSError(
                domain: "FineTuneRunner", code: 14,
                userInfo: [NSLocalizedDescriptionKey:
                    "Output directory must be different from the base model directory — refusing to overwrite the source."])
        }
        try fm.createDirectory(at: spec.outputPath,
                                withIntermediateDirectories: true)

        try checkCancelled(cancellation)
        onEvent(.log("Validated paths."))
        onEvent(.log("Base model: \(spec.baseModelPath.path)"))
        onEvent(.log("Dataset:    \(spec.datasetPath.path)"))
        onEvent(.log("Output:     \(spec.outputPath.path)"))

        // ---- 2. Enumerate dataset ----

        let (exampleCount, datasetBytes) = try countExamples(
            at: spec.datasetPath, format: spec.format)
        onEvent(.discovered(examples: exampleCount, bytes: datasetBytes))
        onEvent(.log("Found \(exampleCount) examples (\(formatBytes(datasetBytes)))."))

        try checkCancelled(cancellation)

        // ---- 3. Print plan ----

        let effectiveBatch = spec.batchSize * spec.gradientAccumulationSteps
        let stepsPerEpoch = max(1, exampleCount / max(1, effectiveBatch))
        let totalSteps = stepsPerEpoch * spec.epochs

        onEvent(.log(""))
        onEvent(.log("Plan:"))
        onEvent(.log("  Optimizer:        \(spec.optimizer.displayName)"))
        onEvent(.log("  Precision:        \(spec.precision.displayName)"))
        onEvent(.log("  Learning rate:    \(formatLR(spec.learningRate))"))
        onEvent(.log("  Weight decay:     \(spec.weightDecay)"))
        onEvent(.log("  Warmup steps:     \(spec.warmupSteps)"))
        onEvent(.log("  Epochs:           \(spec.epochs)"))
        onEvent(.log("  Batch (micro):    \(spec.batchSize)"))
        onEvent(.log("  Grad-accum:       \(spec.gradientAccumulationSteps)"))
        onEvent(.log("  Effective batch:  \(effectiveBatch)"))
        onEvent(.log("  Max seq len:      \(spec.maxSequenceLength)"))
        onEvent(.log("  Steps / epoch:    \(stepsPerEpoch)"))
        onEvent(.log("  Total steps:      \(totalSteps)"))
        onEvent(.log("  Eval split:       \(formatPct(spec.evalSplit))"))
        if spec.saveEverySteps > 0 {
            onEvent(.log("  Snapshot every:   \(spec.saveEverySteps) steps"))
        }
        onEvent(.log("  Seed:             \(spec.seed)"))
        onEvent(.log(""))

        try checkCancelled(cancellation)

        // ---- 4. Bail out: no backend yet ----

        throw FineTuneNotImplemented("""
        Native fine-tuning is not wired up yet — the Metal engine \
        currently implements only forward kernels. \

        To run this training plan you'll need one of:
          • A future native trainer in DeepSeekKit (TODO.md §2/§5 — \
            backward kernels, optimizer state shard, gradient \
            checkpointing).
          • An external CLI (e.g. `mlx_lm.lora`, llama.cpp finetune) \
            invoked through this runner — wire it in \
            `FineTuneRunner.run` once the binary is on $PATH.

        The form / plan above was validated successfully, so when a \
        backend lands the same UI flow will drive a real training run \
        without changes.
        """)
    }

    // ---- Helpers ----

    private static func checkCancelled(_ token: TrainingCancellationToken) throws {
        if token.isCancelled { throw FineTuneCancelled() }
    }

    /// Best-effort example counter. For JSONL we count `\n`-separated
    /// non-empty lines; for plain text we estimate based on byte
    /// size / max-seq-len (rough but enough for a status line).
    private static func countExamples(at url: URL,
                                        format: DatasetFormat) throws -> (Int, UInt64) {
        let fm = FileManager.default
        guard let attrs = try? fm.attributesOfItem(atPath: url.path) else {
            return (0, 0)
        }
        let bytes = (attrs[.size] as? UInt64) ?? 0
        switch format {
        case .jsonlChat, .jsonlPromptCompletion:
            guard let handle = try? FileHandle(forReadingFrom: url) else {
                return (0, bytes)
            }
            defer { try? handle.close() }
            var n = 0
            while let chunk = try? handle.read(upToCount: 1 << 20),
                  !chunk.isEmpty {
                for byte in chunk where byte == 0x0A { n += 1 }
            }
            return (n, bytes)
        case .plainText:
            // Approx: one "example" per ~max-seq-len chars.
            return (max(1, Int(bytes / 2048)), bytes)
        }
    }

    private static func formatBytes(_ b: UInt64) -> String {
        let gib = 1024.0 * 1024.0 * 1024.0
        let mib = 1024.0 * 1024.0
        if Double(b) >= gib { return String(format: "%.2f GB", Double(b) / gib) }
        if Double(b) >= mib { return String(format: "%.2f MB", Double(b) / mib) }
        return "\(b) B"
    }

    private static func formatLR(_ lr: Double) -> String {
        String(format: "%.2e", lr)
    }

    private static func formatPct(_ f: Double) -> String {
        String(format: "%.1f%%", f * 100)
    }
}
