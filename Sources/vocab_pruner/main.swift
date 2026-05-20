import Foundation
import DeepSeekKit
import DeepSeekConverter
import DeepSeekVocabPruner

// MARK: - Usage

func usage() -> Never {
    let exe = (CommandLine.arguments.first as NSString?)?.lastPathComponent ?? "vocab_pruner"
    FileHandle.standardError.write("""
        Italiano-only vocab + expert pruner for DeepSeek-V4 checkpoints.

        Two independent phases, opt-in via flags:
          1. VOCAB phase (--corpus or --keep-ids): shrinks embed/head
             matrices to a smaller vocab. No model forward needed.
          2. EXPERT phase (--prune-experts + --calib-corpus or
             --expert-stats): drops rarely-used MoE experts. Requires
             a model forward pass (heavy).

        Both phases can run in the same command (chained via a temp
        directory), or each can run alone. The expert phase reads an
        already-vocab-pruned model the same way it reads a raw one —
        the two phases compose cleanly.

        Usage:
            \(exe) --input-dir <DIR> --output-dir <DIR>
                   [vocab phase opts] [expert phase opts]
                   [shared opts]

        Required:
            --input-dir <DIR>     Checkpoint convertito (output di `converter`),
                                  o l'output di un precedente vocab_pruner
                                  run. Deve contenere tokenizer.json,
                                  config.json, model.safetensors.index.json
                                  + shards.
            --output-dir <DIR>    Directory di destinazione (DEVE essere
                                  diversa da --input-dir).

        Vocab phase (opt-in):
            --corpus <PATH>       File .txt / .jsonl o directory ricorsiva.
                                  Triggera la vocab phase.
            --keep-ids <FILE>     keep_ids.json pre-computato. Salta
                                  l'analyzer; --corpus viene ignorato
                                  quando questo è settato.
            --coverage <FLOAT>    Copertura cumulativa target (0..1) per
                                  la vocab phase. Default 0.9995.
            --dry-run             Esegue solo l'analyzer vocab; non scrive
                                  output. Mutually exclusive con la
                                  expert phase.

        Expert phase (opt-in):
            --prune-experts       Abilita la fase di expert pruning.
            --calib-corpus <PATH> Corpus di calibrazione. Più piccolo
                                  del corpus vocab: bastano 1-10 MB di
                                  testo rappresentativo (la calibrazione
                                  è O(forward) per token).
            --expert-stats <FILE> expert_usage.json pre-computato (output
                                  di una run precedente o di uno script
                                  esterno). Salta l'analyzer expert;
                                  --calib-corpus viene ignorato.
            --expert-coverage <F> Coverage threshold per layer (0..1).
                                  Default 0.99 (tieni i top-K esperti
                                  che coprono il 99% delle routing).
            --min-experts-floor <N>
                                  Numero minimo di esperti tenuti per
                                  layer. Clampato comunque ≥
                                  n_activated_experts. Default 4.
            --max-calib-tokens <N>
                                  Cap totale di token processati durante
                                  la calibrazione. 0 (default) = no cap.
            --expert-dry-run      Esegue solo l'analyzer expert; non
                                  scrive l'output checkpoint, ma salva
                                  expert_usage.json.

        Shared:
            --concurrency <N>     Thread paralleli per la vocab phase.
                                  Default \(VocabPruneSpec.defaultConcurrency)
                                  (= 80% dei core attivi).
                                  Non ha effetto sulla expert phase
                                  (single-threaded GPU forward).
            --no-resume           Disabilita resume da checkpoint. Default
                                  legge `<output>/checkpoint/{vocab,expert}_pruner.json`.

        Esempi:
            # Solo vocab phase:
            \(exe) --input-dir ~/models/V4-Flash-converted \\
                   --output-dir ~/models/V4-Flash-it \\
                   --corpus ~/corpora/wikipedia-it \\
                   --coverage 0.9995

            # Solo expert phase (su vocab già fatto):
            \(exe) --input-dir ~/models/V4-Flash-it \\
                   --output-dir ~/models/V4-Flash-it-lean \\
                   --prune-experts \\
                   --calib-corpus ~/corpora/calib-it-small \\
                   --expert-coverage 0.99

            # Pipeline vocab + expert in un singolo comando:
            \(exe) --input-dir ~/models/V4-Flash-converted \\
                   --output-dir ~/models/V4-Flash-it-lean \\
                   --corpus ~/corpora/wikipedia-it --coverage 0.9995 \\
                   --prune-experts --calib-corpus ~/corpora/calib-it-small \\
                   --expert-coverage 0.99

        Vedi `docs/VOCAB-PRUNING.md` per i dettagli.

        """.data(using: .utf8)!)
    exit(2)
}

// MARK: - Args

var inputDir: String?
var outputDir: String?

// Vocab phase
var corpus: String?
var coverage: Double = 0.9995
var keepIdsFile: String?
var dryRun = false

// Expert phase
var pruneExperts = false
var calibCorpus: String?
var expertStatsFile: String?
var expertCoverage: Double = 0.99
var minExpertsFloor: Int = 4
var maxCalibTokens: Int = 0
var expertDryRun = false

// Shared
var concurrency = VocabPruneSpec.defaultConcurrency
var resume = true

var args = Array(CommandLine.arguments.dropFirst())
while !args.isEmpty {
    let a = args.removeFirst()
    switch a {
    case "--input-dir":
        guard !args.isEmpty else { usage() }
        inputDir = args.removeFirst()
    case "--output-dir":
        guard !args.isEmpty else { usage() }
        outputDir = args.removeFirst()
    case "--corpus":
        guard !args.isEmpty else { usage() }
        corpus = args.removeFirst()
    case "--coverage":
        guard !args.isEmpty, let v = Double(args.removeFirst()) else { usage() }
        coverage = v
    case "--keep-ids":
        guard !args.isEmpty else { usage() }
        keepIdsFile = args.removeFirst()
    case "--dry-run":
        dryRun = true
    case "--prune-experts":
        pruneExperts = true
    case "--calib-corpus":
        guard !args.isEmpty else { usage() }
        calibCorpus = args.removeFirst()
    case "--expert-stats":
        guard !args.isEmpty else { usage() }
        expertStatsFile = args.removeFirst()
    case "--expert-coverage":
        guard !args.isEmpty, let v = Double(args.removeFirst()) else { usage() }
        expertCoverage = v
    case "--min-experts-floor":
        guard !args.isEmpty, let v = Int(args.removeFirst()), v >= 1 else { usage() }
        minExpertsFloor = v
    case "--max-calib-tokens":
        guard !args.isEmpty, let v = Int(args.removeFirst()), v >= 0 else { usage() }
        maxCalibTokens = v
    case "--expert-dry-run":
        expertDryRun = true
    case "--concurrency":
        guard !args.isEmpty, let v = Int(args.removeFirst()), v >= 1 else { usage() }
        concurrency = v
    case "--no-resume":
        resume = false
    case "-h", "--help":
        usage()
    default:
        FileHandle.standardError.write("Unknown argument: \(a)\n".data(using: .utf8)!)
        usage()
    }
}

guard let inDir = inputDir, let outDir = outputDir else { usage() }

let inputURL = URL(fileURLWithPath: inDir)
let outputURL = URL(fileURLWithPath: outDir)

// Determine which phases run.
let runVocabPhase = (corpus != nil) || (keepIdsFile != nil)
let runExpertPhase = pruneExperts
guard runVocabPhase || runExpertPhase else {
    FileHandle.standardError.write(
        "Nothing to do: pass --corpus / --keep-ids (vocab phase) " +
        "and/or --prune-experts (expert phase).\n".data(using: .utf8)!)
    usage()
}
if runExpertPhase && pruneExperts {
    if calibCorpus == nil && expertStatsFile == nil {
        FileHandle.standardError.write(
            "--prune-experts requires either --calib-corpus or --expert-stats.\n"
                .data(using: .utf8)!)
        usage()
    }
}

// Pipeline plumbing: when both phases run, vocab writes to a temp
// directory that the expert phase consumes. The user's --output-dir
// receives the final (post-expert) output.
let vocabOutputURL: URL
if runVocabPhase && runExpertPhase {
    let tmpName = outputURL.lastPathComponent + ".vocab-stage"
    vocabOutputURL = outputURL.deletingLastPathComponent()
        .appendingPathComponent(tmpName)
} else {
    vocabOutputURL = outputURL
}

let token = CancellationToken()
let status = StatusPrinter()

do {
    // ---- VOCAB phase ----
    if runVocabPhase {
        let vocabSpec = VocabPruneSpec(
            inputDir: inputURL,
            outputDir: vocabOutputURL,
            corpus: corpus.map { URL(fileURLWithPath: $0) },
            coverage: coverage,
            keepIdsFile: keepIdsFile.map { URL(fileURLWithPath: $0) },
            dryRun: dryRun,
            concurrency: concurrency,
            resume: resume)
        try await VocabPruner.run(spec: vocabSpec, cancellation: token) { event in
            status.handle(event)
        }
        if dryRun {
            status.flush()
            print("Done (vocab dry-run).")
            exit(0)
        }
    }

    // ---- EXPERT phase ----
    if runExpertPhase {
        let expertInputURL = runVocabPhase ? vocabOutputURL : inputURL
        let expertSpec = ExpertPruneSpec(
            inputDir: expertInputURL,
            outputDir: outputURL,
            calibCorpus: calibCorpus.map { URL(fileURLWithPath: $0) },
            coverage: expertCoverage,
            minKeptFloor: minExpertsFloor,
            expertStatsFile: expertStatsFile.map { URL(fileURLWithPath: $0) },
            dryRun: expertDryRun,
            maxCalibrationTokens: maxCalibTokens,
            resume: resume)
        try await ExpertPruner.run(spec: expertSpec, cancellation: token) { event in
            status.handle(event)
        }

        // Clean up the intermediate vocab-stage directory once the
        // expert phase has consumed it. Skipped on dry-run so the
        // user can still inspect the vocab output.
        if runVocabPhase && !expertDryRun {
            try? FileManager.default.removeItem(at: vocabOutputURL)
            FileHandle.standardError.write(
                "Cleaned intermediate \(vocabOutputURL.path)\n".data(using: .utf8)!)
        }
    }

    status.flush()
    print("Done.")
} catch {
    FileHandle.standardError.write("ERROR: \(error.localizedDescription)\n".data(using: .utf8)!)
    exit(1)
}

// MARK: - StatusPrinter

/// Renderer minimale degli eventi: stampa progressi a stderr,
/// summary finale a stdout. Mantiene un `\r` per le linee
/// aggiornabili (scanned / shardWritten) e un `\n` per i log.
final class StatusPrinter: @unchecked Sendable {
    private let lock = NSLock()
    private var lastScannedLine = ""
    private var lastShardLine = ""

    func handle(_ event: VocabPruneEvent) {
        lock.lock(); defer { lock.unlock() }
        switch event {
        case .scanned(let lines, let tokens):
            lastScannedLine = "  scanning: \(lines) lines, \(tokens) tokens"
            FileHandle.standardError.write("\r\(lastScannedLine)".data(using: .utf8)!)
        case .coverage(let pct, let kept, let total):
            let pctStr = String(format: "%.2f%%", pct * 100)
            let line = "  coverage: \(pctStr) — keeping \(kept) of \(total) tokens"
            FileHandle.standardError.write("\n\(line)\n".data(using: .utf8)!)
            lastScannedLine = ""
        case .decisionReady(let decision):
            let preview = decision.previewDropped.prefix(5)
            if !preview.isEmpty {
                let head = preview.map { "  - '\($0.content)' (id=\($0.id), count=\($0.count))" }
                    .joined(separator: "\n")
                FileHandle.standardError.write(
                    "  top-5 dropped tokens (by frequency):\n\(head)\n"
                        .data(using: .utf8)!)
            }
        case .shardWritten(let i, let total):
            lastShardLine = "  shard \(i)/\(total)"
            FileHandle.standardError.write("\r\(lastShardLine)".data(using: .utf8)!)
        case .log(let line):
            if !lastScannedLine.isEmpty || !lastShardLine.isEmpty {
                FileHandle.standardError.write("\n".data(using: .utf8)!)
                lastScannedLine = ""
                lastShardLine = ""
            }
            FileHandle.standardError.write("\(line)\n".data(using: .utf8)!)
        case .finished(let bytesIn, let bytesOut, let vIn, let vOut):
            if !lastShardLine.isEmpty {
                FileHandle.standardError.write("\n".data(using: .utf8)!)
                lastShardLine = ""
            }
            let mbIn = Double(bytesIn) / 1_000_000
            let mbOut = Double(bytesOut) / 1_000_000
            let ratio = bytesIn > 0 ? 100.0 * Double(bytesOut) / Double(bytesIn) : 0
            print(String(format:
                "Finished vocab phase. vocab: %d → %d. size: %.1f MB → %.1f MB (%.1f%%).",
                vIn, vOut, mbIn, mbOut, ratio))
        case .expertDecisionReady(let decision):
            if !lastScannedLine.isEmpty || !lastShardLine.isEmpty {
                FileHandle.standardError.write("\n".data(using: .utf8)!)
                lastScannedLine = ""
                lastShardLine = ""
            }
            let totalRouted = decision.nLayers * decision.nRoutedExperts
            FileHandle.standardError.write(Data(
                "  expert decision: drop \(decision.totalDropped) / " +
                "\(totalRouted) (= \(decision.totalKept) kept) " +
                "across \(decision.nLayers) layers\n".utf8))
            // Per-layer breakdown.
            for L in 0..<decision.nLayers {
                let kept = decision.keepIds[L].count
                let dropped = decision.droppedIds[L].count
                let cov = decision.actualCoveragePerLayer[L] * 100
                FileHandle.standardError.write(Data(String(format:
                    "    layer %d: kept=%d, dropped=%d, coverage=%.2f%%\n",
                    L, kept, dropped, cov).utf8))
            }
        case .expertFinished(let bytesIn, let bytesOut, let dropped, let kept):
            if !lastShardLine.isEmpty {
                FileHandle.standardError.write("\n".data(using: .utf8)!)
                lastShardLine = ""
            }
            let mbIn = Double(bytesIn) / 1_000_000
            let mbOut = Double(bytesOut) / 1_000_000
            let ratio = bytesIn > 0 ? 100.0 * Double(bytesOut) / Double(bytesIn) : 0
            print(String(format:
                "Finished expert phase. experts: dropped=%d, kept=%d. " +
                "size: %.1f MB → %.1f MB (%.1f%%).",
                dropped, kept, mbIn, mbOut, ratio))
        }
    }

    func flush() {
        lock.lock(); defer { lock.unlock() }
        if !lastScannedLine.isEmpty || !lastShardLine.isEmpty {
            FileHandle.standardError.write("\n".data(using: .utf8)!)
        }
    }
}
