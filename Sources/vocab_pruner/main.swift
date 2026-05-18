import Foundation
import DeepSeekKit
import DeepSeekConverter
import DeepSeekVocabPruner

// MARK: - Usage

func usage() -> Never {
    let exe = (CommandLine.arguments.first as NSString?)?.lastPathComponent ?? "vocab_pruner"
    FileHandle.standardError.write("""
        Italiano-only vocabulary pruner for DeepSeek-V4 checkpoints.

        Usage:
            \(exe) --input-dir <DIR> --output-dir <DIR> --corpus <PATH>
                   [--coverage 0.9995]
                   [--keep-ids <FILE>]
                   [--dry-run]

        Required:
            --input-dir <DIR>     Checkpoint convertito (output di `converter`).
                                  Deve contenere tokenizer.json, config.json,
                                  model.safetensors.index.json + shards.
            --output-dir <DIR>    Directory di destinazione (DEVE essere
                                  diversa da --input-dir).
            --corpus <PATH>       File .txt / .jsonl o directory walkata
                                  ricorsivamente per .txt/.jsonl. Saltato
                                  se è settato --keep-ids.

        Optional:
            --coverage <FLOAT>    Copertura cumulativa target (0..1).
                                  Default 0.9995 (99.95%).
            --keep-ids <FILE>     keep_ids.json pre-computato. Se settato,
                                  --corpus viene ignorato.
            --dry-run             Esegue solo la Fase 1 (analyzer) e stampa
                                  la statistica di copertura. Niente
                                  scrittura di output.

        Esempio:
            \(exe) --input-dir ~/models/V4-Flash-converted \\
                   --output-dir ~/models/V4-Flash-it \\
                   --corpus ~/corpora/wikipedia-it \\
                   --coverage 0.9995

        Vedi `docs/VOCAB-PRUNING.md` per i dettagli.

        """.data(using: .utf8)!)
    exit(2)
}

// MARK: - Args

var inputDir: String?
var outputDir: String?
var corpus: String?
var coverage: Double = 0.9995
var keepIdsFile: String?
var dryRun = false

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
    case "-h", "--help":
        usage()
    default:
        FileHandle.standardError.write("Unknown argument: \(a)\n".data(using: .utf8)!)
        usage()
    }
}

guard let inDir = inputDir, let outDir = outputDir else { usage() }
guard corpus != nil || keepIdsFile != nil else {
    FileHandle.standardError.write("Either --corpus or --keep-ids is required.\n".data(using: .utf8)!)
    usage()
}

let inputURL = URL(fileURLWithPath: inDir)
let outputURL = URL(fileURLWithPath: outDir)
let corpusURL = corpus.map { URL(fileURLWithPath: $0) }
let keepURL = keepIdsFile.map { URL(fileURLWithPath: $0) }

let spec = VocabPruneSpec(
    inputDir: inputURL,
    outputDir: outputURL,
    corpus: corpusURL,
    coverage: coverage,
    keepIdsFile: keepURL,
    dryRun: dryRun)

// MARK: - Run + render eventi

let token = CancellationToken()
let status = StatusPrinter()

do {
    try VocabPruner.runSync(spec: spec, cancellation: token) { event in
        status.handle(event)
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
            // CLI: log compatto del top-5 dropped per dare visibilità
            // del prunin a chi non guarda la GUI.
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
                "Finished. vocab: %d → %d. size: %.1f MB → %.1f MB (%.1f%%).",
                vIn, vOut, mbIn, mbOut, ratio))
        }
    }

    func flush() {
        lock.lock(); defer { lock.unlock() }
        if !lastScannedLine.isEmpty || !lastShardLine.isEmpty {
            FileHandle.standardError.write("\n".data(using: .utf8)!)
        }
    }
}
