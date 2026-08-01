import Foundation

enum ConversionMode: String, CaseIterable, Identifiable {
    case template = "Modello da template"
    case dspark = "Supporto DSpark"
    var id: String { rawValue }
}

enum ConversionQuant: String, CaseIterable, Identifiable {
    case template = "Dal template"
    case iq2XXS = "iq2_xxs"
    case q2K = "q2_k"
    case q4K = "q4_k"
    case q8 = "q8_0"
    case q8K = "q8_K"
    case f16 = "f16"
    var id: String { rawValue }
    var argument: String? { self == .template ? nil : rawValue }
}

/// UI adapter for antirez/ds4's offline safetensors -> GGUF converter. The C
/// tool remains the source of truth for tensor-name mapping, FP4/FP8 decoding
/// and DeepSeek-specific recipes; this class only validates and launches it.
@MainActor
@Observable
final class SafetensorsConversionController {
    var hfDirectory = ""
    var templatePath = ""
    var outputPath = ""
    var imatrixPath = ""
    var compareTensor = ""
    var converterPath = SafetensorsConversionController.defaultConverterPath
    var mode: ConversionMode = .template
    var experts: ConversionQuant = .template
    var routedW2: ConversionQuant = .template
    var attention: ConversionQuant = .template
    var shared: ConversionQuant = .template
    var output: ConversionQuant = .template
    var threads = max(1, ProcessInfo.processInfo.activeProcessorCount / 2)
    var dryRun = true
    var overwrite = false
    var log = ""
    var isRunning = false

    private let process = ProcessStream()

    static var defaultConverterPath: String {
        let candidates = [
            (AppEnvironment.resourceDir as NSString).appendingPathComponent("gguf-tools/deepseek4-quantize"),
            (AppEnvironment.resourceDir as NSString).appendingPathComponent("bin/deepseek4-quantize"),
            (AppEnvironment.projectRoot as NSString).appendingPathComponent("gguf-tools/deepseek4-quantize")
        ]
        return candidates.first(where: FileManager.default.isExecutableFile(atPath:)) ?? candidates[0]
    }

    var canStart: Bool {
        !isRunning && !hfDirectory.isEmpty && !outputPath.isEmpty
            && (mode == .dspark || !templatePath.isEmpty)
    }

    func start() {
        guard !isRunning else { return }
        guard FileManager.default.fileExists(atPath: hfDirectory) else {
            log = "Errore: seleziona una directory Hugging Face valida.\n"; return
        }
        guard mode == .dspark || FileManager.default.isReadableFile(atPath: templatePath) else {
            log = "Errore: seleziona il GGUF template.\n"; return
        }
        guard FileManager.default.isExecutableFile(atPath: converterPath) else {
            log = "Converter non trovato o non eseguibile: \(converterPath)\nCompilalo con: make -C gguf-tools\n"
            return
        }
        if FileManager.default.fileExists(atPath: outputPath), !overwrite, !dryRun {
            log = "Errore: il file di output esiste già. Abilita ‘Sovrascrivi’ oppure scegli un altro nome.\n"
            return
        }

        let arguments = makeArguments()
        log = "$ \(([converterPath] + arguments).map(Self.shellQuoted).joined(separator: " "))\n\n"
        isRunning = true
        if let error = process.start(
            executable: converterPath,
            arguments: arguments,
            workingDir: AppEnvironment.resourceDir,
            onOutput: { [weak self] chunk in self?.log += chunk },
            onExit: { [weak self] status in
                self?.isRunning = false
                self?.log += "\nProcesso terminato con codice \(status).\n"
            }
        ) {
            isRunning = false
            log += "Avvio fallito: \(error)\n"
        }
    }

    func cancel() {
        guard isRunning else { return }
        log += "\nInterruzione richiesta…\n"
        process.interrupt()
    }

    func applyQ2Recipe() {
        experts = .iq2XXS; routedW2 = .q2K; attention = .q8; shared = .q8; output = .q8
    }

    func applyQ4Recipe() {
        experts = .q4K; routedW2 = .template; attention = .q8; shared = .q8; output = .q8
    }

    func applyQ8KRecipe() {
        experts = .q8K; routedW2 = .template; attention = .template; shared = .template; output = .template
    }

    private func makeArguments() -> [String] {
        var args = ["--hf", hfDirectory]
        if mode == .dspark {
            args.append("--dspark-support")
        } else {
            args += ["--template", templatePath]
            append(experts.argument, flag: "--experts", to: &args)
            append(routedW2.argument, flag: "--routed-w2", to: &args)
            append(attention.argument, flag: "--attention-proj", to: &args)
            append(shared.argument, flag: "--shared", to: &args)
            append(output.argument, flag: "--output", to: &args)
            if !imatrixPath.isEmpty { args += ["--imatrix", imatrixPath] }
        }
        args += ["--out", outputPath, "--threads", String(max(1, threads))]
        if !compareTensor.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            args += ["--compare-tensor", compareTensor.trimmingCharacters(in: .whitespacesAndNewlines)]
        }
        if dryRun { args.append("--dry-run") }
        if overwrite { args.append("--overwrite") }
        return args
    }

    private func append(_ value: String?, flag: String, to args: inout [String]) {
        if let value { args += [flag, value] }
    }

    private static func shellQuoted(_ value: String) -> String {
        guard value.contains(where: { $0.isWhitespace || "'\"\\$".contains($0) }) else { return value }
        return "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }
}
