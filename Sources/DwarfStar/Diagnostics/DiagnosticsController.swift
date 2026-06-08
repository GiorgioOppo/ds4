import Foundation
import DS4Engine

/// Tokenizer diagnostic — NATIVE (DS4Engine.Diagnostics over DS4Core.Tokenizer),
/// no `ds4 --dump-tokens` subprocess. Tokenizes a string exactly as written
/// (recognizing DS4 protocol specials). Opens the GGUF for its tokenizer tables.
@MainActor
@Observable
final class DiagnosticsController {
    var modelPath = AppEnvironment.defaultModelPath
    var text = "Ciao, come stai?"
    var output = ""
    var isRunning = false

    private var task: Task<Void, Never>?

    func dumpTokens() {
        guard !isRunning else { return }
        output = ""
        isRunning = true
        let path = modelPath, t = text
        task = Task {
            do {
                let dump = try await Task.detached { try Diagnostics.dumpTokens(modelPath: path, text: t) }.value
                output = dump
            } catch is CancellationError {
                output = "[annullato]\n"
            } catch {
                output = "Errore: \(error)\n"
            }
            isRunning = false
        }
    }

    func cancel() { task?.cancel() }

    /// Dump the model's chat_template + tool-markup report (to align the tool
    /// format with what the model was actually trained on).
    func dumpChatTemplate() {
        guard !isRunning else { return }
        output = ""
        isRunning = true
        let path = modelPath
        task = Task {
            do {
                let dump = try await Task.detached { try Diagnostics.dumpChatTemplate(modelPath: path) }.value
                output = dump
            } catch is CancellationError {
                output = "[annullato]\n"
            } catch {
                output = "Errore: \(error)\n"
            }
            isRunning = false
        }
    }
}
