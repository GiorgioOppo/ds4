import DS4Core
import DS4Metal
import Foundation

/// Chat inference over the GLM 5.2 resident engine — the deliberate
/// separate-backend counterpart of the DeepSeek `InferenceService`: no code
/// or state is shared with the DeepSeek hot loop. Greedy-only and
/// validation-grade (per-dispatch executor speed); the optimization tranches
/// come after the full-model parity gate.
public final class GLM52InferenceService {
    public let tokenizer: GLM52Tokenizer
    public let engine: GLM52ResidentModel

    public init(modelPath: String,
                options: GLM52ResidentModelOptions = .init()) throws {
        if let issue = ModelFileDiagnostics.openabilityIssue(path: modelPath) {
            throw GGUFError.cannotOpen(issue)
        }
        let model = try GGUFModel(path: modelPath, metalMapping: false,
                                  prefetchCPU: false)
        tokenizer = try GLM52Tokenizer(model: model)
        let runtime = try MetalRuntime()
        engine = try GLM52ResidentModel(runtime: runtime, path: modelPath,
                                        options: options)
    }

    /// Greedy chat completion through the native GLM template. `onToken`
    /// receives each decoded text piece as it is produced.
    @discardableResult
    public func generate(system: String? = nil,
                         prompt: String,
                         reasoning: ThinkMode = .none,
                         maxNewTokens: Int = 256,
                         onToken: ((String) -> Void)? = nil) throws -> String {
        let promptTokens = try tokenizer.encodeChatPrompt(
            system: system, prompt: prompt, reasoning: reasoning)
        var logits = try engine.prefill(promptTokens)
        var produced: [UInt8] = []
        var count = 0
        while count < maxNewTokens {
            guard let token = GLM52GreedyDecoding.argmax(logits) else { break }
            count += 1
            if tokenizer.isStopToken(token, reasoning: reasoning) { break }
            let bytes = tokenizer.tokenText(token)
            produced.append(contentsOf: bytes)
            if let onToken, let piece = String(bytes: bytes,
                                               encoding: .utf8) {
                onToken(piece)
            }
            if count == maxNewTokens { break }
            logits = try engine.forwardNext(token)
        }
        return String(decoding: produced, as: UTF8.self)
    }
}
