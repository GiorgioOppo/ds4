import Foundation

/// Native conversation controls used by GLM 5.2 (`general.architecture = glm-dsa`).
///
/// These are deliberately kept outside the DeepSeek DSML implementation: both
/// families use XML-looking tool calls, but their role framing and grammars are
/// not interchangeable.
public enum GLM52ConversationProtocol {
    public static let mask = "[gMASK]"
    public static let startOfPrompt = "<sop>"
    public static let endOfText = "<|endoftext|>"

    public static let system = "<|system|>"
    public static let user = "<|user|>"
    public static let assistant = "<|assistant|>"
    public static let observation = "<|observation|>"

    public static let thinkOpen = "<think>"
    public static let thinkClose = "</think>"
    public static let toolCallOpen = "<tool_call>"
    public static let toolCallClose = "</tool_call>"
    public static let toolResponseOpen = "<tool_response>"
    public static let toolResponseClose = "</tool_response>"
    public static let argumentKeyOpen = "<arg_key>"
    public static let argumentKeyClose = "</arg_key>"
    public static let argumentValueOpen = "<arg_value>"
    public static let argumentValueClose = "</arg_value>"

    /// Literal controls recognized atomically by `GLM52Tokenizer` when it
    /// consumes an already-rendered transcript.
    public static let controlTokens: [String] = [
        mask, startOfPrompt, endOfText,
        system, user, assistant, observation,
        thinkOpen, thinkClose,
        toolCallOpen, toolCallClose,
        toolResponseOpen, toolResponseClose,
        argumentKeyOpen, argumentKeyClose,
        argumentValueOpen, argumentValueClose,
    ]

    /// Break literal protocol controls in untrusted content without changing
    /// their visible text. This prevents a user or tool payload from creating a
    /// role boundary when the completed transcript is special-token scanned.
    public static func neutralizeControlTokens(
        in text: String,
        preserving trustedTokens: Set<String> = []
    ) -> String {
        var output = text
        for token in controlTokens
            .filter({ !trustedTokens.contains($0) })
            .sorted(by: { $0.utf8.count > $1.utf8.count })
        {
            guard output.contains(token) else { continue }
            let scalars = Array(token.unicodeScalars)
            guard scalars.count > 1 else {
                guard let scalar = scalars.first else { continue }
                output = output.replacingOccurrences(
                    of: token,
                    with: "\\u{\(String(scalar.value, radix: 16, uppercase: true))}"
                )
                continue
            }
            let broken = scalars.enumerated().map { index, scalar in
                (index == 0 ? "" : "\u{2060}") + String(scalar)
            }.joined()
            output = output.replacingOccurrences(of: token, with: broken)
        }
        return output
    }
}

/// GLM-specific prompt mapping for the architecture-neutral reasoning mode.
public extension ThinkMode {
    var glm52EffortText: String? {
        switch self {
        case .none: return nil
        case .high: return "Reasoning Effort: High"
        case .max: return "Reasoning Effort: Max"
        }
    }
}
