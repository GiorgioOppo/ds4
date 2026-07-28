import Foundation

/// Native conversation controls used by Laguna S 2.1
/// (`general.architecture = laguna`).
///
/// Laguna frames roles with XML-looking text tags: only `<assistant>`,
/// `</assistant>`, `<think>`, `</think>`, `<tool_call>` and `</tool_call>` are
/// dedicated vocabulary control tokens, while `<system>`, `<user>`,
/// `<tool_response>` and the argument tags are ordinary BPE text that the
/// model was trained to interpret.  The official template tokenizes each
/// rendered message as contiguous text, so BPE merges are allowed across
/// tag/content boundaries (for example `>You` and `.</`).
public enum LagunaConversationProtocol {
    /// Textual sequence-start marker used by the reference server renderer.
    /// Poolside's tokenizer reuses one control token for BOS and EOS, and the
    /// upstream special-token scanner maps this literal onto the EOS id.
    public static let bosMarker = "〈|EOS|〉"

    public static let systemOpen = "<system>"
    public static let systemClose = "</system>"
    public static let userOpen = "<user>"
    public static let userClose = "</user>"
    public static let assistantOpen = "<assistant>"
    public static let assistantClose = "</assistant>"

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
    public static let availableToolsOpen = "<available_tools>"
    public static let availableToolsClose = "</available_tools>"

    /// Default system prompt used by the reference implementation when the
    /// caller does not provide one.
    public static let defaultSystemPrompt =
        "You are a helpful, conversationally-fluent assistant made by Poolside. "
        + "You are here to be helpful to users through natural language conversations."

    /// Reference sampling defaults (`ds4_engine_sampling_defaults` for the
    /// Laguna family).  Explicit caller options always take precedence.
    public enum SamplingDefaults {
        public static let temperature: Float = 0.7
        public static let topK = 20
        public static let topP: Float = 0.95
        public static let minP: Float = 0.05
    }

    /// Literal controls recognized atomically by `LagunaTokenizer` when it
    /// consumes an already-rendered transcript.  This is the exact Laguna
    /// subset of the upstream special-token scanner; the role/argument text
    /// tags are deliberately absent so they keep tokenizing as plain BPE text.
    public static let atomicControlTokens: [String] = [
        bosMarker,
        assistantOpen, assistantClose,
        thinkOpen, thinkClose,
        toolCallOpen, toolCallClose,
    ]

    /// Everything that can open or close a protocol scope, including the
    /// plain-text role tags.  Untrusted content must not be able to fabricate
    /// any of these boundaries: the textual tags steer the model even though
    /// they are not dedicated vocabulary tokens.
    public static let controlTokens: [String] = atomicControlTokens + [
        systemOpen, systemClose,
        userOpen, userClose,
        toolResponseOpen, toolResponseClose,
        argumentKeyOpen, argumentKeyClose,
        argumentValueOpen, argumentValueClose,
        availableToolsOpen, availableToolsClose,
    ]

    /// Break literal protocol controls in untrusted content without changing
    /// their visible text. This prevents a user or tool payload from creating a
    /// role boundary when the completed transcript is special-token scanned or
    /// re-read by the model as plain text.
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
