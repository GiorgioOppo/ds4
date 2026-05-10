import Foundation

/// Minimal-viable port of `Reference/encoding/encoding_dsv4.py` (744 lines).
///
/// This implements ONLY the plain chat path:
///   - one or more user/assistant turns
///   - no tool calls
///   - no `<think>` blocks (chat / Non-think mode)
///   - no system messages with task tokens
///   - no response_format / latest_reminder injection
///
/// The full reference handles tool calling DSML, three thinking effort
/// modes, task tokens, response format schemas, and latest-reminder
/// injection — all of which are stubbed here. When the production CLI
/// needs them they should be ported in chunks, with each chunk validated
/// against the golden tests under `Reference/encoding/tests/`.
public enum EncodingDSV4 {
    public static let bosToken = "<｜begin▁of▁sentence｜>"
    public static let eosToken = "<｜end▁of▁sentence｜>"
    public static let userToken = "<｜User｜>"
    public static let assistantToken = "<｜Assistant｜>"
    public static let thinkOpen = "<think>"
    public static let thinkClose = "</think>"

    /// Encode a list of messages into the prompt string the model expects.
    /// `mode == .chat` produces the Non-think format. `.high` / `.max`
    /// emit the `<think>` markers but currently leave the prompt the same
    /// since the per-mode system instructions aren't yet ported.
    public static func encodeMessages(_ messages: [Message],
                                       mode: ThinkingMode = .chat) -> String {
        var out = bosToken
        for msg in messages {
            switch msg.role {
            case .system:
                out += msg.content
            case .user:
                out += userToken + msg.content
            case .assistant:
                out += assistantToken
                if let r = msg.reasoningContent, !r.isEmpty, mode != .chat {
                    out += thinkOpen + r + thinkClose
                }
                out += msg.content + eosToken
            }
        }
        // Open the final assistant turn for generation.
        out += assistantToken
        return out
    }

    /// Parse a model completion back into a `Message`. Splits on optional
    /// `<think>...</think>` and treats whatever follows as the assistant's
    /// content. Trailing EOS is stripped.
    public static func parseCompletion(_ text: String,
                                        mode: ThinkingMode = .chat) -> Message {
        var work = text
        if work.hasSuffix(eosToken) {
            work = String(work.dropLast(eosToken.count))
        }
        var reasoning: String? = nil
        if let openRange = work.range(of: thinkOpen),
           let closeRange = work.range(of: thinkClose, range: openRange.upperBound..<work.endIndex) {
            reasoning = String(work[openRange.upperBound..<closeRange.lowerBound])
            work.removeSubrange(openRange.lowerBound..<closeRange.upperBound)
        }
        return Message(role: .assistant,
                       content: work.trimmingCharacters(in: .whitespacesAndNewlines),
                       reasoningContent: reasoning)
    }
}
