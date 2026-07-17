import Foundation

/// Pure GLM 5.2 chat-template renderer. It mirrors the native `glm-dsa`
/// framing while keeping tool syntax and untrusted-payload containment separate
/// from DeepSeek's DSML renderer.
public enum GLM52ChatRenderer {
    /// Render a complete transcript. GLM uses the next role marker as the turn
    /// boundary, so historical assistant turns do not receive DeepSeek's EOS.
    public static func render(
        turns: [ChatTurn],
        tools: [ToolSpec] = [],
        reasoning: ThinkMode = .none,
        addGenerationPrompt: Bool = true,
        neutralizeUntrustedContent: Bool = true
    ) throws -> String {
        let p = GLM52ConversationProtocol.self
        let safe: (String) -> String = { text in
            neutralizeUntrustedContent
                ? p.neutralizeControlTokens(in: text)
                : text
        }

        var output = p.mask + p.startOfPrompt
        if let effort = reasoning.glm52EffortText {
            output += p.system + effort
        }
        if !tools.isEmpty {
            output += p.system + (try GLM52ToolCodec.toolsPrompt(tools))
        }
        var pendingAssistant = false
        var previousWasToolResult = false
        for turn in turns {
            switch turn {
            case .system(let content):
                // `tokenizer.chat_template` renders messages in input order.
                // In particular, a later system/developer message is not part
                // of a hoisted prologue.
                output += p.system + safe(content)
                previousWasToolResult = false
            case .user(let content):
                output += p.user + safe(content)
                pendingAssistant = true
                previousWasToolResult = false
            case .toolResult(_, _, let content):
                // Consecutive tool messages share one observation role marker,
                // but retain one tool-response wrapper per result.
                if !previousWasToolResult { output += p.observation }
                output += p.toolResponseOpen
                output += neutralizeUntrustedContent
                    ? GLM52ToolCodec.escapeToolResponse(content)
                    : content.replacingOccurrences(
                        of: p.toolResponseClose,
                        with: "&lt;" + p.toolResponseClose.dropFirst()
                    )
                output += p.toolResponseClose
                pendingAssistant = true
                previousWasToolResult = true
            case .assistant(let text, let calls):
                output += p.assistant
                let trustedThink = Set([p.thinkOpen, p.thinkClose])
                let assistantText = neutralizeUntrustedContent
                    ? p.neutralizeControlTokens(in: text, preserving: trustedThink)
                    : text
                if !assistantText.hasPrefix(p.thinkOpen),
                   !assistantText.hasPrefix(p.thinkClose) {
                    output += p.thinkOpen + p.thinkClose
                }
                output += assistantText
                if !calls.isEmpty {
                    output += "\n\n" + (try GLM52ToolCodec.renderToolCalls(calls, tools: tools))
                }
                pendingAssistant = false
                previousWasToolResult = false
            }
        }

        if addGenerationPrompt && pendingAssistant {
            output += p.assistant
            output += reasoning.enabled
                ? p.thinkOpen
                : p.thinkOpen + p.thinkClose
        }
        return output
    }

    /// Convenience for the common one-user-turn demo path.
    public static func renderPrompt(
        system: String? = nil,
        prompt: String,
        tools: [ToolSpec] = [],
        reasoning: ThinkMode = .none,
        neutralizeUntrustedContent: Bool = true
    ) throws -> String {
        var turns: [ChatTurn] = []
        if let system, !system.isEmpty { turns.append(.system(system)) }
        turns.append(.user(prompt))
        return try render(
            turns: turns,
            tools: tools,
            reasoning: reasoning,
            neutralizeUntrustedContent: neutralizeUntrustedContent
        )
    }
}
