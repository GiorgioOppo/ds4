import Foundation

/// Pure Laguna S 2.1 chat-template renderer. It mirrors
/// `render_laguna_chat_prompt_text` in upstream `ds4_server.c` while keeping
/// tool syntax and untrusted-payload containment separate from the DeepSeek
/// and GLM renderers.
///
/// Framing summary: the transcript opens with the `〈|EOS|〉` sequence marker,
/// the first system message (or the Poolside default prompt) is hoisted into
/// one `<system>` block together with the optional `### Tools` section, and
/// each turn is a text-tagged block terminated by a newline.  Assistant turns
/// open with the dedicated `<assistant>` token, carry `<think>…</think>` (or a
/// bare `</think>` when reasoning is absent), then visible content and native
/// tool calls, and close with `</assistant>`, the family's end-of-turn token.
public enum LagunaChatRenderer {
    /// Render a complete transcript.
    public static func render(
        turns: [ChatTurn],
        tools: [ToolSpec] = [],
        reasoning: ThinkMode = .none,
        addGenerationPrompt: Bool = true,
        neutralizeUntrustedContent: Bool = true
    ) throws -> String {
        let p = LagunaConversationProtocol.self
        let safe: (String) -> String = { text in
            neutralizeUntrustedContent
                ? p.neutralizeControlTokens(in: text)
                : text
        }

        var system = p.defaultSystemPrompt
        var start = 0
        if case .system(let content)? = turns.first {
            system = content
            start = 1
        }

        var output = p.bosMarker
        let toolsSection = try LagunaToolCodec.toolsSection(tools)
        let systemHasText = hasNonWhitespace(system)
        if systemHasText || !toolsSection.isEmpty || reasoning.enabled {
            output += p.systemOpen
            if systemHasText {
                output += trimTrailingWhitespace(safe(system))
                if !toolsSection.isEmpty { output += "\n\n" }
            }
            output += toolsSection
            output += p.systemClose + "\n"
        }

        var pendingAssistant = false
        for turn in turns[start...] {
            switch turn {
            case .system(let content):
                // Later system/developer messages render in input order; they
                // are not part of the hoisted prologue.
                output += p.systemOpen + safe(content) + p.systemClose + "\n"
            case .user(let content):
                output += p.userOpen + safe(content) + p.userClose + "\n"
                pendingAssistant = true
            case .toolResult(_, _, let content):
                // Unlike GLM there is no shared observation role marker: every
                // result is an independent text-tagged block.
                output += p.toolResponseOpen
                output += neutralizeUntrustedContent
                    ? LagunaToolCodec.escapeToolResponse(content)
                    : content.replacingOccurrences(
                        of: p.toolResponseClose,
                        with: "&lt;" + p.toolResponseClose.dropFirst()
                    )
                output += p.toolResponseClose + "\n"
                pendingAssistant = true
            case .assistant(let text, let calls):
                output += p.assistantOpen
                let trustedThink = Set([p.thinkOpen, p.thinkClose])
                let assistantText = neutralizeUntrustedContent
                    ? p.neutralizeControlTokens(in: text, preserving: trustedThink)
                    : text
                if !assistantText.hasPrefix(p.thinkOpen),
                   !assistantText.hasPrefix(p.thinkClose) {
                    output += p.thinkClose
                }
                output += assistantText
                if !calls.isEmpty {
                    output += try LagunaToolCodec.renderToolCalls(calls, tools: tools)
                }
                output += p.assistantClose + "\n"
                pendingAssistant = false
            }
        }

        if addGenerationPrompt && pendingAssistant {
            output += p.assistantOpen
            output += reasoning.enabled ? p.thinkOpen : p.thinkClose
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

    private static func hasNonWhitespace(_ text: String) -> Bool {
        text.contains { !$0.isWhitespace }
    }

    private static func trimTrailingWhitespace(_ text: String) -> String {
        var view = Substring(text)
        while let last = view.last, last.isWhitespace { view.removeLast() }
        return String(view)
    }
}
