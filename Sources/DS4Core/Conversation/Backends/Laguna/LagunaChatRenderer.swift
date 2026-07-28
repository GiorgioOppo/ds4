import Foundation

/// One Laguna chat message as the reference server models it (`chat_msg` in
/// upstream `ds4_server.c`): the assistant role carries its reasoning as a
/// separate field — never embedded in `content` — plus structured tool calls
/// and, when the calls were parsed from generated text, the raw tool block
/// that is replayed verbatim on re-render.
public struct LagunaChatMessage: Sendable, Equatable {
    public enum Role: Sendable, Equatable {
        case system
        case user
        case tool
        case assistant
    }

    public var role: Role
    public var content: String
    /// Assistant-only.  In thinking mode the renderer always emits
    /// `<think>reasoning</think>` (empty reasoning included); with thinking
    /// disabled the stored reasoning is dropped, exactly like upstream.
    public var reasoning: String?
    public var toolCalls: [ToolCall]
    /// Assistant-only.  When present and non-empty it is emitted verbatim in
    /// place of re-rendering `toolCalls`, mirroring `raw_tool_text` upstream
    /// so a parse→render round trip is byte-stable.
    public var rawToolText: String?

    public init(role: Role, content: String, reasoning: String? = nil,
                toolCalls: [ToolCall] = [], rawToolText: String? = nil) {
        self.role = role
        self.content = content
        self.reasoning = reasoning
        self.toolCalls = toolCalls
        self.rawToolText = rawToolText
    }

    /// Map a backend-neutral turn onto the reference message shape.  The
    /// shared `ChatTurn` has no reasoning field, so an embedded
    /// `<think>…</think>` prefix in assistant text is split into the separate
    /// reasoning field (the inverse of `split_reasoning_content` upstream).
    public static func from(_ turn: ChatTurn) -> LagunaChatMessage {
        switch turn {
        case .system(let content):
            return LagunaChatMessage(role: .system, content: content)
        case .user(let content):
            return LagunaChatMessage(role: .user, content: content)
        case .toolResult(_, _, let content):
            return LagunaChatMessage(role: .tool, content: content)
        case .assistant(let text, let calls):
            let p = LagunaConversationProtocol.self
            // `split_reasoning_content` upstream: strip one leading <think>,
            // split at the first </think>; with no close tag the text stays
            // whole (prefix included) and there is no reasoning.
            var body = Substring(text)
            if body.hasPrefix(p.thinkOpen) { body = body.dropFirst(p.thinkOpen.count) }
            var reasoning: String?
            var content = text
            if let close = body.range(of: p.thinkClose) {
                reasoning = String(body[..<close.lowerBound])
                content = String(body[close.upperBound...])
            }
            return LagunaChatMessage(role: .assistant, content: content,
                                     reasoning: reasoning, toolCalls: calls)
        }
    }
}

/// Pure Laguna S 2.1 chat-template renderer. It mirrors
/// `render_laguna_chat_prompt_text` in upstream `ds4_server.c` while keeping
/// tool syntax and untrusted-payload containment separate from the DeepSeek
/// and GLM renderers.
///
/// Framing summary: the transcript opens with the `〈|EOS|〉` sequence marker,
/// the first system message (or the Poolside default prompt) is hoisted into
/// one `<system>` block together with the optional `### Tools` section, and
/// each turn is a text-tagged block terminated by a newline.  Assistant turns
/// open with the dedicated `<assistant>` token, carry `<think>…</think>` in
/// thinking mode (a bare `</think>` with thinking disabled), then visible
/// content and native tool calls, and close with `</assistant>`, the family's
/// end-of-turn token.
public enum LagunaChatRenderer {
    /// Render a complete transcript from backend-neutral turns.
    public static func render(
        turns: [ChatTurn],
        tools: [ToolSpec] = [],
        reasoning: ThinkMode = .none,
        addGenerationPrompt: Bool = true,
        neutralizeUntrustedContent: Bool = true
    ) throws -> String {
        try render(
            messages: turns.map(LagunaChatMessage.from),
            tools: tools,
            reasoning: reasoning,
            addGenerationPrompt: addGenerationPrompt,
            neutralizeUntrustedContent: neutralizeUntrustedContent
        )
    }

    /// Render a complete transcript (`render_laguna_chat_prompt_text`).
    public static func render(
        messages: [LagunaChatMessage],
        tools: [ToolSpec] = [],
        reasoning: ThinkMode = .none,
        addGenerationPrompt: Bool = true,
        neutralizeUntrustedContent: Bool = true
    ) throws -> String {
        let p = LagunaConversationProtocol.self

        var system = p.defaultSystemPrompt
        var start = 0
        if let first = messages.first, first.role == .system {
            system = first.content
            start = 1
        }

        var output = p.bosMarker
        let toolsSection = try LagunaToolCodec.toolsSection(
            tools, neutralize: neutralizeUntrustedContent
        )
        let systemHasText = p.hasNonWhitespaceASCII(system)
        if systemHasText || !toolsSection.isEmpty || reasoning.enabled {
            output += p.systemOpen
            if systemHasText {
                let safeSystem = neutralizeUntrustedContent
                    ? p.neutralizeControlTokens(in: system)
                    : system
                output += String(p.trimTrailingASCIIWhitespace(safeSystem))
                if !toolsSection.isEmpty { output += "\n\n" }
            }
            output += toolsSection
            output += p.systemClose + "\n"
        }

        output += try renderBody(
            messages: messages[start...],
            tools: tools,
            reasoning: reasoning,
            addGenerationPrompt: addGenerationPrompt,
            neutralizeUntrustedContent: neutralizeUntrustedContent
        )
        return output
    }

    /// The incremental continuation appended to a live KV prefix after a tool
    /// round (`render_laguna_live_tool_tail`): the generated stop token is
    /// inspected but never accepted into the cache, so the tail re-closes the
    /// assistant turn before the new messages.
    public static func liveToolTail(
        messages: [LagunaChatMessage],
        tools: [ToolSpec] = [],
        reasoning: ThinkMode = .none,
        neutralizeUntrustedContent: Bool = true
    ) throws -> String {
        LagunaConversationProtocol.assistantClose + "\n" + (try renderBody(
            messages: messages[...],
            tools: tools,
            reasoning: reasoning,
            addGenerationPrompt: true,
            neutralizeUntrustedContent: neutralizeUntrustedContent
        ))
    }

    /// The recovery suffix the reference server feeds back when generated
    /// tool-call markup is malformed (`build_invalid_laguna_tool_error_suffix`):
    /// close the turn, surface a retryable tool error, reopen the assistant.
    public static func invalidToolCallRecoverySuffix(
        detail: String? = nil,
        reasoning: ThinkMode = .none,
        insideThinking: Bool = false
    ) -> String {
        let p = LagunaConversationProtocol.self
        var toolError = "Tool error: invalid Laguna tool call"
        if let detail, !detail.isEmpty { toolError += ": " + detail }
        toolError += "\nThe previous assistant output was not executed because "
            + "the <tool_call> syntax was malformed. Emit a new valid "
            + "<tool_call>, or answer normally if no tool is needed."

        var suffix = ""
        if reasoning.enabled && insideThinking { suffix += p.thinkClose }
        suffix += p.assistantClose + "\n" + p.toolResponseOpen
        suffix += toolError.replacingOccurrences(
            of: p.toolResponseClose,
            with: "&lt;" + p.toolResponseClose.dropFirst()
        )
        suffix += p.toolResponseClose + "\n" + p.assistantOpen
        suffix += reasoning.enabled ? p.thinkOpen : p.thinkClose
        return suffix
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

    private static func renderBody(
        messages: ArraySlice<LagunaChatMessage>,
        tools: [ToolSpec],
        reasoning: ThinkMode,
        addGenerationPrompt: Bool,
        neutralizeUntrustedContent: Bool
    ) throws -> String {
        let p = LagunaConversationProtocol.self
        let safe: (String) -> String = { text in
            neutralizeUntrustedContent
                ? p.neutralizeControlTokens(in: text)
                : text
        }

        var output = ""
        var pendingAssistant = false
        for message in messages {
            switch message.role {
            case .system:
                // Later system/developer messages render in input order; they
                // are not part of the hoisted prologue.
                output += p.systemOpen + safe(message.content) + p.systemClose + "\n"
            case .user:
                output += p.userOpen + safe(message.content) + p.userClose + "\n"
                pendingAssistant = true
            case .tool:
                // Unlike GLM there is no shared observation role marker: every
                // result is an independent text-tagged block.
                output += p.toolResponseOpen
                output += neutralizeUntrustedContent
                    ? LagunaToolCodec.escapeToolResponse(message.content)
                    : message.content.replacingOccurrences(
                        of: p.toolResponseClose,
                        with: "&lt;" + p.toolResponseClose.dropFirst()
                    )
                output += p.toolResponseClose + "\n"
                pendingAssistant = true
            case .assistant:
                output += p.assistantOpen
                let trustedThink = Set([p.thinkOpen, p.thinkClose])
                let content = neutralizeUntrustedContent
                    ? p.neutralizeControlTokens(in: message.content,
                                                preserving: trustedThink)
                    : message.content
                if reasoning.enabled {
                    // The reference server always frames historical reasoning
                    // in thinking mode, empty reasoning included; with
                    // thinking disabled the reasoning is dropped.
                    let thought = message.reasoning ?? ""
                    output += p.thinkOpen
                    output += neutralizeUntrustedContent
                        ? p.neutralizeControlTokens(in: thought,
                                                    preserving: trustedThink)
                        : thought
                    output += p.thinkClose
                } else {
                    output += p.thinkClose
                }
                output += content
                if let raw = message.rawToolText, !raw.isEmpty {
                    output += raw
                } else if !message.toolCalls.isEmpty {
                    output += try LagunaToolCodec.renderToolCalls(
                        message.toolCalls, tools: tools,
                        neutralize: neutralizeUntrustedContent
                    )
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
}
