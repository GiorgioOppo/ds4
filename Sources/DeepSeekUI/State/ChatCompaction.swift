import Foundation

/// Stable marker prepended to the synthetic system message that
/// `ChatStore.compactOlderTurns(of:keepLastN:)` writes in place of
/// the compacted prefix. Wrapped in a nonisolated enum so the
/// constant has no actor isolation to inherit — neither from a
/// surrounding `@MainActor` class nor from Swift 6's implicit
/// top-level-let inference — and stays readable from any
/// generic-predicate closure in `ChatView`'s compaction banner
/// or the future re-encoder that recognises summary messages.
enum ChatCompactionConstants {
    static let marker = "[compacted summary of older turns]"
}

/// Compaction reduces long chat history by replacing older turns
/// with a model-generated summary. Works against the same
/// `OpenRouterClient` / `AnthropicClient` that `runRemoteLoop`
/// already uses, so the user doesn't need to configure a
/// separate "summarizer" model — the chat's own endpoint
/// performs the work.
///
/// The summary message gets a `.system` role and is tagged with
/// `chatCompactionMarker` at the start of its content so the UI
/// can render it visually distinct and `applySlidingWindow`
/// doesn't accidentally drop it.
extension ChatStore {

    /// Back-compat alias for `ChatCompactionConstants.marker`.
    /// New code should reach the enum directly; this stays around
    /// so existing references via `ChatStore.compactionMarker`
    /// keep resolving. Marked `nonisolated` for the same reason
    /// as the enum's static: the value has no actor-dependent
    /// state, and callers reach it from generic-predicate closures
    /// that don't inherit the surrounding `@MainActor`.
    nonisolated static var compactionMarker: String {
        ChatCompactionConstants.marker
    }

    /// Replace user-led turns older than the last `keepLastN` with
    /// a single `.system` message containing a model-generated
    /// summary. The transcript shrinks both on screen and in the
    /// request body. Idempotent in the no-op direction: when
    /// there's nothing older than the cap, the chat is left
    /// untouched.
    ///
    /// Errors at any stage (no endpoint, no API key, summary
    /// stream failure) abort cleanly: the original messages stay
    /// in place and the user sees `.error(...)` in `phases`.
    func compactOlderTurns(of conversationID: UUID,
                            keepLastN: Int = 4) async {
        guard let idx = conversations.firstIndex(where: { $0.id == conversationID }),
              let endpoint = conversations[idx].endpoint
                ?? modelState.loadedEndpoint
        else { return }

        // Compute cutoff. The cap counts USER messages so we don't
        // accidentally cut between an assistant's tool-call and its
        // tool-output.
        let userIndices = conversations[idx].messages.enumerated()
            .filter { $0.element.role == .user }
            .map(\.offset)
        guard userIndices.count > keepLastN else { return }
        let cutoff = userIndices[userIndices.count - keepLastN]
        let toCompact = Array(conversations[idx].messages.prefix(cutoff))
        guard !toCompact.isEmpty else { return }

        setPhase(.streaming(buffer: "",
                              status: "Compacting older turns…",
                              metrics: GenerationMetrics()),
                  for: conversationID)

        let summary: String
        do {
            summary = try await runCompactionRequest(
                messages: toCompact, endpoint: endpoint)
        } catch {
            setPhase(.error(
                "Compaction failed: " +
                ((error as? LocalizedError)?.errorDescription
                 ?? error.localizedDescription)),
                      for: conversationID)
            return
        }
        let trimmed = summary.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            setPhase(.error(
                "Compaction returned an empty summary; the older "
                + "turns were left untouched."),
                      for: conversationID)
            return
        }

        let summaryMessage = StoredMessage(
            role: .system,
            content: ChatCompactionConstants.marker + "\n\n" + trimmed)
        // Replace the prefix [0..<cutoff] with the single summary
        // message. The rest of the transcript — including any
        // .system messages that landed inside the compacted span
        // — moves up by `cutoff - 1` positions; the agent prompt
        // (which lives on `AgentConfig`, not in messages) is
        // unaffected.
        conversations[idx].messages.replaceSubrange(
            0..<cutoff, with: [summaryMessage])
        // Invalidate the local fast-path token cache. The cached
        // prefix encoded the now-replaced messages; re-encoding
        // from scratch on the next turn is the correct fix.
        conversations[idx].encodedTokens = nil
        conversations[idx].lastEncodedMode = nil
        setPhase(.idle, for: conversationID)
        requestSave(conversationID)
    }

    /// Drive the summarisation request against whichever provider
    /// the chat is using. Streams the response and concatenates
    /// the deltas — same pattern as `runRemoteLoop` minus the
    /// tool-roundtrip loop.
    private func runCompactionRequest(messages: [StoredMessage],
                                        endpoint: ModelEndpoint) async throws
        -> String
    {
        let transcript = renderTranscriptForCompaction(messages)
        let summarizerSystem =
            "You are compacting a long chat history. Produce a precise summary "
            + "that preserves: (1) what the user asked, (2) any conclusions or "
            + "decisions reached, (3) tool calls and their notable outputs, "
            + "(4) the current state of the work in progress. The summary will "
            + "REPLACE the transcript below in the chat history, so the model "
            + "continuing the conversation will only see your summary. Keep it "
            + "under ~500 words. Do not invent details that aren't in the "
            + "transcript."
        let userPrompt =
            "Summarize the following transcript according to the system "
            + "instructions. Output ONLY the summary text — no preface, no "
            + "trailing notes.\n\n---\n\(transcript)\n---"

        switch endpoint {
        case .openRouter(let modelID):
            return try await streamCompactionOpenRouter(
                modelID: modelID,
                system: summarizerSystem,
                userPrompt: userPrompt)
        case .anthropic(let modelID):
            return try await streamCompactionAnthropic(
                modelID: modelID,
                system: summarizerSystem,
                userPrompt: userPrompt)
        case .localDirectory:
            throw NSError(
                domain: "ChatCompaction", code: 1,
                userInfo: [NSLocalizedDescriptionKey:
                    "Compaction isn't wired for local endpoints yet — the "
                    + "feature is currently remote-only."])
        }
    }

    private func renderTranscriptForCompaction(_ messages: [StoredMessage])
        -> String
    {
        var lines: [String] = []
        for msg in messages {
            let role: String = {
                switch msg.role {
                case .user: return "USER"
                case .assistant: return "ASSISTANT"
                case .system: return "SYSTEM"
                }
            }()
            if !msg.content.isEmpty {
                lines.append("[\(role)] \(msg.content)")
            }
            if !msg.toolCalls.isEmpty {
                for tc in msg.toolCalls {
                    lines.append("[\(role) tool_call: \(tc.name)] \(tc.args)")
                }
            }
            if let outputs = msg.toolOutputs, !outputs.isEmpty {
                for (i, o) in outputs.enumerated() {
                    lines.append("[\(role) tool_output #\(i)] \(o)")
                }
            }
        }
        return lines.joined(separator: "\n")
    }

    private func streamCompactionOpenRouter(modelID: String,
                                              system: String,
                                              userPrompt: String) async throws
        -> String
    {
        let apiKey = KeychainStore.get(
            account: KeychainAccount.openRouterAPIKey) ?? ""
        guard !apiKey.isEmpty else {
            throw NSError(
                domain: "ChatCompaction", code: 2,
                userInfo: [NSLocalizedDescriptionKey:
                    "OpenRouter API key missing."])
        }
        let body: [String: Any] = [
            "model": modelID,
            "messages": [
                ["role": "system", "content": system],
                ["role": "user", "content": userPrompt],
            ],
            "temperature": 0.3,
            "max_tokens": 1024,
        ]
        let client = OpenRouterClient()
        var summary = ""
        for try await chunk in client.streamChatCompletion(
            apiKey: apiKey, body: body)
        {
            if let piece = chunk.choices.first?.delta?.content {
                summary += piece
            }
        }
        return summary
    }

    private func streamCompactionAnthropic(modelID: String,
                                             system: String,
                                             userPrompt: String) async throws
        -> String
    {
        let apiKey = KeychainStore.get(
            account: KeychainAccount.anthropicAPIKey) ?? ""
        guard !apiKey.isEmpty else {
            throw NSError(
                domain: "ChatCompaction", code: 3,
                userInfo: [NSLocalizedDescriptionKey:
                    "Anthropic API key missing."])
        }
        let userMsg = StoredMessage(role: .user, content: userPrompt)
        // Explicit `Float()` wrappers: `AnthropicMessageBuilder.buildBody`
        // declares `temperature: Float? = nil`, and Swift 6's stricter
        // literal-inference path won't silently promote a `Double`
        // literal across the `Float?` boundary. Drop `topP` so the
        // builder uses its nil-default — Anthropic treats it as
        // "disabled" anyway, identical to passing 1.0.
        let body = AnthropicMessageBuilder.buildBody(
            model: modelID,
            maxTokens: 1024,
            history: [userMsg],
            agentSystem: system,
            tools: nil,
            temperature: Float(0.3))
        let client = AnthropicClient()
        var summary = ""
        for try await chunk in client.streamMessages(
            apiKey: apiKey, body: body)
        {
            if let piece = chunk.choices.first?.delta?.content {
                summary += piece
            }
        }
        return summary
    }
}
