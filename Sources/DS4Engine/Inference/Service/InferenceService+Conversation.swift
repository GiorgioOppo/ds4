import Foundation
import DS4Core
import DS4Metal

extension InferenceService {
public func resetDecodeProfile() { decoder.resetProfile() }
    public func decodeProfileReport() -> String { decoder.profile.report() }
    /// Per-phase profile of the LAST prefill: captured at the prefill→decode
    /// boundary (where the counters are reset for the decode profile).
    public func prefillProfileReport() -> String { lastPrefillProfile }

    public func resetConversation(systemPrompt: String?) {
        self.systemPrompt = (systemPrompt?.isEmpty == false) ? systemPrompt : nil
        committedIds = []
        needsClose = false
        kvDirty = false   // next generation starts at pos 0 and resets the compressor
        lastDiskStoreCount = 0
    }

    /// Enable/disable the disk KV cache. `dir` nil turns it off. Takes effect on
    /// the next generation; existing checkpoints in `dir` become restorable.
    /// The budget is in TOKENS (total across stored checkpoints — the live
    /// context window stays `contextSize`); bytes follow the model's per-token
    /// checkpoint size, so tokens are the stable unit to configure.
    public func setDiskKV(directory: URL?, budgetTokens: Int) {
        guard let directory else { diskKV = nil; return }
        let bits: UInt8 = dims.gateQuant == .iq2_xxs ? 2 : 4
        diskKV = try? DiskKVStore(directory: directory, budgetMB: 0, quantBits: bits,
                                  contextSize: contextSize, budgetTokens: max(1, budgetTokens))
    }

    /// Declare the tools available to the model. Tools are baked into the first
    /// prompt of a conversation, so a change takes effect on the next new chat.
    public func setTools(_ tools: [ToolSpec]) { self.tools = tools }

    /// Use the compact (name-list) tool declaration to save prefill tokens.
    public func setCompactTools(_ on: Bool) { compactTools = on }

    func assistantOpen(_ think: DS4ThinkMode) -> String {
        "<｜Assistant｜>" + (think == .high ? "<think>" : "</think>")
    }

    /// Keep untrusted payload text from becoming model control tokens when the
    /// completed transcript is passed to `tokenizeRenderedChat`. Trusted framing
    /// is added by `ChatRenderer` only after these field-level transformations.
    func promptSafeText(_ text: String) -> String {
        tok.neutralizeSpecialTokenLiterals(in: text)
    }

    func promptSafeTools(_ specs: [ToolSpec]) -> [ToolSpec] {
        specs.map {
            ToolSpec(name: promptSafeText($0.name),
                     description: promptSafeText($0.description),
                     parametersJSON: tok.neutralizeSpecialTokenLiterals(inJSON: $0.parametersJSON))
        }
    }

    func promptSafeTurns(_ turns: [ChatTurn]) -> [ChatTurn] {
        turns.map { turn in
            switch turn {
            case .system(let text):
                return .system(promptSafeText(text))
            case .user(let text):
                return .user(promptSafeText(text))
            case .assistant(let text, let calls):
                let safeCalls = calls.map {
                    ToolCall(id: $0.id, name: promptSafeText($0.name),
                             argumentsJSON: tok.neutralizeSpecialTokenLiterals(inJSON: $0.argumentsJSON))
                }
                return .assistant(text: promptSafeText(text), toolCalls: safeCalls)
            case .toolResult(let callId, let name, let content):
                return .toolResult(callId: callId, name: promptSafeText(name),
                                   content: promptSafeText(content))
            }
        }
    }

    /// The prefix that opens a new user/tool turn: BOS + system (+ tools) the first
    /// time, otherwise the <eos> that closes the previous (still-open) assistant turn.
    func openingPrefix() -> String {
        if committedIds.isEmpty {
            let systemTurns = promptSafeTurns(systemPrompt.map { [.system($0)] } ?? [])
            let sys = ChatRenderer.systemBlock(turns: systemTurns,
                                               tools: promptSafeTools(tools),
                                               markup: markup, compact: compactTools)
            return "<｜begin▁of▁sentence｜>" + sys
        }
        return needsClose ? "<｜end▁of▁sentence｜>" : ""
    }

    /// Append a user message and generate the assistant reply (prefills only the
    /// new suffix, reusing the KV cache of the prior turns).
    public func send(userText: String, thinkMode: DS4ThinkMode, sampling: SamplingParams,
                     maxTokens: Int) -> AsyncThrowingStream<GenEvent, Error> {
        let suffix = openingPrefix() + "<｜User｜>" + promptSafeText(userText) + assistantOpen(thinkMode)
        return run(suffix: suffix, think: thinkMode, sampling: sampling, maxTokens: maxTokens)
    }

    /// Re-prime a REOPENED conversation whose KV the engine no longer holds (the
    /// GUI restored a persisted chat), then generate the reply. The prior `history`
    /// turns + the new user turn are rendered as one prompt: on a cold KV the disk
    /// cache restores the longest matching prefix (so this is NOT a full re-prefill
    /// when the chat was checkpointed), and only the remainder is prefilled. After
    /// this call the KV holds the whole conversation, so the next turns reuse it
    /// incrementally via `send`. `tools` (set by the agent) are preserved.
    public func sendWithHistory(_ history: [ChatTurn], userText: String, systemPrompt: String?,
                                thinkMode: DS4ThinkMode, sampling: SamplingParams,
                                maxTokens: Int) -> AsyncThrowingStream<GenEvent, Error> {
        resetConversation(systemPrompt: systemPrompt)
        var turns: [ChatTurn] = self.systemPrompt.map { [.system($0)] } ?? []
        turns.append(contentsOf: history)
        turns.append(.user(userText))
        let suffix = ChatRenderer.render(turns: promptSafeTurns(turns),
                                         tools: promptSafeTools(tools), think: thinkMode.core,
                                         markup: markup, compactTools: compactTools)
        return run(suffix: suffix, think: thinkMode, sampling: sampling, maxTokens: maxTokens)
    }

    /// Append tool results (inside a user turn) and continue the assistant turn.
    public func provideToolResults(_ outputs: [ToolOutput], thinkMode: DS4ThinkMode,
                                   sampling: SamplingParams, maxTokens: Int) -> AsyncThrowingStream<GenEvent, Error> {
        var suffix = openingPrefix() + "<｜User｜>"
        for o in outputs {
            let content = promptSafeText(ChatRenderer.escapeToolResult(o.content))
            suffix += "<tool_result>" + content + "</tool_result>"
        }
        suffix += assistantOpen(thinkMode)
        return run(suffix: suffix, think: thinkMode, sampling: sampling, maxTokens: maxTokens)
    }

    /// Stateless completion for the local HTTP server: reset the KV and render the
    /// FULL message list as a fresh prompt, then generate. Mirrors OpenAI semantics
    /// where each request carries the whole conversation (no server-side history).
    public func complete(turns: [ChatTurn], tools: [ToolSpec], thinkMode: DS4ThinkMode,
                         sampling: SamplingParams, maxTokens: Int) -> AsyncThrowingStream<GenEvent, Error> {
        self.tools = tools
        // Render + tokenizzazione dell'INTERO transcript: è lavoro sincrono
        // sull'actor PRIMA che parta il prefill — se è lento è "tempo morto"
        // invisibile per il client, quindi lo misuriamo e lo denunciamo.
        let tPrep = Date()
        let safeTurns = promptSafeTurns(turns)
        let safeTools = promptSafeTools(tools)
        let suffix = ChatRenderer.render(turns: safeTurns, tools: safeTools, think: thinkMode.core,
                                         markup: markup, compactTools: compactTools)
        let ids = tok.tokenizeRenderedChat(suffix).map { Int($0) }
        let prepS = Date().timeIntervalSince(tPrep)
        if prepS > 0.25 {
            FileHandle.standardError.write(Data(String(
                format: "DS4 server: render+tokenizzazione %d char → %d token in %.2fs\n",
                suffix.count, ids.count, prepS).utf8))
        }
        // CONTINUITÀ KV per il server STATELESS (Xcode & co. ri-inviano
        // l'intero transcript a ogni richiesta, system prompt di migliaia di
        // token incluso): se i token ESTENDONO esattamente quelli già
        // committati, prefilla SOLO il suffisso — da minuti a secondi dal
        // secondo scambio in poi. Qualunque mismatch (chat GUI interlacciata
        // sullo stesso motore, transcript diverso, stato sporco) ricade nel
        // reset di sempre: prefill freddo, correttezza invariata.
        if !kvDirty, !committedIds.isEmpty, ids.count > committedIds.count,
           ids.starts(with: committedIds) {
            FileHandle.standardError.write(Data(
                "DS4 server: KV riusato in memoria (\(committedIds.count) token già caldi)\n".utf8))
            return run(suffixIds: Array(ids.dropFirst(committedIds.count)),
                       think: thinkMode, sampling: sampling, maxTokens: maxTokens,
                       resumablePrefill: true)
        }
        resetConversation(systemPrompt: nil)
        self.tools = tools
        // Confine del PREFISSO CONDIVISO (BOS + system prompt + tool): è la
        // parte identica tra conversazioni diverse dello stesso client (ogni
        // chat Xcode ripete lo stesso system prompt). Se è grande e non è già
        // su disco, `generate` lo checkpointa APPENA prefillato: uno stream
        // annullato o una chat nuova lo RESTAURANO in secondi invece di
        // ripagare minuti di prefill. systemBlock è lo stesso pezzo che apre
        // il render completo, quindi il confine è un prefisso per costruzione.
        var checkpointAfter = 0
        if let store = diskKV {
            let sysPrefix = "<｜begin▁of▁sentence｜>" + ChatRenderer.systemBlock(
                turns: safeTurns, tools: safeTools, markup: markup, compact: compactTools)
            let sysCount = tok.tokenizeRenderedChat(sysPrefix).count
            if sysCount >= store.options.minTokens, ids.count > sysCount,
               (store.storedPrefixLengths(of: ids, modelName: modelName).first ?? 0) < sysCount {
                checkpointAfter = sysCount
            }
        }
        return run(suffixIds: ids, think: thinkMode, sampling: sampling, maxTokens: maxTokens,
                   checkpointAfter: checkpointAfter, resumablePrefill: true)
    }
}
