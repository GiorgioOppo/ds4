import SwiftUI
import AppKit
import UniformTypeIdentifiers
import DS4Engine
import DS4Core

extension ChatStore {
    /// Start a new asynchronous owner for the active conversation. Cancelling the
    /// old task is not sufficient on its own: a stream or tool await may still
    /// resume once, so every continuation also compares this epoch before touching
    /// state or executing another tool.
    @discardableResult
    func beginConversationWork() -> UInt64 {
        generation?.cancel()
        generation = nil
        conversationEpoch &+= 1
        return conversationEpoch
    }

    /// Invalidate every stream/tool continuation that captured the previous epoch.
    /// Used by Stop and by conversation identity changes.
    @discardableResult
    func invalidateConversationWork() -> UInt64 {
        generation?.cancel()
        generation = nil
        conversationEpoch &+= 1
        return conversationEpoch
    }

    /// `Task.isCancelled` closes the small window before an epoch-changing UI
    /// action runs; the epoch closes the larger window after an awaited operation
    /// resumes despite cancellation.
    func ownsConversationWork(_ epoch: UInt64) -> Bool {
        !Task.isCancelled && conversationEpoch == epoch
    }

    var sampling: SamplingParams {
        // topK 40 (default llama.cpp): con topK=0 si campiona sull'INTERO
        // vocabolario DeepSeek (129k token, in gran parte cinesi) e sulla coda
        // rumorosa di un modello 2-bit basta pescare UN token cinese perché il
        // contesto trascini tutta la risposta in cinese — visto in campo a
        // temperature del tutto normali (0.6). Il tetto a 40 taglia quella
        // coda senza togliere varietà; motore/server/demo restano fedeli al C.
        SamplingParams(temperature: Float(temperature), topK: 40,
                       repetitionPenalty: Float(repetitionPenalty))
    }

    /// Send the current input (+ any imported text files) and stream the reply,
    /// running the tool loop. Attachments are folded into the turn sent to the
    /// model; the transcript shows just the typed text and the filenames.
    func send() {
        let typed = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard EngineActivityGate.shared.activeOwner == nil,
              service != nil || glmService != nil, !isGenerating,
              !(typed.isEmpty && attachments.isEmpty) else { return }
        let service = self.service
        let glm = self.glmService
        let epoch = beginConversationWork()
        let atts = attachments
        let text = Self.composeUserText(typed: typed, attachments: atts)
        input = ""
        attachments = []
        attachmentNote = nil
        // If the engine doesn't hold this (reopened) chat yet, re-feed the prior
        // turns on this first send. Capture them BEFORE appending the new rows.
        let primed = enginePrimed
        let history = primed ? [] : Self.chatTurns(from: messages)
        let sys = primed ? nil : (resolvedAgent().systemPrompt.isEmpty ? nil : resolvedAgent().systemPrompt)
        enginePrimed = true
        messages.append(UIMessage(role: .user, text: typed, attachments: atts.map(\.name)))
        let index = appendAssistant()
        isGenerating = true
        toolRounds = 0                     // fresh user turn resets the tool-loop guard
        previousToolRoundFingerprints = []
        persistActiveSession()             // checkpoint the user turn right away

        let mode = thinkMode
        let params = sampling             // capture: `self` is weak inside the Task
        // .userInitiated: the decode runs on the InferenceService actor's executor at
        // THIS task's QoS. A default-priority task lets macOS deprioritize the work —
        // and, worse for the SSD-streaming path, throttle its expert-gather reads — so
        // the app decodes slower than the foreground CLI demo. Match the model-load
        // task's priority (and the CLI's foreground QoS) explicitly.
        generation = Task(priority: .userInitiated) { [weak self] in
            let stream: AsyncThrowingStream<GenEvent, Error>
            if let glm {
                stream = primed
                    ? await glm.send(userText: text, thinkMode: mode,
                                     sampling: params, maxTokens: 4096)
                    : await glm.sendWithHistory(
                        history, userText: text, systemPrompt: sys,
                        thinkMode: mode, sampling: params, maxTokens: 4096)
            } else if let service {
                stream = primed
                    ? await service.send(userText: text, thinkMode: mode, sampling: params, maxTokens: 4096)
                    : await service.sendWithHistory(history, userText: text, systemPrompt: sys,
                                                    thinkMode: mode, sampling: params, maxTokens: 4096)
            } else { return }
            guard let self, self.ownsConversationWork(epoch) else { return }
            await self.consume(stream, into: index, epoch: epoch)
            guard self.ownsConversationWork(epoch) else { return }
            let continued = await self.handleToolCalls(assistantIndex: index, epoch: epoch)
            if !continued { self.finishIfIdle(epoch: epoch) }
        }
    }

    /// Submit manually-entered results for the pending (non-built-in) tool calls.
    func submitManualResults(_ contents: [String: String]) {
        guard service != nil || glmService != nil, !pendingManualCalls.isEmpty,
              let epoch = pendingManualEpoch,
              conversationEpoch == epoch,
              pendingManualCalls.count == pendingManualOutputIndices.count else { return }
        var slots = pendingOrderedToolOutputs
        for (c, slot) in zip(pendingManualCalls, pendingManualOutputIndices) {
            guard slots.indices.contains(slot) else { return }
            let content = contents[c.id] ?? ""
            slots[slot] = ToolOutput(callId: c.id, name: c.name, content: content)
            messages.append(UIMessage(role: .tool, text: "\(c.name) → \(content)"))
        }
        // Every call (automatic, denied, duplicate, over-budget, sub-agent, or
        // manual) owns exactly one positional slot. Refuse a structurally partial
        // batch instead of silently shifting later tool_result associations.
        guard slots.allSatisfy({ $0 != nil }) else { return }
        let outputs = slots.compactMap { $0 }
        pendingManualCalls = []
        pendingOrderedToolOutputs = []
        pendingManualOutputIndices = []
        pendingManualEpoch = nil
        awaitingManualResults = false
        if let service {
            continueWithToolOutputs(outputs, service: service, epoch: epoch)
        } else if let glm = glmService {
            continueWithGLMToolOutputs(outputs, glm: glm, epoch: epoch)
        }
    }

    /// Abandon the pending manual tool calls without answering them. The calls
    /// stay in the committed KV (the model emitted them), so the next user turn
    /// follows an unanswered call — the model generally copes, but we surface the
    /// abandonment in the transcript so the state is visible.
    func cancelManualResults() {
        if !pendingManualCalls.isEmpty {
            let names = pendingManualCalls.map(\.name).joined(separator: ", ")
            messages.append(UIMessage(role: .tool, text: "✗ no results provided for: \(names)"))
        }
        pendingManualCalls = []
        pendingOrderedToolOutputs = []
        pendingManualOutputIndices = []
        pendingManualEpoch = nil
        awaitingManualResults = false
        let epoch = invalidateConversationWork()
        finishIfIdle(epoch: epoch)
    }

    func stop() {
        let epoch = invalidateConversationWork()
        if !pendingManualCalls.isEmpty {
            let names = pendingManualCalls.map(\.name).joined(separator: ", ")
            messages.append(UIMessage(role: .tool, text: "✗ stopped before results were provided for: \(names)"))
        }
        pendingManualCalls = []
        pendingOrderedToolOutputs = []
        pendingManualOutputIndices = []
        pendingManualEpoch = nil
        awaitingManualResults = false
        // The stale task is no longer allowed to perform its old cleanup, so Stop
        // finalizes the visible state itself. In-flight sub-agent cards remain as
        // trace evidence but are no longer shown as running.
        for i in messages.indices where messages[i].subAgentRunning {
            messages[i].subAgentRunning = false
            if let run = messages[i].subAgent, run.answer.isEmpty {
                messages[i].subAgent = InferenceService.SubAgentRun(
                    target: run.target, question: run.question,
                    answer: "(sub-agent stopped)", steps: run.steps)
            }
        }
        isGenerating = false
        status = ""
        finishIfIdle(epoch: epoch)
    }
}
