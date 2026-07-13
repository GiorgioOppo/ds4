import SwiftUI
import AppKit
import UniformTypeIdentifiers
import DS4Engine
import DS4Core

extension ChatStore {
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
        guard let service, !isGenerating, !(typed.isEmpty && attachments.isEmpty) else { return }
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
        persistActiveSession()             // checkpoint the user turn right away

        let mode = thinkMode
        let params = sampling             // capture: `self` is weak inside the Task
        // .userInitiated: the decode runs on the InferenceService actor's executor at
        // THIS task's QoS. A default-priority task lets macOS deprioritize the work —
        // and, worse for the SSD-streaming path, throttle its expert-gather reads — so
        // the app decodes slower than the foreground CLI demo. Match the model-load
        // task's priority (and the CLI's foreground QoS) explicitly.
        generation = Task(priority: .userInitiated) { [weak self] in
            let stream = primed
                ? await service.send(userText: text, thinkMode: mode, sampling: params, maxTokens: 4096)
                : await service.sendWithHistory(history, userText: text, systemPrompt: sys,
                                                thinkMode: mode, sampling: params, maxTokens: 4096)
            await self?.consume(stream, into: index)
            let continued = await self?.handleToolCalls(assistantIndex: index) ?? false
            if !continued { await MainActor.run { self?.finishIfIdle() } }
        }
    }

    /// Submit manually-entered results for the pending (non-built-in) tool calls.
    func submitManualResults(_ contents: [String: String]) {
        guard let service, !pendingManualCalls.isEmpty else { return }
        var outputs = partialAutoOutputs
        for c in pendingManualCalls {
            let content = contents[c.id] ?? ""
            outputs.append(ToolOutput(callId: c.id, name: c.name, content: content))
            messages.append(UIMessage(role: .tool, text: "\(c.name) → \(content)"))
        }
        pendingManualCalls = []
        partialAutoOutputs = []
        awaitingManualResults = false
        continueWithToolOutputs(outputs, service: service)
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
        partialAutoOutputs = []
        awaitingManualResults = false
        finishIfIdle()
    }

    func stop() { generation?.cancel() }
}
