import SwiftUI
import AppKit
import UniformTypeIdentifiers
import DS4Engine
import DS4Core

extension ChatStore {
    // MARK: - Internals

    /// Parse the (target, question, agent, tools) arguments of a `subagent_run`
    /// call. `tools` accepts a JSON array or a comma/space-separated string (some
    /// models quote list arguments).
    private static func subAgentArgs(_ json: String) -> (target: String, question: String, agent: String, tools: [String]) {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return ("", "", "", []) }
        var tools: [String] = []
        if let arr = obj["tools"] as? [Any] { tools = arr.compactMap { $0 as? String } }
        else if let s = obj["tools"] as? String {
            tools = s.split(whereSeparator: { $0 == "," || $0 == " " }).map(String.init)
        }
        return ((obj["target"] as? String) ?? "", (obj["question"] as? String) ?? "",
                (obj["agent"] as? String) ?? "", tools)
    }

    /// Validate a `subagent_run` call before executing it: nil when well-formed,
    /// otherwise an explanatory error the model can act on (fix and retry).
    /// Silent fallbacks here (empty question, ignored unknown role…) would waste
    /// a whole sub-agent run and leave the user staring at a garbage answer.
    private static func subAgentCallProblem(_ argumentsJSON: String,
                                            question: String, agent: String, tools: [String],
                                            allowedDelegatedTools: Set<String>) -> String? {
        guard let data = argumentsJSON.data(using: .utf8),
              (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] != nil else {
            return #"the arguments are not a JSON object; expected {"target":"<file path or project>","question":"<self-contained task>"}"#
        }
        if question.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "missing 'question': pass a self-contained task (the sub-agent does not see this chat)"
        }
        let agentId = agent.trimmingCharacters(in: .whitespacesAndNewlines)
        if !agentId.isEmpty, !AgentRegistry.shared.all().contains(where: { $0.id == agentId }) {
            let ids = AgentRegistry.shared.all().map(\.id).joined(separator: ", ")
            return "unknown agent '\(agentId)'; available agent ids: \(ids) (see agents_list)"
        }
        // MCP tools run app-side against external servers; the sub-agent loop is
        // engine-side and would silently drop them — fail loudly instead so the
        // model retries with built-ins (or the user learns why it can't work).
        let mcpRequested = tools.filter { MCPManager.shared.isMCPTool(named: $0) }
        if !mcpRequested.isEmpty {
            return "MCP tools cannot be granted to sub-agents: \(mcpRequested.joined(separator: ", ")); retry with built-in tools only"
        }
        let denied = tools.filter { !allowedDelegatedTools.contains($0) }
        if !denied.isEmpty {
            let allowed = allowedDelegatedTools.sorted().joined(separator: ", ")
            return "requested tools exceed the parent agent's delegation scope: \(denied.joined(separator: ", ")); allowed: \(allowed.isEmpty ? "(none)" : allowed)"
        }
        return nil
    }

    func appendAssistant() -> Int {
        messages.append(UIMessage(role: .assistant, text: ""))
        return messages.count - 1
    }

    func finishIfIdle(epoch: UInt64) {
        guard conversationEpoch == epoch else { return }
        if pendingManualCalls.isEmpty {
            generation = nil
            isGenerating = false
            status = ""
            refreshContextUsage(epoch: epoch)
            persistActiveSession()        // checkpoint the completed turn
            // Il Profilo decode va nel Log motore DOPO OGNI risposta: i contatori
            // sono raccolti comunque, il report costa nulla, e "a quanto genera
            // davvero l'app e dove va il tempo" deve essere leggibile dal log
            // senza attivare niente. (profileRouteEnabled resta il gate della
            // sola scomposizione route/attn, che aggiunge sync GPU.)
            emitDecodeProfile(epoch: epoch)
        }
    }

    /// Print the last turn's prefill + decode profiles to stderr so they land in
    /// the Log motore (EngineLog captures fd 2), mirroring the demo's DIAG output.
    private func emitDecodeProfile(epoch: UInt64) {
        guard let service else { return }
        Task { [weak self] in
            let prefill = await service.prefillProfileReport()
            let report = await service.decodeProfileReport()
            guard let self, self.conversationEpoch == epoch else { return }
            FileHandle.standardError.write(Data(("\n" + prefill + "\n\n" + report + "\n").utf8))
        }
    }

    /// Refresh the committed-token count (context usage) from the engine.
    private func refreshContextUsage(epoch: UInt64) {
        guard let service else { contextUsed = 0; return }
        Task { [weak self] in
            let tokens = await service.committedTokens()
            guard let self, self.conversationEpoch == epoch else { return }
            self.contextUsed = tokens
        }
    }

    /// Drain one generation stream into the assistant message at `index`.
    func consume(_ stream: AsyncThrowingStream<GenEvent, Error>, into index: Int,
                 epoch: UInt64) async {
        do {
            for try await event in stream {
                guard ownsConversationWork(epoch), index < messages.count else { return }
                switch event {
                case .reasoning(let r): messages[index].reasoning += r
                case .text(let t): messages[index].text += t
                case .toolStream(let s): messages[index].toolStreamText += s
                case .toolCall(let calls):
                    messages[index].toolCalls = calls
                    // The block closed: drop the raw live markup; the formatted card
                    // (ToolCallView) takes over.
                    messages[index].toolStreamText = ""
                    // When the model spelled the DSML markup out as plain text (it
                    // streamed into the bubble), strip the parsed block + any leaked
                    // malformed markup from view.
                    if !messages[index].text.isEmpty {
                        let visible = ToolCallParser.parse(messages[index].text, markup: .dsv4).visibleText
                        messages[index].text = ToolCallParser.stripLeakedMarkup(visible, markup: .dsv4)
                    }
                case .progress(let p): status = p
                }
            }
            // The stream ended: the raw live markup was ephemeral feedback — drop it
            // (a parsed call shows as a card). Also scrub any malformed tool markup
            // the model emitted as text (degraded 2-bit output) so the final bubble
            // shows clean prose. A tool block that streamed but never parsed into a
            // call must NOT vanish silently: surface it as an explicit error row so
            // the user sees the model attempted (and botched) a tool call.
            if ownsConversationWork(epoch), index < messages.count {
                let unparsed = messages[index].toolStreamText.trimmingCharacters(in: .whitespacesAndNewlines)
                messages[index].toolStreamText = ""
                messages[index].text = ToolCallParser.stripLeakedMarkup(messages[index].text, markup: .dsv4)
                if !unparsed.isEmpty, messages[index].toolCalls.isEmpty {
                    messages.append(UIMessage(role: .tool,
                        text: "✗ malformed tool call (not executed): \(String(unparsed.prefix(300)))"))
                }
            }
        } catch is CancellationError {
            // User-initiated stop: keep the partial text, no error banner.
        } catch {
            guard ownsConversationWork(epoch) else { return }
            let tail = EngineLog.shared.tail()
            if index < messages.count {
                messages[index].text += "\n[errore: \(error)]"
                if !tail.isEmpty { messages[index].text += "\n\n--- log motore ---\n\(tail)" }
            }
        }
    }

    /// Execute the tool calls the assistant emitted: auto-run built-ins, collect
    /// manual ones, and continue the conversation. Returns true if generation
    /// continues (a continuation was spawned or we're awaiting manual input) — in
    /// which case the caller must NOT mark generation finished.
    func handleToolCalls(assistantIndex index: Int, epoch: UInt64) async -> Bool {
        guard ownsConversationWork(epoch), let service, index < messages.count else { return false }
        let calls = messages[index].toolCalls
        guard !calls.isEmpty else { return false }

        toolRounds += 1
        if toolRounds > maxToolRounds {
            messages.append(UIMessage(role: .tool, text: "Too many tool rounds (\(maxToolRounds)); stopped."))
            return false
        }

        // Slots mirror the model's call order exactly. Mixed automatic/manual
        // batches must not be regrouped by execution kind: DSML tool results are
        // positional, so even denial/duplicate/budget errors retain their slot.
        var outputSlots = [ToolOutput?](repeating: nil, count: calls.count)
        var manual: [ToolCall] = []
        var manualIndices: [Int] = []
        var currentFingerprints = Set<String>()
        let policy = ToolExecutionPolicy(allowedToolNames: activeConversationToolNames)
        let executableCalls = calls.prefix(maxToolCallsPerRound)
        for (callIndex, c) in executableCalls.enumerated() {
            // Check immediately before every call, including synchronous built-ins.
            // Stop/session changes advance the epoch, while direct task cancellation
            // closes the interval before that main-actor invalidation is observed.
            guard ownsConversationWork(epoch) else { return false }
            let fingerprint = c.fingerprint
            let firstInRound = currentFingerprints.insert(fingerprint).inserted
            if !firstInRound || previousToolRoundFingerprints.contains(fingerprint) {
                let content = #"{"error":"duplicate_tool_call","message":"identical consecutive tool call rejected; use the previous result or change the arguments"}"#
                messages.append(UIMessage(role: .tool, text: "✗ \(c.name) not executed: duplicate consecutive call"))
                outputSlots[callIndex] = ToolOutput(callId: c.id, name: c.name, content: content)
                continue
            }
            // Authorization is checked before every special execution path.
            // In particular, subagent_run is driven directly by the engine and
            // would otherwise bypass ToolRegistry's central policy boundary.
            if !policy.allows(c.name) {
                let out = ToolRegistry.execute(c, policy: policy)
                    ?? ToolOutput(callId: c.id, name: c.name,
                                  content: #"{"error":"tool_not_allowed"}"#)
                outputSlots[callIndex] = out
                messages.append(UIMessage(role: .tool, text: "✗ \(c.name) not executed: not allowed for this conversation"))
                continue
            }
            // subagent_run runs ON the engine (it drives the decoder in an isolated
            // context): the main KV only commits this call + the returned answer.
            if c.name == "subagent_run" {
                let (target, question, agent, tools) = Self.subAgentArgs(c.argumentsJSON)
                // A malformed call must fail loudly BEFORE spending a sub-agent run
                // on it: the explanatory error goes back to the model (so it can fix
                // the call) and into the transcript (so the failure is visible).
                if let problem = Self.subAgentCallProblem(
                    c.argumentsJSON, question: question, agent: agent, tools: tools,
                    allowedDelegatedTools: activeConversationDelegatedToolNames) {
                    messages.append(UIMessage(role: .tool, text: "✗ subagent_run not executed: \(problem)"))
                    outputSlots[callIndex] = ToolOutput(
                        callId: c.id, name: c.name,
                        content: "Error, sub-agent NOT run: \(problem)")
                    continue
                }
                guard ownsConversationWork(epoch) else { return false }
                status = "sub-agent su \(target)…"
                // Show the run in the transcript IMMEDIATELY (a sub-agent can take
                // minutes) and stream its internal steps into the card as they
                // happen; the placeholder is replaced in place when it finishes.
                let placeholder = messages.count
                messages.append(UIMessage(role: .tool, text: "",
                    subAgent: InferenceService.SubAgentRun(
                        target: target.isEmpty ? "project" : target, question: question,
                        answer: "", steps: []),
                    subAgentRunning: true))
                // The steps streamed so far: kept when the run errors out/stops, so
                // the transcript shows how far it got instead of losing the trace.
                func streamedSteps() -> [String] {
                    placeholder < messages.count ? (messages[placeholder].subAgent?.steps ?? []) : []
                }
                let run: InferenceService.SubAgentRun
                do {
                    run = try await service.runSubAgent(
                        target: target, question: question, agent: agent, tools: tools,
                        allowedTools: activeConversationDelegatedToolNames.sorted(),
                        onStep: { [weak self] step in
                            Task { @MainActor in
                                guard let self, self.conversationEpoch == epoch,
                                      placeholder < self.messages.count,
                                      self.messages[placeholder].subAgentRunning,
                                      let sa = self.messages[placeholder].subAgent else { return }
                                self.messages[placeholder].subAgent = InferenceService.SubAgentRun(
                                    target: sa.target, question: sa.question,
                                    answer: sa.answer, steps: sa.steps + [step])
                            }
                        })
                } catch is CancellationError {
                    run = InferenceService.SubAgentRun(target: target, question: question,
                                                       answer: "(sub-agent stopped)", steps: streamedSteps())
                } catch {
                    run = InferenceService.SubAgentRun(target: target, question: question,
                                                       answer: "Sub-agent error: \(error)", steps: streamedSteps())
                }
                // The await above yields the main actor. Stop/new chat/session switch
                // may have replaced `messages`; never index or append after that.
                guard ownsConversationWork(epoch) else { return false }
                if placeholder < messages.count, messages[placeholder].subAgent != nil {
                    messages[placeholder].subAgent = run
                    messages[placeholder].subAgentRunning = false
                } else {
                    messages.append(UIMessage(role: .tool, text: "", subAgent: run))
                }
                outputSlots[callIndex] = ToolOutput(callId: c.id, name: c.name, content: run.answer)
                continue
            }
            // Built-ins run synchronously; MCP tools go async to their server
            // (a failure — server gone, timeout — comes back as an error output
            // the model can react to).
            if MCPManager.shared.isMCPTool(named: c.name) { status = "MCP: \(c.name)…" }
            guard ownsConversationWork(epoch) else { return false }
            if let out = await ToolRegistry.executeAuto(c, policy: policy) {
                // MCP execution yields; even a synchronous built-in passes the
                // cancellation guard immediately above before it is dispatched.
                guard ownsConversationWork(epoch) else { return false }
                outputSlots[callIndex] = out
                messages.append(UIMessage(role: .tool, text: "\(c.name) → \(out.content)"))
                continue
            }
            manual.append(c)
            manualIndices.append(callIndex)
        }

        // A single malformed/degraded block must not execute an unbounded batch.
        // Return one explicit result for every dropped call so the transcript and
        // model remain structurally aligned.
        if calls.count > maxToolCallsPerRound {
            guard ownsConversationWork(epoch) else { return false }
            for callIndex in maxToolCallsPerRound..<calls.count {
                let c = calls[callIndex]
                let content = #"{"error":"tool_call_batch_limit","message":"call not executed: too many calls in one round"}"#
                outputSlots[callIndex] = ToolOutput(callId: c.id, name: c.name, content: content)
                messages.append(UIMessage(role: .tool, text: "✗ \(c.name) not executed: per-round limit \(maxToolCallsPerRound)"))
            }
        }
        guard ownsConversationWork(epoch) else { return false }
        previousToolRoundFingerprints = currentFingerprints

        if !manual.isEmpty {
            pendingOrderedToolOutputs = outputSlots
            pendingManualCalls = manual
            pendingManualOutputIndices = manualIndices
            pendingManualEpoch = epoch
            awaitingManualResults = true
            return true
        }
        guard outputSlots.allSatisfy({ $0 != nil }) else {
            messages.append(UIMessage(role: .tool,
                                      text: "✗ internal tool-result alignment error; continuation stopped"))
            return false
        }
        return continueWithToolOutputs(outputSlots.compactMap { $0 }, service: service,
                                       epoch: epoch)
    }

    /// Feed tool outputs back and stream the model's continuation (which may emit
    /// further tool calls — the loop repeats, bounded by maxToolRounds).
    @discardableResult
    func continueWithToolOutputs(_ outputs: [ToolOutput], service: InferenceService,
                                 epoch: UInt64) -> Bool {
        guard ownsConversationWork(epoch) else { return false }
        let index = appendAssistant()
        isGenerating = true
        let mode = thinkMode
        let params = sampling             // capture: `self` is weak inside the Task
        generation = Task(priority: .userInitiated) { [weak self] in   // see send(): keep decode QoS high
            let stream = await service.provideToolResults(outputs, thinkMode: mode,
                                                          sampling: params, maxTokens: 4096)
            guard let self, self.ownsConversationWork(epoch) else { return }
            await self.consume(stream, into: index, epoch: epoch)
            guard self.ownsConversationWork(epoch) else { return }
            let continued = await self.handleToolCalls(assistantIndex: index, epoch: epoch)
            if !continued { self.finishIfIdle(epoch: epoch) }
        }
        return true
    }
}
