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
                                            question: String, agent: String, tools: [String]) -> String? {
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
        if !tools.isEmpty, !tools.contains(where: { ToolRegistry.subAgentGrantable.contains($0) }) {
            return "none of the requested tools exist or are grantable to sub-agents; grantable tools: \(ToolRegistry.subAgentGrantable.sorted().joined(separator: ", "))"
        }
        return nil
    }

    func appendAssistant() -> Int {
        messages.append(UIMessage(role: .assistant, text: ""))
        return messages.count - 1
    }

    func finishIfIdle() {
        if pendingManualCalls.isEmpty {
            isGenerating = false
            status = ""
            refreshContextUsage()
            persistActiveSession()        // checkpoint the completed turn
            // Il Profilo decode va nel Log motore DOPO OGNI risposta: i contatori
            // sono raccolti comunque, il report costa nulla, e "a quanto genera
            // davvero l'app e dove va il tempo" deve essere leggibile dal log
            // senza attivare niente. (profileRouteEnabled resta il gate della
            // sola scomposizione route/attn, che aggiunge sync GPU.)
            emitDecodeProfile()
        }
    }

    /// Print the last turn's prefill + decode profiles to stderr so they land in
    /// the Log motore (EngineLog captures fd 2), mirroring the demo's DIAG output.
    private func emitDecodeProfile() {
        guard let service else { return }
        Task {
            let prefill = await service.prefillProfileReport()
            let report = await service.decodeProfileReport()
            FileHandle.standardError.write(Data(("\n" + prefill + "\n\n" + report + "\n").utf8))
        }
    }

    /// Refresh the committed-token count (context usage) from the engine.
    private func refreshContextUsage() {
        guard let service else { contextUsed = 0; return }
        Task { contextUsed = await service.committedTokens() }
    }

    /// Drain one generation stream into the assistant message at `index`.
    func consume(_ stream: AsyncThrowingStream<GenEvent, Error>, into index: Int) async {
        do {
            for try await event in stream {
                guard index < messages.count else { break }
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
            if index < messages.count {
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
    func handleToolCalls(assistantIndex index: Int) async -> Bool {
        guard let service, index < messages.count else { return false }
        let calls = messages[index].toolCalls
        guard !calls.isEmpty else { return false }

        toolRounds += 1
        if toolRounds > maxToolRounds {
            messages.append(UIMessage(role: .tool, text: "Too many tool rounds (\(maxToolRounds)); stopped."))
            return false
        }

        var outputs: [ToolOutput] = []
        var manual: [ToolCall] = []
        for c in calls {
            // subagent_run runs ON the engine (it drives the decoder in an isolated
            // context): the main KV only commits this call + the returned answer.
            if c.name == "subagent_run" {
                let (target, question, agent, tools) = Self.subAgentArgs(c.argumentsJSON)
                // A malformed call must fail loudly BEFORE spending a sub-agent run
                // on it: the explanatory error goes back to the model (so it can fix
                // the call) and into the transcript (so the failure is visible).
                if let problem = Self.subAgentCallProblem(c.argumentsJSON, question: question,
                                                          agent: agent, tools: tools) {
                    messages.append(UIMessage(role: .tool, text: "✗ subagent_run not executed: \(problem)"))
                    outputs.append(ToolOutput(callId: c.id, name: c.name,
                                              content: "Error, sub-agent NOT run: \(problem)"))
                    continue
                }
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
                        onStep: { [weak self] step in
                            Task { @MainActor in
                                guard let self, placeholder < self.messages.count,
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
                if placeholder < messages.count, messages[placeholder].subAgent != nil {
                    messages[placeholder].subAgent = run
                    messages[placeholder].subAgentRunning = false
                } else {
                    messages.append(UIMessage(role: .tool, text: "", subAgent: run))
                }
                outputs.append(ToolOutput(callId: c.id, name: c.name, content: run.answer))
                continue
            }
            // Built-ins run synchronously; MCP tools go async to their server
            // (a failure — server gone, timeout — comes back as an error output
            // the model can react to).
            if MCPManager.shared.isMCPTool(named: c.name) { status = "MCP: \(c.name)…" }
            if let out = await ToolRegistry.executeAuto(c) {
                outputs.append(out)
                messages.append(UIMessage(role: .tool, text: "\(c.name) → \(out.content)"))
                // Stop pressed during the (long) MCP await: show the result but
                // do NOT spawn a continuation — the user asked this turn to end.
                if Task.isCancelled { return false }
                continue
            }
            manual.append(c)
        }

        if !manual.isEmpty {
            partialAutoOutputs = outputs
            pendingManualCalls = manual
            awaitingManualResults = true
            return true
        }
        continueWithToolOutputs(outputs, service: service)
        return true
    }

    /// Feed tool outputs back and stream the model's continuation (which may emit
    /// further tool calls — the loop repeats, bounded by maxToolRounds).
    func continueWithToolOutputs(_ outputs: [ToolOutput], service: InferenceService) {
        let index = appendAssistant()
        isGenerating = true
        let mode = thinkMode
        let params = sampling             // capture: `self` is weak inside the Task
        generation = Task(priority: .userInitiated) { [weak self] in   // see send(): keep decode QoS high
            let stream = await service.provideToolResults(outputs, thinkMode: mode,
                                                          sampling: params, maxTokens: 4096)
            await self?.consume(stream, into: index)
            let continued = await self?.handleToolCalls(assistantIndex: index) ?? false
            if !continued { await MainActor.run { self?.finishIfIdle() } }
        }
    }
}
