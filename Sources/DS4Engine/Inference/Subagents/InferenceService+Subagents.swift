import Foundation
import DS4Core
import DS4Metal

extension InferenceService {
// MARK: - Sub-agents (isolated context, returns only the answer)

    /// The outcome of a sub-agent run. `answer` is fed back to the main agent as a
    /// tool result (so the main KV commits only the question + this answer); `steps`
    /// is a display-only trace of the sub-agent's internal tool rounds.
    public struct SubAgentRun: Sendable {
        public let target: String
        public let question: String
        public let answer: String
        public let steps: [String]
        public init(target: String, question: String, answer: String, steps: [String]) {
            self.target = target; self.question = question; self.answer = answer; self.steps = steps
        }
    }

    struct SubContext { let system: String; let content: String; let tools: [ToolSpec]; let label: String; let toolNames: [String] }

    /// Resolve a sub-agent target ("project"/"" or a project file path), an optional
    /// ROLE to assume (`agentId`), and the MINIMAL tool set into the sub-agent's
    /// system prompt, the content block that seeds the KV, and the declared tools.
    /// Granted tools = explicit `requested` (∩ grantable), else the role's tools,
    /// else a read-only default. Works WITHOUT an imported project: that is not an
    /// error — the sub-agent then runs on the task alone (no project content/tools).
    /// `seedFileContent: false` builds the FALLBACK context for a file target too
    /// large to preload: same file focus, but the sub-agent reads it in chunks
    /// with the read tools instead of having the text seeded into the prefix.
    func subContext(for target: String, agent agentId: String, toolNames requested: [String],
                            seedFileContent: Bool = true) -> SubContext {
        let t = target.trimmingCharacters(in: .whitespacesAndNewlines)
        let isProject = t.isEmpty || t.lowercased() == "project" || t == "."
        let role = agentId.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil
            : AgentRegistry.shared.all().first { $0.id == agentId.trimmingCharacters(in: .whitespacesAndNewlines) }
        let projectInfo = ProjectCache.shared.info()
        let fileText = (projectInfo != nil && !isProject && seedFileContent)
            ? ProjectCache.shared.fullText(of: t) : nil

        // Granted = explicit ∩ grantable, else role's tools ∩ grantable, else default.
        var granted = requested.filter { ToolRegistry.subAgentGrantable.contains($0) }
        if granted.isEmpty, let role { granted = role.toolNames.filter { ToolRegistry.subAgentGrantable.contains($0) } }
        if granted.isEmpty {
            // File targets also get raw file_read/file_lines: the target may exist
            // in the root without being indexed (too large / unusual extension).
            granted = projectInfo == nil ? []
                : (isProject ? ["project_tree", "project_list", "project_find", "project_read", "project_search"]
                             : ["project_read", "project_search", "file_read", "file_lines"])
        }
        // Without an imported project, project-scoped tools can't do anything → drop them.
        if projectInfo == nil { granted = granted.filter { !ToolRegistry.projectScoped.contains($0) } }
        var seen = Set<String>(); granted = granted.filter { seen.insert($0).inserted }   // stable de-dup
        let specs = ToolRegistry.specs(enabled: Set(granted))
        var toolLine = granted.isEmpty ? "You have no tools: answer from your own knowledge."
                                       : "Available tools (use only these): " + granted.joined(separator: ", ") + "."
        if granted.contains("project_read") {
            // Every tool round costs a full prefill+decode on a local model: steer
            // the sub-agent away from paging a long file 120 lines at a time.
            toolLine += " Read long files in FEW large chunks: project_read accepts 'lines' up to 400 per call."
        }
        let rolePrefix = (role.map { $0.systemPrompt.isEmpty ? "" : $0.systemPrompt + "\n\n" }) ?? ""
        let roleLabel = role.map { " · \($0.name)" } ?? ""

        if let info = projectInfo, isProject {
            let map = ProjectCache.shared.fileList().prefix(200).joined(separator: "\n")
            let content = "Project \"\(info.name)\" - \(info.fileCount) files.\nPartial map:\n\(map)\n\n"
            let sys = rolePrefix + "You are an autonomous sub-agent working only on the imported project. \(toolLine) Conclude with a concise answer: what you found/did, with file:line."
            return SubContext(system: sys, content: content, tools: specs, label: "project:\(info.name)\(roleLabel)", toolNames: granted)
        }
        if let text = fileText {
            let content = "Contents of file \"\(t)\" (already in context):\n```\n\(text)\n```\n\n"
            let sys = rolePrefix + "You are a sub-agent focused on file \"\(t)\", already in context. \(toolLine) If you edit, act only on this file (exact and unique find text, including indentation). Conclude with a concise answer."
            return SubContext(system: sys, content: content, tools: specs, label: "file:\(t)\(roleLabel)", toolNames: granted)
        }
        if projectInfo != nil, !isProject, !seedFileContent,
           let full = ProjectCache.shared.fullText(of: t) {
            // File target NOT seeded (too large for the context): keep the file
            // focus, hand over chunked reading instead of the content itself.
            let lineCount = full.components(separatedBy: "\n").count
            let content = "Target file \"\(t)\" (~\(lineCount) lines): too large to preload into context.\n\n"
            let sys = rolePrefix + "You are a sub-agent focused on file \"\(t)\" (~\(lineCount) lines), which is TOO LARGE to preload. \(toolLine) Read only the parts the task needs, in few LARGE chunks (project_read with 'lines' up to 400, continuing with from_line). Conclude with a concise answer."
            return SubContext(system: sys, content: content, tools: specs,
                              label: "file:\(t)\(roleLabel)", toolNames: granted)
        }
        // No project imported (or the file isn't in it): a plain sub-agent that
        // answers the task directly — NOT an error (a chat may have no project).
        // A file that EXISTS in the root but is outside the text index (too
        // large / binary-looking) is still reachable via raw ranged reads.
        var note = ""
        if projectInfo != nil, !isProject {
            if !t.contains(".."), let root = ProjectCache.shared.rootURL(),
               FileManager.default.fileExists(atPath: root.appendingPathComponent(t).path) {
                note = "Note: \"\(t)\" exists but is NOT in the text index (too large or binary): check its size with file_lines and read it in ranges with file_read (from_line/to_line). "
            } else {
                note = "Note: \"\(t)\" is not in the imported project. "
            }
        }
        let sys = rolePrefix + "You are a sub-agent. \(note)\(toolLine) Complete the task and conclude with a concise answer."
        return SubContext(system: sys, content: "", tools: specs, label: "task\(roleLabel)", toolNames: granted)
    }

    /// Run an isolated sub-agent on `target` with `question`. The MAIN conversation
    /// KV is snapshotted and restored around the run, so the caller's context only
    /// ever sees the question (the tool call) and this answer (the tool result) —
    /// the sub-agent's internal tool rounds happen in a separate, discarded context.
    /// The target's content prefix is cached (content-keyed) and reused next time.
    /// `onStep` fires as each internal step is recorded (KV reuse, tool calls…),
    /// so the UI can show live execution detail; the same lines end up in `steps`.
    /// `maxRounds` bounds the tool rounds (default 16 — a degraded model can
    /// otherwise loop on tools forever); when it runs out the sub-agent is asked
    /// to answer with what it has instead of returning empty-handed.
    /// `maxTokens` caps ONE decode turn and includes the (discarded) reasoning
    /// tokens: 1024 was routinely eaten by a long think before any visible
    /// answer, which surfaced as sub-agents "cutting their replies" — hence the
    /// 4096 default (a cap, not a target: EOS ends the turn normally).
    public func runSubAgent(target: String, question: String, agent: String = "", tools: [String] = [],
                            maxTokens: Int = 4096, maxRounds: Int = 16,
                            onStep: (@Sendable (String) -> Void)? = nil) async throws -> SubAgentRun {
        var ctx = subContext(for: target, agent: agent, toolNames: tools)
        // A dirty main KV must be rebuilt before snapshotting so the restore is exact.
        if kvDirty, !committedIds.isEmpty { _ = try decoder.prefill(tokens: committedIds, startPos: 0); kvDirty = false }

        // Snapshot the MAIN context and restore it however the sub-agent ends.
        let savedIds = committedIds, savedClose = needsClose, savedDirty = kvDirty, savedDisk = lastDiskStoreCount
        let mainSnap: KVSnapshot? = savedIds.isEmpty ? nil : decoder.exportKV(nKeys: savedIds.count)
        defer {
            committedIds = savedIds; needsClose = savedClose; lastDiskStoreCount = savedDisk
            if let mainSnap {
                do { try decoder.importKV(mainSnap); kvDirty = savedDirty }
                catch { kvDirty = true }   // next main turn rebuilds from committedIds
            } else { kvDirty = true }
        }

        var steps: [String] = []
        func note(_ s: String) { steps.append(s); onStep?(s) }
        // Tool results go into the trace as a BOUNDED excerpt: the first lines are
        // kept verbatim (a project_read step really shows the lines it read), but
        // the step can't blow the card up — capped in lines and characters.
        func excerpt(_ s: String) -> String {
            let maxLines = 8, maxChars = 600
            var lines = s.split(separator: "\n", omittingEmptySubsequences: false)
            let dropped = max(0, lines.count - maxLines)
            if dropped > 0 { lines = Array(lines.prefix(maxLines)) }
            var out = lines.joined(separator: "\n")
            if out.count > maxChars { out = String(out.prefix(maxChars)) + "…" }
            if dropped > 0 { out += "\n… (+\(dropped) more lines)" }
            return out
        }
        let sampling = SamplingParams()

        // 1. Build or restore the content-keyed KV prefix (lazy cache).
        func renderPrefix(_ c: SubContext) -> [Int] {
            let text = "<｜begin▁of▁sentence｜>"
                + ChatRenderer.systemBlock(turns: [.system(c.system)], tools: c.tools, markup: markup, compact: true)
                + "<｜User｜>" + c.content
            return tok.tokenizeRenderedChat(text).map { Int($0) }
        }
        var prefixIds = renderPrefix(ctx)
        if prefixIds.count >= contextSize - 32 {
            // File target too large to preload: fall back to CHUNKED-READ mode
            // (same file focus, read tools instead of seeded content) rather
            // than refusing the run outright.
            let oversized = prefixIds.count
            ctx = subContext(for: target, agent: agent, toolNames: tools, seedFileContent: false)
            prefixIds = renderPrefix(ctx)
            guard prefixIds.count < contextSize - 32 else {
                return SubAgentRun(target: ctx.label, question: question,
                                   answer: "The contents of \"\(ctx.label)\" exceed the sub-agent context.", steps: steps)
            }
            note("target too large to preload (\(oversized) tokens vs context \(contextSize)): chunked-read mode")
        }
        var pos = 0
        if let subKV, subKV.restore(forTokens: prefixIds, modelName: modelName, into: decoder) {
            // Streamed layer-by-layer from disk (peak RAM = one layer).
            pos = prefixIds.count
            note("KV \"\(ctx.label)\" reused (\(pos) tokens)")
        } else {
            note("prefill \"\(ctx.label)\" (\(prefixIds.count) tokens)…")
            _ = try decoder.prefill(tokens: prefixIds, startPos: 0); pos = prefixIds.count
            // SnapshotBox: the writer owns the export and frees each layer as
            // it lands on disk instead of holding the whole snapshot.
            subKV?.store(tokens: prefixIds, modelName: modelName,
                         box: .init(decoder.exportKV(nKeys: pos)), reason: .cold)
            note("KV \"\(ctx.label)\" created (\(pos) tokens)")
        }
        note("tool: " + (ctx.toolNames.isEmpty ? "(none)" : ctx.toolNames.joined(separator: ", ")))

        // 2. Sub-agent tool loop: question → answer/tool-calls → results → … (bounded).
        var recent: [Int] = []
        var suffix = question + assistantOpen(.none)
        var answer = ""
        var round = 0
        while true {
            let suffixIds = tok.tokenizeRenderedChat(suffix).map { Int($0) }
            guard pos + suffixIds.count < contextSize else { note("sub-agent context exhausted"); break }
            note("round \(round + 1): generating…")
            var lastLogits = try decoder.prefill(tokens: suffixIds, startPos: pos)
            pos += suffixIds.count
            let turn = try decodeSubTurn(lastLogits: &lastLogits, pos: &pos, recent: &recent,
                                         sampling: sampling, maxTokens: maxTokens)
            answer = turn.visible
            var calls = turn.calls
            if turn.truncated {
                if turn.openTool {
                    // The block never closed: any parsed call may carry half-generated
                    // arguments — executing it could edit the wrong thing. Drop it.
                    note("output cap (\(maxTokens) tokens) hit inside a tool call: truncated call discarded")
                    calls = []
                }
                if calls.isEmpty {
                    note("output cap (\(maxTokens) tokens) reached: answer truncated")
                    if !answer.isEmpty { answer += "\n…[truncated at the \(maxTokens)-token output cap]" }
                }
            }
            guard !calls.isEmpty else { break }
            // Budget exhausted with the model still asking for tools: force one
            // last answer from what it gathered instead of returning empty-handed.
            if round >= maxRounds {
                note("round budget (\(maxRounds)) exhausted: forcing the final answer")
                suffix = "<｜end▁of▁sentence｜><｜User｜>Tool budget exhausted: no more tool calls. Answer the question now, concisely, with what you have."
                    + assistantOpen(.none)
                let finalIds = tok.tokenizeRenderedChat(suffix).map { Int($0) }
                if pos + finalIds.count < contextSize {
                    var ll = try decoder.prefill(tokens: finalIds, startPos: pos)
                    pos += finalIds.count
                    let fin = try decodeSubTurn(lastLogits: &ll, pos: &pos, recent: &recent,
                                                sampling: sampling, maxTokens: maxTokens)
                    if !fin.visible.isEmpty { answer = fin.visible }
                }
                break
            }
            round += 1
            var results = ""
            for c in calls {
                let out = ToolRegistry.execute(c)
                    ?? ToolOutput(callId: c.id, name: c.name, content: #"{"error":"tool unavailable in sub-agent"}"#)
                note("\(c.name) \(c.argumentsJSON) → " + excerpt(out.content))
                results += "<tool_result>" + ChatRenderer.escapeToolResult(out.content) + "</tool_result>"
            }
            suffix = "<｜end▁of▁sentence｜><｜User｜>" + results + assistantOpen(.none)
        }
        let final = answer.trimmingCharacters(in: .whitespacesAndNewlines)
        note("answer: \(final.count) characters")
        return SubAgentRun(target: ctx.label, question: question,
                           answer: final.isEmpty ? "(no answer)" : final, steps: steps)
    }

    /// Decode one assistant turn in the sub-agent context: returns the visible
    /// answer text and any tool calls. Reasoning (<think>…</think>) is discarded —
    /// but it still CONSUMES the token budget, so callers must treat `truncated`
    /// turns explicitly (a long think can eat the whole cap before any answer).
    /// `truncated` = the turn ended on the cap/context, not on EOS; `openTool` =
    /// it ended INSIDE a tool block (any parsed calls have unreliable arguments).
    func decodeSubTurn(lastLogits: inout [Float], pos: inout Int, recent: inout [Int],
                               sampling: SamplingParams, maxTokens: Int) throws
        -> (visible: String, calls: [ToolCall], truncated: Bool, openTool: Bool) {
        var rng = sampling.seed &+ UInt64(pos)
        var inTool = false, inReasoning = false
        var sawEos = false
        var visibleBytes: [UInt8] = []
        var toolBytes: [UInt8] = []
        let dsmlId = tok.dsmlId
        var produced = 0
        while produced < maxTokens && pos < contextSize {
            try Task.checkCancellation()
            let lo = max(0, recent.count - sampling.repeatLastN)
            let next = Sampler.sample(lastLogits, temperature: sampling.temperature, topK: sampling.topK,
                                      topP: sampling.topP, minP: sampling.minP,
                                      repetitionPenalty: sampling.repetitionPenalty,
                                      recent: recent[lo...], rng: &rng)
            if Int32(next) == tok.eosId { sawEos = true; break }
            if !inTool, Int32(next) == dsmlId {
                inTool = true; toolBytes.append(contentsOf: tok.tokenText(Int32(next)))
            } else if inTool {
                toolBytes.append(contentsOf: tok.tokenText(Int32(next)))
            } else if Int32(next) == tok.thinkStartId {
                inReasoning = true
            } else if Int32(next) == tok.thinkEndId {
                inReasoning = false
            } else if !inReasoning {
                visibleBytes.append(contentsOf: tok.tokenText(Int32(next)))
            }
            produced += 1
            recent.append(next)
            lastLogits = try decoder.forward(token: next, pos: pos, nKeys: pos + 1)
            pos += 1
        }
        let visibleRaw = String(bytes: visibleBytes, encoding: .utf8) ?? ""
        let toolText = String(bytes: toolBytes, encoding: .utf8) ?? ""
        let parsed = ToolCallParser.parse(inTool ? visibleRaw + toolText : visibleRaw, markup: markup)
        return (parsed.visibleText.trimmingCharacters(in: .whitespacesAndNewlines), parsed.calls,
                !sawEos, inTool)
    }

    /// Per-phase decode timing (route/attn vs expert gather I/O vs experts compute…).
}

