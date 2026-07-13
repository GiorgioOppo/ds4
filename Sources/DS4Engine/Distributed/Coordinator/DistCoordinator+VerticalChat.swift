import Foundation
import DS4Core

extension DistCoordinator {
    /// Esegue `work` su un thread GCD classico, MAI sul pool cooperativo di
    /// Swift Concurrency: il decode locale fa fan-out con concurrentPerform
    /// (gather esperti) — che da un thread cooperativo può degradare al
    /// quasi-seriale — e in verticale blocca anche sui round-trip di rete.
    static func onGCD<T: Sendable>(_ work: @escaping @Sendable () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { cont in
            DispatchQueue.global(qos: .userInitiated).async {
                cont.resume(with: Result(catching: work))
            }
        }
    }

    /// Un turno di chat sulla route VERTICALE (Fase D): stesso contratto di
    /// `send` — rendering della conversazione, streaming di reasoning/testo,
    /// DSML parse — ma il decode gira sul BACKBONE LOCALE (verticalEngine),
    /// con la FFN routed sui worker dentro ogni forward. Continuità KV in
    /// memoria (prefisso esteso → prefill del solo suffisso); disk-KV in D2.
    @discardableResult
    public func sendVertical(turns: [ChatTurn], tools: [ToolSpec] = [], think: Bool, maxTokens: Int,
                             sampling: SamplingParams,
                             onLog: @Sendable (String) -> Void,
                             onProgress: @Sendable (String) -> Void = { _ in },
                             onReasoning: @Sendable (String) -> Void,
                             onToken: @Sendable (String) -> Void) async throws -> [ToolCall] {
        guard let vEngine = verticalEngine else { throw DistError.closed }
        let ids = engine.chatPromptIds(turns: turns, tools: tools, think: think)
        guard ids.count < config.contextSize else { throw DistError.sliceGap("prompt exceeds context") }
        var startPos = 0
        if kvValid, !committedIds.isEmpty, ids.count > committedIds.count,
           ids.starts(with: committedIds) {
            startPos = committedIds.count
            onLog("KV riusato in memoria: \(startPos) token già nel backbone\n")
        }
        kvValid = false
        onLog("prefill \(ids.count - startPos) di \(ids.count) token (verticale)…\n")
        let suffix = Array(ids[startPos...])
        let sp = startPos
        var lastLogits = try await Self.onGCD {
            try vEngine.verticalPrefill(tokens: suffix, startPos: sp)
        }
        var pos = ids.count

        // Decode: identico a `send` (reasoning/think, '<' trattenuto, DSML
        // bufferizzato e parsato alla fine) — cambia solo CHI fa il forward.
        var rng = sampling.seed
        var produced = 0
        var producedIds: [Int] = []
        var inReasoning = think
        var inTool = false
        var pendingLT = false
        var visible = ""
        var toolText = ""
        var recentIds = Array(ids.suffix(sampling.repeatLastN))
        let t0 = Date()
        func emit(_ s: String) {
            if inReasoning { onReasoning(s) } else { visible += s; onToken(s) }
        }
        while produced < maxTokens {
            try Task.checkCancellation()
            let next = engine.sample(lastLogits, params: sampling, recent: recentIds[...], rng: &rng)
            if next == engine.eosId { break }
            if !inTool, next == engine.dsmlId {
                if pendingLT { pendingLT = false; toolText += "<" }
                inTool = true
                toolText += engine.tokenText(next)
            } else if inTool {
                toolText += engine.tokenText(next)
            } else if next == engine.thinkEndId { inReasoning = false }
            else if next == engine.thinkStartId { inReasoning = true }
            else {
                if pendingLT { pendingLT = false; emit("<") }
                let s = engine.tokenText(next)
                if s.hasSuffix("<") {
                    let head = String(s.dropLast())
                    if !head.isEmpty { emit(head) }
                    pendingLT = true
                } else {
                    emit(s)
                }
            }
            recentIds.append(next)
            if recentIds.count > sampling.repeatLastN { recentIds.removeFirst() }
            let p = pos
            lastLogits = try await Self.onGCD {
                try vEngine.verticalForward(token: next, pos: p)
            }
            producedIds.append(next)
            pos += 1; produced += 1
            let elapsed = Date().timeIntervalSince(t0)
            onProgress(String(format: "%d tokens · %.2f tok/s", produced,
                              elapsed > 0 ? Double(produced) / elapsed : 0))
        }
        if pendingLT, !inTool { emit("<") }
        let dt = Date().timeIntervalSince(t0)
        onLog("[\(produced) tokens · \(String(format: "%.2f", dt > 0 ? Double(produced) / dt : 0)) tok/s · verticale]\n")
        committedIds = ids + producedIds
        kvValid = true
        guard inTool else { return engine.parseToolCalls(visible).calls }
        return engine.parseToolCalls(visible + toolText).calls
    }
}
