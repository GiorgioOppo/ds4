import Foundation

/// V4.1 introduces an explicit System token and merged user/tool turns.
public enum DeepSeek41ChatRenderer {
    public static func effort(_ reasoning: ThinkMode) -> String? {
        guard reasoning.enabled else { return nil }
        return "Reasoning Effort: \(reasoning == .max ? 100 : 75) (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
    }
    public static func render(turns: [ChatTurn], tools: [ToolSpec] = [],
                              reasoning: ThinkMode = .none,
                              compactTools: Bool = false) throws -> String {
        try ToolHistoryValidator.validate(turns)
        func safe(_ text: String) -> String {
            DeepSeekV4Tokenizer.neutralizeSpecialTokenLiterals(in: text, specialTokens: [
                "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜System｜>",
                "<｜User｜>", "<｜Assistant｜>", "<think>", "</think>", "｜DSML｜"
            ])
        }
        var out = "<｜begin▁of▁sentence｜>"
        let effort = effort(reasoning)
        let initialSystem = effort != nil || !tools.isEmpty
        if initialSystem {
            out += "<｜System｜>" + (effort ?? "")
            if !tools.isEmpty { out += ChatRenderer.systemBlock(turns: [], tools: tools, markup: .dsv4, compact: compactTools) }
        }
        // Parallel tool results are restored to the preceding call order.
        var ordered = turns, callOrder: [String: Int] = [:], index = 0
        while index < ordered.count {
            if case .assistant(_, let calls) = ordered[index] {
                callOrder = [:]
                for (offset, call) in calls.enumerated() where !call.id.isEmpty { callOrder[call.id] = offset }
            }
            if case .toolResult = ordered[index] {
                var end = index + 1
                while end < ordered.count {
                    guard case .toolResult = ordered[end] else { break }; end += 1
                }
                let sorted = ordered[index..<end].enumerated().sorted { lhs, rhs in
                    func rank(_ turn: ChatTurn, fallback: Int) -> Int {
                        guard case .toolResult(let id, _, _) = turn else { return fallback }
                        return callOrder[id] ?? (callOrder.count + fallback)
                    }
                    return rank(lhs.element, fallback: lhs.offset) < rank(rhs.element, fallback: rhs.offset)
                }.map(\.element)
                ordered.replaceSubrange(index..<end, with: sorted); index = end
            } else { index += 1 }
        }
        var pending = false, userOpen = false
        for (index, turn) in ordered.enumerated() {
            switch turn {
            case .system(let content):
                if index == 0 && initialSystem { if !tools.isEmpty { out += "\n\n" } }
                else { out += "<｜System｜>" }
                out += safe(content); pending = index > 0; userOpen = false
            case .user(let content):
                out += (userOpen ? "\n\n" : "<｜User｜>") + safe(content)
                pending = true; userOpen = true
            case .toolResult(_, _, let content):
                out += (userOpen ? "\n\n" : "<｜User｜>") + "<tool_result>" + ChatRenderer.escapeToolResult(safe(content)) + "</tool_result>"
                pending = true; userOpen = true
            case .assistant(let text, let calls):
                if pending { out += "<｜Assistant｜></think>" }
                out += safe(text)
                if !calls.isEmpty { out += ChatRenderer.renderToolCalls(calls, markup: .dsv4) }
                out += "<｜end▁of▁sentence｜>"; pending = false; userOpen = false
            }
        }
        if pending { out += "<｜Assistant｜>" + (reasoning.enabled ? "<think>" : "</think>") }
        return out
    }
}
