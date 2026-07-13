import Foundation
import DS4Core

public enum ChatRole: Sendable { case system, user, assistant, tool }

public enum InferenceError: Error, CustomStringConvertible {
    case contextExceeded(prompt: Int, context: Int)
    public var description: String {
        switch self {
        case let .contextExceeded(p, c):
            return "the conversation (\(p) tokens) exceeds the context (\(c)). Start a new chat or increase the context."
        }
    }
}

public enum DS4ThinkMode: Sendable {
    case none, high
    var core: ThinkMode { self == .high ? .high : .none }
}

public struct SamplingParams: Sendable {
    public var temperature: Float
    public var topK: Int
    public var topP: Float
    public var minP: Float
    public var seed: UInt64
    /// Repetition penalty (llama.cpp `penalty_repeat`): >1 discourages re-emitting
    /// the last `repeatLastN` tokens — breaks the repeat-loop collapse on
    /// quantized models. 1.0 = off — the DEFAULT: the C original (ds4.c
    /// sample_top_p_min_p) has no penalty, so engine/server/demo stay faithful
    /// to it; the chat GUI opts in explicitly with its own user-set value.
    public var repetitionPenalty: Float
    public var repeatLastN: Int
    public init(temperature: Float = 0.6, topK: Int = 0, topP: Float = 0.95, minP: Float = 0.05,
                seed: UInt64 = 0xD54, repetitionPenalty: Float = 1.0, repeatLastN: Int = 64) {
        self.temperature = temperature; self.topK = topK; self.topP = topP; self.minP = minP
        self.seed = seed; self.repetitionPenalty = repetitionPenalty; self.repeatLastN = repeatLastN
    }
}

public struct ModelInfo: Sendable {
    public let name: String
    public let layers: Int
    public let nEmbd: Int
    public let nVocab: Int
    public let contextSize: Int
    public let routedQuantBits: Int
    public let kvCacheBytes: UInt64
}

public enum GenEvent: Sendable {
    case reasoning(String)
    case text(String)
    case toolStream(String)     // raw tool-call markup, streamed live during generation
    case toolCall([ToolCall])   // the model requested one or more tools; generation paused
    case progress(String)       // prefill/decode status (e.g. "prefill 3/11" or "12 tok · 1.4 tok/s")
}

