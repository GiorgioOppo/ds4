import CDS4
import Foundation

/// Read-only facts about the loaded model, safe to hand to the UI.
public struct ModelInfo: Sendable {
    public let name: String
    public let layers: Int
    public let vocabSize: Int
    public let routedQuantBits: Int
    public let contextSize: Int
    /// Estimated KV cache + scratch bytes for `contextSize`.
    public let kvCacheBytes: UInt64
}

/// The single owner of the in-process engine and live session.
///
/// Every engine/session call is serialized by actor isolation, which matches the
/// C engine's single-graph-worker model: there is one live KV timeline and one
/// generation at a time. The UI talks to this actor with async calls and
/// consumes generated tokens as an `AsyncThrowingStream`.
public actor InferenceService {
    private let engine: DS4Engine
    private let session: ChatSession
    private var history: [ChatMessage] = []
    private var rng: UInt64

    /// Open the model and create the live session. This maps the GGUF and
    /// compiles the Metal kernels, so it is slow; call it off the main thread.
    /// When true, decode runs one transformer layer at a time via
    /// `ds4_session_eval_layer_slice`, hinting `MADV_DONTNEED` on each layer's
    /// weight pages between iterations. Intended for tight-RAM hosts.
    /// Incompatible with SSD streaming (the engine's native streaming decode
    /// path already handles per-layer mapping; replicating it from Swift via
    /// eval_layer_slice misses required static-map calls and breaks).
    public var perLayerStreaming: Bool = false
    private let ssdStreamingActive: Bool

    public init(modelPath: String,
                metalSourceDir: String?,
                contextSize: Int,
                systemPrompt: String?,
                streaming: StreamingOptions = .disabled,
                minimumRAMMode: Bool = false,
                perLayerStreaming: Bool = false,
                seed: UInt64 = 0x9E37_79B9_7F4A_7C15) throws {
        self.engine = try DS4Engine(modelPath: modelPath,
                                    backend: .metal,
                                    metalSourceDir: metalSourceDir,
                                    streaming: streaming,
                                    minimumRAMMode: minimumRAMMode)
        self.session = try ChatSession(engine: engine, contextSize: contextSize)
        self.rng = seed
        self.perLayerStreaming = perLayerStreaming
        self.ssdStreamingActive = streaming.enabled
        if let systemPrompt, !systemPrompt.isEmpty {
            history.append(ChatMessage(role: .system, content: systemPrompt))
        }
    }

    public func modelInfo() -> ModelInfo {
        let ctx = session.contextSize
        let mem = engine.contextMemoryEstimate(ctxSize: ctx)
        return ModelInfo(name: engine.modelName,
                         layers: engine.layerCount,
                         vocabSize: engine.vocabSize,
                         routedQuantBits: engine.routedQuantBits,
                         contextSize: ctx,
                         kvCacheBytes: mem.totalBytes)
    }

    /// Drop the conversation history (keeps the loaded model). The next turn
    /// re-prefills from the system prompt.
    public func resetConversation(systemPrompt: String?) {
        history.removeAll()
        if let systemPrompt, !systemPrompt.isEmpty {
            history.append(ChatMessage(role: .system, content: systemPrompt))
        }
    }

    /// Generate the assistant reply to `userText` as a stream of reasoning/text
    /// events. Cancelling the consuming task (or tearing down the stream) stops
    /// generation cooperatively; whatever was produced is kept in history so the
    /// transcript stays consistent with what the user saw.
    public func send(userText: String,
                     thinkMode: DS4ThinkMode,
                     sampling: SamplingParams,
                     maxTokens: Int) -> AsyncThrowingStream<DS4Event, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try self.runTurn(userText: userText,
                                     thinkMode: thinkMode,
                                     sampling: sampling,
                                     maxTokens: maxTokens,
                                     continuation: continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func runTurn(userText: String,
                         thinkMode: DS4ThinkMode,
                         sampling: SamplingParams,
                         maxTokens: Int,
                         continuation: AsyncThrowingStream<DS4Event, Error>.Continuation) throws {
        history.append(ChatMessage(role: .user, content: userText))

        // Re-render the whole transcript; sync reuses the KV prefix.
        try session.sync(engine.renderConversation(history, thinkMode: thinkMode))

        let eos = engine.eosToken
        var decoder = IncrementalUTF8()
        var splitter = ThinkSplitter(startsInReasoning: thinkMode != .none)
        var visibleAnswer = ""
        var produced = 0

        func emit(_ events: [DS4Event]) {
            for event in events {
                if case .text(let t) = event { visibleAnswer += t }
                continuation.yield(event)
            }
        }

        // Stage B PerLayerDriver is INCOMPATIBLE with SSD streaming: the engine
        // maps weights lazily and the native streaming decode loop already does
        // per-layer mapping (stream_map_decode_static_all + stream_map_layer_decode
        // before each layer). Replicating that from Swift via eval_layer_slice
        // misses the static map step and fails ("metal layer-slice decode
        // failed"). So we only enable Stage B when the model is fully resident.
        let useStageB = perLayerStreaming && !ssdStreamingActive
        let perLayer: PerLayerDriver? = useStageB
            ? PerLayerDriver(engine: engine, session: session) : nil

        while produced < maxTokens && !Task.isCancelled {
            let token = sampling.isGreedy ? session.argmax()
                                          : session.sample(sampling, rng: &rng)
            if token == eos { break }
            if let text = decoder.push(engine.tokenText(token)), !text.isEmpty {
                emit(splitter.push(text))
            }
            produced += 1
            if let perLayer {
                try perLayer.eval(token: token)
            } else {
                try session.eval(token)
            }
        }
        if let tail = decoder.flush(), !tail.isEmpty { emit(splitter.push(tail)) }
        emit(splitter.flush())

        history.append(ChatMessage(role: .assistant, content: visibleAnswer))
    }
}

/// Splits a decoded text stream into reasoning (chain-of-thought) and answer.
/// DeepSeek emits reasoning up to a `</think>` marker, then the visible answer.
/// In thinking mode the assistant prefix already opens reasoning, so generation
/// starts inside it. A possible marker straddling two chunks is held back.
struct ThinkSplitter {
    private var inReasoning: Bool
    private var pending = ""
    private let closeMarker = "</think>"

    init(startsInReasoning: Bool) {
        self.inReasoning = startsInReasoning
    }

    mutating func push(_ chunk: String) -> [DS4Event] {
        pending += chunk
        var events: [DS4Event] = []

        if inReasoning {
            if let r = pending.range(of: closeMarker) {
                let before = String(pending[pending.startIndex..<r.lowerBound])
                if !before.isEmpty { events.append(.reasoning(before)) }
                pending = String(pending[r.upperBound...])
                inReasoning = false
                // Fall through: remaining `pending` is answer text.
            } else {
                let keep = partialMarkerTail(pending, closeMarker)
                let emit = String(pending.dropLast(keep))
                pending = String(pending.suffix(keep))
                if !emit.isEmpty { events.append(.reasoning(emit)) }
                return events
            }
        }

        if !pending.isEmpty {
            events.append(.text(pending))
            pending = ""
        }
        return events
    }

    mutating func flush() -> [DS4Event] {
        guard !pending.isEmpty else { return [] }
        let event: DS4Event = inReasoning ? .reasoning(pending) : .text(pending)
        pending = ""
        return [event]
    }
}

/// Drives single-token decode one transformer layer at a time, hinting
/// `MADV_DONTNEED` on each finished layer's weights so the OS page cache can
/// release them. The resident weight working-set converges toward one layer
/// plus the always-touched globals (embedding, output head, norm).
///
/// Requirements: the engine must have been opened with all layers loaded (the
/// default) — slicing the weights at open time would defeat eviction since
/// untouched weights cannot be evicted (they are not even mapped).
final class PerLayerDriver {
    private let engine: DS4Engine
    private let session: ChatSession
    private let layerCount: Int
    private let hcCount: Int
    private var hcBufA: [Float]
    private var hcBufB: [Float]
    private var logits: [Float]
    /// Absolute KV position the next token will be written at. Mirrors the
    /// engine's per-layer KV state so layer-slice calls receive a correct pos.
    private var pos: UInt32

    init(engine: DS4Engine, session: ChatSession) {
        self.engine = engine
        self.session = session
        self.layerCount = engine.layerCount
        self.hcCount = engine.hiddenF32Values
        self.hcBufA = [Float](repeating: 0, count: hcCount)
        self.hcBufB = [Float](repeating: 0, count: hcCount)
        self.logits = [Float](repeating: 0, count: engine.vocabSize)
        // Start the layer-slice timeline from the session's current position.
        self.pos = UInt32(max(0, session.position))
    }

    func eval(token: Int32) throws {
        // Read newest logits back into the session so argmax/sample see them.
        try hcBufA.withUnsafeMutableBufferPointer { aPtr in
            try hcBufB.withUnsafeMutableBufferPointer { bPtr in
                try logits.withUnsafeMutableBufferPointer { lPtr in
                    var input: UnsafePointer<Float>? = nil
                    var outputBuf = aPtr
                    for layer in 0..<layerCount {
                        let isLast = (layer == layerCount - 1)
                        // SSD streaming maps weights lazily — map this layer's
                        // non-routed weights BEFORE the slice runs, otherwise
                        // Metal hits "model range not covered by mapped model
                        // views". No-op when streaming is off.
                        _ = engine.streamMapLayer(layer)
                        try session.evalLayerSlice(
                            token: token,
                            pos: pos,
                            layerStart: UInt32(layer),
                            layerEnd: UInt32(layer),
                            inputHC: input,
                            outputHC: isLast ? nil : outputBuf.baseAddress,
                            logits: isLast ? lPtr.baseAddress : nil)
                        // Tell the OS this layer's weight pages are evict-eligible.
                        engine.adviseLayerDontNeed(layer)
                        if !isLast {
                            input = UnsafePointer(outputBuf.baseAddress)
                            outputBuf = (outputBuf.baseAddress == aPtr.baseAddress) ? bPtr : aPtr
                        }
                    }
                    // Publish the logits we just computed into the session, so the
                    // outer loop's argmax/sample pick from the correct distribution.
                    _ = ds4_session_set_logits(session.handle,
                                               lPtr.baseAddress,
                                               Int32(lPtr.count))
                }
            }
        }
        pos += 1
    }
}

/// Longest suffix of `s` that is also a prefix of `marker` (the part we must
/// hold back because the marker might complete in the next chunk).
private func partialMarkerTail(_ s: String, _ marker: String) -> Int {
    let maxK = min(s.count, marker.count - 1)
    guard maxK > 0 else { return 0 }
    let sChars = Array(s), mChars = Array(marker)
    for k in stride(from: maxK, through: 1, by: -1) {
        if Array(sChars.suffix(k)) == Array(mChars.prefix(k)) { return k }
    }
    return 0
}
