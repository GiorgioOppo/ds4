import CDS4
import Foundation

/// One mutable inference timeline over an engine: owns the live KV cache and the
/// next-token logits. Wraps the C `ds4_session`.
///
/// Phase 0 exposes a synchronous greedy generator that drives the decode loop
/// token-by-token, which gives the caller full control over streaming and
/// cancellation. Sampling, async streaming, and KV reuse across turns come in
/// later phases.
public final class ChatSession {
    let handle: OpaquePointer            // ds4_session *
    private let engine: DS4Engine        // keep the engine alive

    public init(engine: DS4Engine, contextSize: Int) throws {
        self.engine = engine
        var s: OpaquePointer?
        let rc = ds4_session_create(&s, engine.handle, Int32(contextSize))
        guard rc == 0, let sp = s else {
            throw DS4Error.sessionCreateFailed(rc)
        }
        self.handle = sp
    }

    deinit {
        ds4_session_free(handle)
    }

    public var position: Int { Int(ds4_session_pos(handle)) }
    public var contextSize: Int { Int(ds4_session_ctx(handle)) }

    // MARK: - Low-level decode primitives

    /// Prefill/synchronize the live session to a rendered token prefix, reusing
    /// the existing KV checkpoint for the common prefix.
    func sync(_ tokens: Tokens) throws {
        var err = [CChar](repeating: 0, count: 512)
        // autoreleasepool: the Metal backend creates many autoreleased command
        // buffers/encoders per prefill. Driven off the main thread there is no
        // ambient pool, so drain explicitly to avoid unbounded GPU buffer buildup.
        let rc = autoreleasepool {
            err.withUnsafeMutableBufferPointer { eb in
                ds4_session_sync(handle, &tokens.raw, eb.baseAddress, eb.count)
            }
        }
        guard rc == 0 else { throw DS4Error.syncFailed(errorString(err)) }
    }

    /// Argmax of the current next-token logits.
    func argmax() -> Int32 { ds4_session_argmax(handle) }

    /// Sample the next token from the current logits.
    func sample(_ params: SamplingParams, rng: inout UInt64) -> Int32 {
        ds4_session_sample(handle, params.temperature, params.topK, params.topP, params.minP, &rng)
    }

    // MARK: - Per-layer slice primitives
    //
    // The two helpers below wrap ds4_session_eval_layer_slice and
    // ds4_session_eval_output_head_from_hc, used by the per-layer streaming
    // driver to run one layer of the decode at a time and to apply the output
    // head after the last layer.

    /// Reset the layer-slice timeline (engine-internal token bookkeeping).
    func layerSliceReset() throws {
        var err = [CChar](repeating: 0, count: 512)
        let rc = err.withUnsafeMutableBufferPointer { eb in
            ds4_session_layer_slice_reset(handle, eb.baseAddress, eb.count)
        }
        guard rc == 0 else { throw DS4Error.syncFailed(errorString(err)) }
    }

    /// Evaluate `token` at absolute position `pos` through layers
    /// [layerStart, layerEnd]. The first slice may pass `inputHC == nil` to let
    /// the engine compute embedding internally; later slices pass the previous
    /// slice's HC output. Optionally writes HC or final logits.
    func evalLayerSlice(token: Int32,
                        pos: UInt32,
                        layerStart: UInt32,
                        layerEnd: UInt32,
                        inputHC: UnsafePointer<Float>?,
                        outputHC: UnsafeMutablePointer<Float>?,
                        logits: UnsafeMutablePointer<Float>?) throws {
        var err = [CChar](repeating: 0, count: 512)
        var t = token
        let wantLogits = logits != nil
        let rc = autoreleasepool {
            err.withUnsafeMutableBufferPointer { eb in
                ds4_session_eval_layer_slice(handle,
                                             &t, 1, pos,
                                             layerStart, layerEnd,
                                             inputHC, outputHC,
                                             wantLogits, logits,
                                             eb.baseAddress, eb.count)
            }
        }
        guard rc == 0 else { throw DS4Error.evalFailed(errorString(err)) }
    }

    /// Apply the output head to a hidden state, producing logits.
    func evalOutputHeadFromHC(hidden: UnsafePointer<Float>,
                              logits: UnsafeMutablePointer<Float>) throws {
        var err = [CChar](repeating: 0, count: 512)
        let rc = autoreleasepool {
            err.withUnsafeMutableBufferPointer { eb in
                ds4_session_eval_output_head_from_hc(handle, hidden, 1, logits,
                                                     eb.baseAddress, eb.count)
            }
        }
        guard rc == 0 else { throw DS4Error.evalFailed(errorString(err)) }
    }

    /// Advance the session by committing `token`, producing fresh next logits.
    func eval(_ token: Int32) throws {
        var err = [CChar](repeating: 0, count: 512)
        let rc = autoreleasepool {
            err.withUnsafeMutableBufferPointer { eb in
                ds4_session_eval(handle, token, eb.baseAddress, eb.count)
            }
        }
        guard rc == 0 else { throw DS4Error.evalFailed(errorString(err)) }
    }

    /// Greedy (argmax) generation for a single chat turn.
    ///
    /// Builds the DeepSeek chat prompt, prefills it, then decodes up to
    /// `maxTokens` tokens, invoking `onToken` with decoded UTF-8 text deltas.
    /// Returns the number of tokens generated. Returning `false` from
    /// `shouldContinue` stops generation early (cooperative cancellation).
    @discardableResult
    public func generateGreedy(system: String?,
                               prompt: String,
                               thinkMode: DS4ThinkMode = .none,
                               maxTokens: Int,
                               shouldContinue: () -> Bool = { true },
                               onToken: (String) -> Void) throws -> Int {
        // 1. Render the single-turn conversation and prefill it.
        var history: [ChatMessage] = []
        if let system { history.append(ChatMessage(role: .system, content: system)) }
        history.append(ChatMessage(role: .user, content: prompt))
        try sync(engine.renderConversation(history, thinkMode: thinkMode))

        // 2. Decode loop. After sync, logits already describe the next token.
        let eos = engine.eosToken
        var decoder = IncrementalUTF8()
        var produced = 0

        while produced < maxTokens && shouldContinue() {
            let token = argmax()
            if token == eos { break }

            if let text = decoder.push(engine.tokenText(token)), !text.isEmpty {
                onToken(text)
            }
            produced += 1
            try eval(token)
        }

        if let tail = decoder.flush(), !tail.isEmpty {
            onToken(tail)
        }
        return produced
    }
}

/// Token text arrives as raw bytes that may split a multi-byte UTF-8 sequence
/// across tokens (the tokenizer is byte-level BPE). Buffer bytes and emit only
/// when the accumulated buffer is valid UTF-8.
struct IncrementalUTF8 {
    private var buffer: [UInt8] = []

    mutating func push(_ bytes: [UInt8]) -> String? {
        buffer.append(contentsOf: bytes)
        if let s = String(bytes: buffer, encoding: .utf8) {
            buffer.removeAll(keepingCapacity: true)
            return s
        }
        // Incomplete trailing multi-byte sequence: wait for more bytes.
        return nil
    }

    /// Emit whatever is left, replacing any invalid tail so nothing is lost.
    mutating func flush() -> String? {
        guard !buffer.isEmpty else { return nil }
        let s = String(decoding: buffer, as: UTF8.self)
        buffer.removeAll(keepingCapacity: true)
        return s
    }
}

/// Decode a NUL-terminated C error buffer into a Swift String.
func errorString(_ buffer: [CChar]) -> String {
    let bytes = buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
    return String(decoding: bytes, as: UTF8.self)
}

/// Call `body` with a C string for `value`, or NULL when `value` is nil.
func withOptionalCString<R>(_ value: String?, _ body: (UnsafePointer<CChar>?) -> R) -> R {
    if let value {
        return value.withCString { body($0) }
    }
    return body(nil)
}
