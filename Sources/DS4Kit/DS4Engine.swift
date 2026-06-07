import CDS4
import Foundation

/// Which compute backend the engine should use. Metal is the only production
/// path on macOS; CPU exists for reference/debug and is slow.
public enum DS4Backend: Sendable {
    case metal
    case cpu

    var c: ds4_backend {
        switch self {
        case .metal: return DS4_BACKEND_METAL
        case .cpu: return DS4_BACKEND_CPU
        }
    }
}

/// DeepSeek V4 thinking modes. `.none` is a direct answer; `.high` is normal
/// thinking; `.max` (Think Max) needs a large enough context or it falls back.
public enum DS4ThinkMode: Sendable {
    case none
    case high
    case max

    var c: ds4_think_mode {
        switch self {
        case .none: return DS4_THINK_NONE
        case .high: return DS4_THINK_HIGH
        case .max: return DS4_THINK_MAX
        }
    }
}

/// SSD streaming of routed MoE experts: keeps non-routed weights resident and
/// streams routed experts from the GGUF through an in-memory cache. Use it to
/// run a model larger than RAM. `cacheSpec` accepts the engine's own syntax,
/// e.g. "32GB" (byte budget) or a plain expert count like "4854"; nil = auto.
public struct StreamingOptions: Sendable {
    public var enabled: Bool
    public var cacheSpec: String?

    public init(enabled: Bool, cacheSpec: String? = nil) {
        self.enabled = enabled
        self.cacheSpec = cacheSpec
    }

    public static let disabled = StreamingOptions(enabled: false)
}

/// Estimated context (KV cache + scratch) memory for a given context size.
public struct ContextMemory: Sendable {
    public let totalBytes: UInt64
    public let rawBytes: UInt64
    public let compressedBytes: UInt64
    public let scratchBytes: UInt64
}

public enum DS4Error: Error, CustomStringConvertible {
    case engineOpenFailed(Int32)
    case sessionCreateFailed(Int32)
    case syncFailed(String)
    case evalFailed(String)

    public var description: String {
        switch self {
        case .engineOpenFailed(let rc): return "ds4_engine_open failed (rc=\(rc))"
        case .sessionCreateFailed(let rc): return "ds4_session_create failed (rc=\(rc))"
        case .syncFailed(let m): return "session sync failed: \(m)"
        case .evalFailed(let m): return "token eval failed: \(m)"
        }
    }
}

/// A loaded model. Wraps the C `ds4_engine`. One engine maps the GGUF and owns
/// all global Metal state, so there is exactly one engine per process.
///
/// Not yet concurrency-safe: in Phase 1 this becomes an `actor` that serializes
/// every engine/session call onto a single inference task, matching the C
/// engine's single-graph-worker model. For now use it from one thread.
public final class DS4Engine {
    /// Opaque `ds4_engine *`.
    let handle: OpaquePointer
    /// Kept alive for the engine's lifetime: the engine may reference the model
    /// path after open (e.g. SSD streaming reopens the GGUF).
    private let modelPathC: UnsafeMutablePointer<CChar>

    /// Open a GGUF model.
    ///
    /// - Parameters:
    ///   - modelPath: path to the ds4-compatible GGUF (e.g. ./ds4flash.gguf).
    ///   - backend: compute backend, defaults to Metal.
    ///   - metalSourceDir: directory containing the metal/*.metal kernel sources.
    ///     Required for a bundled app; pass the parent project's `metal/` folder
    ///     when running from a working directory that does not contain it.
    public init(modelPath: String,
                backend: DS4Backend = .metal,
                metalSourceDir: String? = nil,
                streaming: StreamingOptions = .disabled,
                minimumRAMMode: Bool = false) throws {
        if let dir = metalSourceDir {
            ds4gui_set_metal_source_dir(dir)
        }
        // Tier-A per-layer streaming: ask the engine not to pin or warm up the
        // model. Has effect only when set before ds4_engine_open(), which is
        // why we do it here.
        ds4gui_set_minimum_ram_mode(minimumRAMMode ? 1 : 0)

        self.modelPathC = strdup(modelPath)

        var opt = ds4_engine_options()  // zero-initialized by the importer
        opt.model_path = UnsafePointer(modelPathC)
        opt.backend = backend.c
        opt.power_percent = 100
        opt.n_threads = 0  // engine default

        opt.ssd_streaming = streaming.enabled
        if streaming.enabled, let spec = streaming.cacheSpec, !spec.isEmpty {
            var experts: UInt32 = 0
            var bytes: UInt64 = 0
            let ok = spec.withCString {
                ds4_parse_streaming_cache_experts_arg($0, &experts, &bytes)
            }
            if ok {
                opt.ssd_streaming_cache_experts = experts
                opt.ssd_streaming_cache_bytes = bytes
            }
        }

        var enginePtr: OpaquePointer?
        let rc = autoreleasepool { ds4_engine_open(&enginePtr, &opt) }
        guard rc == 0, let e = enginePtr else {
            free(modelPathC)
            throw DS4Error.engineOpenFailed(rc)
        }
        self.handle = e
    }

    deinit {
        ds4_engine_close(handle)
        free(modelPathC)
    }

    public var modelName: String {
        guard let c = ds4_engine_model_name(handle) else { return "unknown" }
        return String(cString: c)
    }

    public var vocabSize: Int { Int(ds4_engine_vocab_size(handle)) }
    public var layerCount: Int { Int(ds4_engine_layer_count(handle)) }
    public var routedQuantBits: Int { Int(ds4_engine_routed_quant_bits(handle)) }

    public var eosToken: Int32 { ds4_token_eos(handle) }

    /// Number of FP32 values in one position's hidden-state tensor (HC × n_embd).
    /// Used to size the buffers passed to layer-slice evaluation.
    public var hiddenF32Values: Int { Int(ds4_engine_hidden_f32_values(handle)) }

    /// Hint the OS to release the memory pages that hold layer K's weights.
    /// Used by per-layer streaming between successive layer-slice evaluations.
    /// Returns the number of tensors hinted, or -1 on error.
    @discardableResult
    public func adviseLayerDontNeed(_ layer: Int) -> Int32 {
        ds4_engine_advise_layer_dontneed(handle, UInt32(layer))
    }

    /// In SSD streaming mode the engine only maps the token embedding at open
    /// time; layer weights are mapped lazily. The per-layer Swift driver must
    /// call this before `evalLayerSlice(K, K)` or Metal will hit a
    /// "model range not covered by mapped model views" error.
    @discardableResult
    public func streamMapLayer(_ layer: Int) -> Bool {
        ds4_engine_stream_map_layer(handle, UInt32(layer)) != 0
    }

    /// Estimate KV cache + scratch memory for a context size. Valid only after
    /// the GGUF is open (it uses the active Flash/Pro model shape).
    public func contextMemoryEstimate(ctxSize: Int) -> ContextMemory {
        let m = ds4_context_memory_estimate(DS4_BACKEND_METAL, Int32(ctxSize))
        return ContextMemory(totalBytes: m.total_bytes,
                             rawBytes: m.raw_bytes,
                             compressedBytes: m.compressed_bytes,
                             scratchBytes: m.scratch_bytes)
    }

    public func tokenText(_ token: Int32) -> [UInt8] {
        var len: size_t = 0
        guard let ptr = ds4_token_text(handle, token, &len), len > 0 else { return [] }
        return Array(UnsafeBufferPointer(start: UnsafeRawPointer(ptr).assumingMemoryBound(to: UInt8.self),
                                         count: Int(len)))
    }

    /// Render a full conversation plus the assistant generation prefix into a
    /// token buffer, using the engine's native chat formatting. Rebuilding the
    /// whole transcript each turn keeps the rendered prefix stable, so
    /// `ds4_session_sync` reuses the live KV cache for the unchanged prefix.
    func renderConversation(_ messages: [ChatMessage], thinkMode: DS4ThinkMode) -> Tokens {
        let tokens = Tokens()
        ds4_chat_begin(handle, &tokens.raw)
        for message in messages {
            message.content.withCString { contentC in
                message.role.rawValue.withCString { roleC in
                    ds4_chat_append_message(handle, &tokens.raw, roleC, contentC)
                }
            }
        }
        ds4_chat_append_assistant_prefix(handle, &tokens.raw, thinkMode.c)
        return tokens
    }
}
