import Foundation

/// Native Qwen3.8 Flash Next (`qwen4exp`) text trunk. Shapes and hash arithmetic
/// follow GiorgioOppo/ds4 acf5c16bb6a01f8c4f027a0db265ae270d8907ab.
public struct Qwen38Configuration: Sendable {
    public let embedding, layers, vocabulary, heads, kvHeads, headDimension, rotaryDimension: Int
    public let experts, expertsUsed, expertWidth, indexHeads, indexDimension, indexTopK: Int
    public let hcCount, hcRank, linearKHeads, linearVHeads, linearDimension: Int
    public let nativeContext, pleHeadDimension: Int
    public let epsilon: Float = 1e-6
    public let ropeBase: Float = 10_000_000
    public let pleEOS: Int = 248_044
    public let pleMultipliers: [UInt64]
    public let pleOffsets, pleVocabulary: [UInt32]
    public var hcDimension: Int { embedding * hcCount }
    public var queryDimension: Int { heads * headDimension }
    public var linearValueDimension: Int { linearVHeads * linearDimension }
    public var linearQKVDimension: Int { (2 * linearKHeads + linearVHeads) * linearDimension }
    public var ngramRows: UInt64 { zip(pleOffsets, pleVocabulary).map { UInt64($0) + UInt64($1) }.max()! }
    public func isLinear(layer: Int) -> Bool { (layer + 1) % 4 != 0 }

    public init(model: GGUFModel) throws {
        func fail(_ text: String) -> Error { Qwen38Error.invalidModel(text) }
        guard model.string("general.architecture") == "qwen4exp" else { throw fail("Expected qwen4exp architecture") }
        func integer(_ key: String) throws -> Int {
            guard let v = model.u64Compat("qwen4exp." + key), v <= UInt64(Int32.max) else {
                throw fail("Missing or invalid qwen4exp.\(key)")
            }
            return Int(v)
        }
        let e = try integer("embedding_length")
        guard e == 2560 || e == 64 else { throw fail("Unsupported embedding length \(e)") }
        let mini = e == 64
        embedding = e; layers = mini ? 8 : 48; vocabulary = 248_320
        heads = mini ? 4 : 24; kvHeads = 2; headDimension = mini ? 32 : 256
        rotaryDimension = mini ? 8 : 64; experts = mini ? 32 : 512; expertsUsed = 10
        expertWidth = mini ? 32 : 640; indexHeads = 4; indexDimension = mini ? 32 : 128
        indexTopK = mini ? 8 : 2048; hcCount = 4; hcRank = mini ? 8 : 320
        linearKHeads = mini ? 2 : 16; linearVHeads = mini ? 6 : 48; linearDimension = mini ? 32 : 128
        pleHeadDimension = mini ? 4 : 160
        nativeContext = try integer("context_length")
        guard nativeContext > 0, nativeContext <= 262_144 else { throw fail("Unsupported native context") }
        let mtpRaw = model.u64Compat("qwen4exp.nextn_predict_layers") ?? 0
        guard mtpRaw <= 1 else { throw fail("At most one optional MTP layer is supported") }
        let mtp = Int(mtpRaw)
        let expected: [(String, Int)] = [
            ("block_count", layers + mtp), ("attention.head_count", heads), ("attention.head_count_kv", kvHeads),
            ("attention.key_length", headDimension), ("attention.value_length", headDimension),
            ("rope.dimension_count", rotaryDimension), ("expert_count", experts), ("expert_used_count", expertsUsed),
            ("expert_feed_forward_length", expertWidth), ("expert_shared_feed_forward_length", expertWidth),
            ("ssm.conv_kernel", 4), ("ssm.state_size", linearDimension), ("ssm.group_count", linearKHeads),
            ("ssm.time_step_rank", linearVHeads), ("ssm.inner_size", linearVHeads * linearDimension),
            ("full_attention_interval", 4), ("hyper_connection.count", hcCount), ("hyper_connection.low_rank", hcRank),
            ("attention.indexer.head_count", indexHeads), ("attention.indexer.key_length", indexDimension),
            ("attention.indexer.top_k", indexTopK), ("ple.ngram_size", 3), ("ple.heads_per_ngram", 8),
            ("ple.conv_kernel", 4), ("ple.eos_token_id", pleEOS), ("embedding_length_per_layer_input", pleHeadDimension)
        ]
        for (key, value) in expected where try integer(key) != value { throw fail("Unsupported qwen4exp.\(key); expected \(value)") }
        guard model.f32Compat("qwen4exp.rope.freq_base") == ropeBase,
              let eps = model.f32Compat("qwen4exp.attention.layer_norm_rms_epsilon"), abs(eps - epsilon) < 1e-10 else {
            throw fail("Unsupported rotary base or RMS epsilon")
        }
        let ratios = try Self.integerArray(model, "attention.compress_ratios", maximum: 49)
        guard ratios.count >= layers + mtp else { throw fail("Missing compression ratios") }
        for i in 0..<(layers + mtp) where ratios[i] != ((i < layers && (i + 1) % 4 != 0) ? 0 : 4) {
            throw fail("Invalid compression ratio at layer \(i)")
        }
        guard try Self.integerArray(model, "ple.layers", maximum: 1) == [1] else { throw fail("PLE must be on layer 1") }
        pleMultipliers = try Self.integerArray(model, "ple.layer_multipliers", maximum: 3)
        let offsets = try Self.integerArray(model, "ple.head_offsets", maximum: 16)
        let vocabs = try Self.integerArray(model, "ple.head_vocab_sizes", maximum: 16)
        guard pleMultipliers.count == 3, offsets.count == 16, vocabs.count == 16,
              offsets.allSatisfy({ $0 <= UInt32.max }), vocabs.allSatisfy({ $0 > 0 && $0 <= UInt32.max }),
              zip(offsets, vocabs).allSatisfy({ $0 + $1 <= UInt64(UInt32.max) + 1 }) else { throw fail("Invalid PLE hash tables") }
        pleOffsets = offsets.map(UInt32.init); pleVocabulary = vocabs.map(UInt32.init)
    }

    private static func integerArray(_ model: GGUFModel, _ key: String, maximum: Int) throws -> [UInt64] {
        guard let a = model.array("qwen4exp." + key), a.len <= UInt64(maximum), [4, 5, 10, 11].contains(a.type) else {
            throw Qwen38Error.invalidModel("Missing or invalid qwen4exp.\(key) array")
        }
        let width = a.type >= 10 ? 8 : 4
        guard a.dataPos <= model.size, a.len * UInt64(width) <= model.size - a.dataPos else {
            throw Qwen38Error.invalidModel("Truncated qwen4exp.\(key)")
        }
        let p = model.mapBase + Int(a.dataPos)
        return (0..<Int(a.len)).map { i in
            width == 8 ? UInt64(littleEndian: p.loadUnaligned(fromByteOffset: i * 8, as: UInt64.self))
                : UInt64(UInt32(littleEndian: p.loadUnaligned(fromByteOffset: i * 4, as: UInt32.self)))
        }
    }
}

public enum Qwen38Error: Error, CustomStringConvertible {
    case invalidModel(String), io(String)
    public var description: String { switch self { case .invalidModel(let s): "Qwen3.8: \(s)"; case .io(let s): "Qwen3.8 I/O: \(s)" } }
}

/// The unsigned wraparound hash is part of the model, including EOS truncation.
public struct Qwen38NgramState: Sendable {
    public private(set) var previous: [Int]
    public init(eos: Int = 248_044) { previous = [eos, eos] }
    public mutating func rows(token: Int, multipliers: [UInt64], offsets: [UInt32], vocabularies: [UInt32], eos: Int = 248_044) -> [UInt32] {
        precondition(token >= 0 && multipliers.count == 3 && offsets.count == 16 && vocabularies.count == 16)
        var context = [token, eos, eos], cut = false
        for j in 1..<3 { let t = cut ? eos : previous[j - 1]; cut = cut || t == eos; context[j] = cut ? eos : t }
        var result = [UInt32](); result.reserveCapacity(16)
        for n in 2...3 {
            var hash = UInt64(context[0]) &* multipliers[0]
            for j in 1..<n { hash ^= UInt64(context[j]) &* multipliers[j] }
            for h in ((n - 2) * 8)..<((n - 1) * 8) {
                result.append(UInt32(hash % UInt64(vocabularies[h])) &+ offsets[h])
            }
        }
        previous = [token, previous[0]]
        return result
    }
}
