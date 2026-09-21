import Foundation

/// Native GPT-2 byte BPE for the two explicitly supported Qwen35-pretokenized
/// checkpoints. Ordinary user text never uses the rendered-control scanner.
public final class QwenTokenizer: TokenizerProtocol {
    public let architecture: ModelArchitectureID
    public let nVocab: Int
    public let tokens: [[UInt8]]
    public let messageStart, messageEnd, endOfText, thinkOpen, thinkClose: Int32
    public var stopTokens: Set<Int32> { [messageEnd, endOfText] }
    private let tokenToID: [[UInt8]: Int32]
    private let mergeRank: [[UInt8]: Int32]
    private let controls: Set<Int32>
    private let buckets: [[(bytes: [UInt8], id: Int32)]]

    public convenience init(model: GGUFModel) throws {
        let architecture = try ModelArchitectureDetector.detect(in: model).id
        guard architecture == .bonsai2 || architecture == .qwen38FlashNext else {
            throw ModelArchitectureError.unsupportedArchitecture(architecture)
        }
        guard model.string("tokenizer.ggml.model") == "gpt2",
              model.string("tokenizer.ggml.pre") == "qwen35",
              let tokens = model.stringArrayBytes("tokenizer.ggml.tokens"),
              let merges = model.stringArrayBytes("tokenizer.ggml.merges") else {
            throw GGUFError.message("Qwen richiede vocabolario GPT-2, merge e pre-tokenizer qwen35.")
        }
        try self.init(architecture: architecture, tokens: tokens, merges: merges,
                      types: model.intArray("tokenizer.ggml.token_type"))
    }

    init(architecture: ModelArchitectureID, tokens: [[UInt8]], merges: [[UInt8]],
         types: [Int64]? = nil) throws {
        guard !tokens.isEmpty, tokens.count <= Int(Int32.max) else {
            throw GGUFError.message("Vocabolario Qwen vuoto o troppo grande.")
        }
        self.architecture = architecture; self.tokens = tokens; nVocab = tokens.count
        var ids: [[UInt8]: Int32] = [:], ranks: [[UInt8]: Int32] = [:]
        for (index, token) in tokens.enumerated() { ids[token] = Int32(index) }
        for (index, merge) in merges.enumerated() { ranks[merge] = Int32(index) }
        tokenToID = ids; mergeRank = ranks
        func require(_ literal: String) throws -> Int32 {
            guard let id = ids[Array(literal.utf8)] else {
                throw GGUFError.message("Token Qwen obbligatorio assente: \(literal)")
            }
            return id
        }
        messageStart = try require("<|im_start|>")
        messageEnd = try require("<|im_end|>")
        endOfText = try require("<|endoftext|>")
        thinkOpen = try require("<think>"); thinkClose = try require("</think>")
        var special: Set<Int32> = [messageStart, messageEnd, endOfText, thinkOpen, thinkClose]
        if let types, types.count == tokens.count {
            for (index, type) in types.enumerated() where type == 3 || type == 4 {
                special.insert(Int32(index))
            }
        }
        controls = special
        var index = [[(bytes: [UInt8], id: Int32)]](repeating: [], count: 256)
        for id in special {
            let bytes = tokens[Int(id)]
            if let first = bytes.first { index[Int(first)].append((bytes, id)) }
        }
        for bucket in index.indices { index[bucket].sort { $0.bytes.count > $1.bytes.count } }
        buckets = index
    }

    public func tokenID(_ literal: String) -> Int32? { tokenToID[Array(literal.utf8)] }
    public func tokenize(_ text: String) -> [Int32] { tokenizeBytes(Array(text.utf8)) }
    private func tokenizeBytes(_ bytes: [UInt8]) -> [Int32] {
        var output: [Int32] = []
        for piece in QwenPretokenizer.split(bytes) { emit(piece, into: &output) }
        return output
    }
    public func tokenizeRenderedChat(_ text: String) -> [Int32] {
        let bytes = Array(text.utf8)
        var output: [Int32] = [], position = 0, start = 0
        while position < bytes.count {
            guard let match = buckets[Int(bytes[position])].first(where: {
                position + $0.bytes.count <= bytes.count
                    && bytes[position..<(position + $0.bytes.count)].elementsEqual($0.bytes)
            }) else { position += 1; continue }
            output += tokenizeBytes(Array(bytes[start..<position]))
            output.append(match.id); position += match.bytes.count; start = position
        }
        output += tokenizeBytes(Array(bytes[start..<bytes.count]))
        return output
    }
    public func tokenText(_ id: Int32) -> [UInt8] {
        guard id >= 0, Int(id) < nVocab else { return [] }
        let token = tokens[Int(id)]
        if controls.contains(id) { return token }
        var result: [UInt8] = [], position = 0
        while position < token.count {
            let decoded = ByteLevel.decodeOne(token, token.count, position)
            position = decoded.next
            if let byte = ByteLevel.codepointToByte(decoded.cp) { result.append(byte) }
        }
        return result
    }

    private func emit(_ piece: [UInt8], into output: inout [Int32]) {
        let encoded = ByteLevel.byteEncode(piece[...])
        var symbols: [[UInt8]] = [], position = 0
        while position < encoded.count {
            let count = min(ByteLevel.utf8Len(encoded[position]), encoded.count - position)
            symbols.append(Array(encoded[position..<(position + count)])); position += count
        }
        while symbols.count > 1 {
            var best = -1, rank = Int32.max
            for index in 0..<(symbols.count - 1) {
                let key = symbols[index] + [32] + symbols[index + 1]
                if let candidate = mergeRank[key], candidate < rank { best = index; rank = candidate }
            }
            guard best >= 0 else { break }
            symbols[best] += symbols[best + 1]; symbols.remove(at: best + 1)
        }
        for symbol in symbols {
            if let id = tokenToID[symbol] { output.append(id) }
            else { for byte in symbol { if let id = tokenToID[[byte]] { output.append(id) } } }
        }
    }
}
