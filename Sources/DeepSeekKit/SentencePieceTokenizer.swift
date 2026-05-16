import Foundation

/// Minimal SentencePiece tokenizer (unigram + byte-fallback). Parses
/// the `.model` protobuf file ad-hoc — only the tags we need are
/// decoded, the rest is skipped, so we can ship without a dependency
/// on `swift-protobuf`.
///
/// Algorithm: Viterbi decode over the unigram lattice using piece
/// log-probabilities. Whitespace is normalised to `▁` before lookup.
/// Tokens not in the vocab fall back to `<0xNN>` byte tokens (typed
/// `BYTE = 6` in the proto) — every SentencePiece model that ships
/// with HuggingFace LLMs (Mistral / Llama / Gemma / Qwen) includes
/// the 256-byte BYTE table by default.
public final class SentencePieceTokenizer: Tokenizer {
    private struct Piece {
        let id: Int
        let text: String
        let score: Float
        let type: Int           // 1 normal, 2 unknown, 3 control, 4 user-defined, 6 byte
    }

    private let pieces: [Piece]
    private let textToId: [String: Int]
    private let unkId: Int
    private let byteFallback: [Int]   // id of <0xNN> for each NN, or -1 if absent

    public let bosId: Int?
    public let eosId: Int?
    public let stopTokenIds: Set<Int>

    public init(modelBytes: Data) throws {
        var parser = ProtobufReader(data: modelBytes)
        var pieces: [Piece] = []
        // Top-level ModelProto: field 1 = pieces (repeated SentencePiece,
        // length-delimited). We skip everything else (trainer_spec,
        // normalizer_spec, …).
        while parser.hasMore {
            let tag = try parser.readVarint()
            let fieldNumber = Int(tag >> 3)
            let wireType = Int(tag & 0x7)
            if fieldNumber == 1, wireType == 2 {
                let pieceLen = Int(try parser.readVarint())
                let pieceData = try parser.readSubdata(length: pieceLen)
                let piece = try SentencePieceTokenizer.parsePiece(pieceData, idHint: pieces.count)
                pieces.append(piece)
            } else {
                try parser.skipField(wireType: wireType)
            }
        }
        self.pieces = pieces
        var dict: [String: Int] = [:]
        dict.reserveCapacity(pieces.count)
        for p in pieces { dict[p.text] = p.id }
        self.textToId = dict
        self.unkId = pieces.firstIndex(where: { $0.type == 2 }) ?? 0
        var bytes = [Int](repeating: -1, count: 256)
        for p in pieces where p.type == 6 {
            // BYTE pieces follow the `<0xNN>` convention.
            if p.text.count == 6, p.text.hasPrefix("<0x"), p.text.hasSuffix(">"),
               let v = UInt8(p.text.dropFirst(3).dropLast(), radix: 16) {
                bytes[Int(v)] = p.id
            }
        }
        self.byteFallback = bytes
        self.bosId = dict["<s>"]
        self.eosId = dict["</s>"]
        var stops = Set<Int>()
        if let e = self.eosId { stops.insert(e) }
        self.stopTokenIds = stops
    }

    public func encode(_ text: String) -> [Int] {
        // SentencePiece normalises spaces by prepending `▁` (U+2581) to
        // the start and replacing internal spaces with `▁`.
        let normalised = "\u{2581}" + text.replacingOccurrences(of: " ", with: "\u{2581}")
        return viterbi(normalised)
    }

    public func decode(_ ids: [Int]) -> String {
        var bytesBuf: [UInt8] = []
        var out = ""
        func flushBytes() {
            if !bytesBuf.isEmpty {
                out += String(decoding: bytesBuf, as: UTF8.self)
                bytesBuf.removeAll(keepingCapacity: true)
            }
        }
        for id in ids {
            guard id >= 0, id < pieces.count else { continue }
            let p = pieces[id]
            if p.type == 6 {
                if p.text.hasPrefix("<0x"), let v = UInt8(p.text.dropFirst(3).dropLast(), radix: 16) {
                    bytesBuf.append(v)
                }
            } else {
                flushBytes()
                out += p.text.replacingOccurrences(of: "\u{2581}", with: " ")
            }
        }
        flushBytes()
        // SentencePiece convention: leading `▁` (now space) is the
        // synthetic word boundary, drop it from the final string.
        if out.first == " " { out.removeFirst() }
        return out
    }

    // MARK: - Viterbi

    private func viterbi(_ text: String) -> [Int] {
        let scalars = Array(text.unicodeScalars)
        let n = scalars.count
        if n == 0 { return [] }
        // dp[i] = best log-prob ending at position i; back[i] = (start, pieceId)
        var dp = [Float](repeating: -.infinity, count: n + 1)
        dp[0] = 0
        var back = [(start: Int, pieceId: Int)?](repeating: nil, count: n + 1)
        for end in 1...n {
            // Try every possible start. O(N²); fine for chat prompts.
            for start in stride(from: end - 1, through: max(0, end - 32), by: -1) {
                let substring = String(String.UnicodeScalarView(scalars[start..<end]))
                if let id = textToId[substring] {
                    let cand = dp[start] + pieces[id].score
                    if cand > dp[end] {
                        dp[end] = cand
                        back[end] = (start, id)
                    }
                }
            }
            // Byte-fallback: always available for a single-codepoint span
            // that maps to UTF-8 bytes.
            if back[end] == nil {
                let scalar = scalars[end - 1]
                let utf8: [UInt8] = Array(String(scalar).utf8)
                if !utf8.isEmpty, utf8.allSatisfy({ byteFallback[Int($0)] >= 0 }) {
                    // Use the first byte's id as the back-pointer; the
                    // remaining bytes are emitted in the trace-back below.
                    let cand = dp[end - 1] - 10.0  // arbitrary low score
                    if cand > dp[end] {
                        dp[end] = cand
                        // Store start = end-1, id = -1 sentinel meaning
                        // "byte-fallback this single codepoint".
                        back[end] = (end - 1, -1)
                    }
                }
            }
            // Last resort: <unk>.
            if back[end] == nil {
                dp[end] = dp[end - 1] - 20.0
                back[end] = (end - 1, unkId)
            }
        }
        // Trace-back.
        var ids: [Int] = []
        var cur = n
        while cur > 0, let step = back[cur] {
            if step.pieceId == -1 {
                let utf8 = Array(String(scalars[step.start]).utf8)
                for byte in utf8.reversed() {
                    ids.append(byteFallback[Int(byte)])
                }
            } else {
                ids.append(step.pieceId)
            }
            cur = step.start
        }
        return ids.reversed()
    }

    // MARK: - Protobuf inline

    private static func parsePiece(_ data: Data, idHint: Int) throws -> Piece {
        var reader = ProtobufReader(data: data)
        var text = ""
        var score: Float = 0
        var type = 1
        while reader.hasMore {
            let tag = try reader.readVarint()
            let fieldNumber = Int(tag >> 3)
            let wireType = Int(tag & 0x7)
            switch (fieldNumber, wireType) {
            case (1, 2):
                let len = Int(try reader.readVarint())
                let sub = try reader.readSubdata(length: len)
                text = String(decoding: sub, as: UTF8.self)
            case (2, 5):
                score = Float(bitPattern: try reader.readFixed32())
            case (3, 0):
                type = Int(try reader.readVarint())
            default:
                try reader.skipField(wireType: wireType)
            }
        }
        return Piece(id: idHint, text: text, score: score, type: type)
    }
}

// MARK: - Tiny protobuf reader

private struct ProtobufReader {
    let data: Data
    var idx: Int = 0
    var hasMore: Bool { idx < data.count }

    init(data: Data) { self.data = data }

    mutating func readVarint() throws -> UInt64 {
        var value: UInt64 = 0
        var shift: UInt64 = 0
        while idx < data.count {
            let b = data[idx]
            idx += 1
            value |= UInt64(b & 0x7F) << shift
            if (b & 0x80) == 0 { return value }
            shift += 7
            if shift > 63 {
                throw ChatTemplateError.parseFailure("protobuf: varint overflow")
            }
        }
        throw ChatTemplateError.parseFailure("protobuf: varint truncated")
    }

    mutating func readFixed32() throws -> UInt32 {
        guard idx + 4 <= data.count else {
            throw ChatTemplateError.parseFailure("protobuf: fixed32 truncated")
        }
        var out: UInt32 = 0
        for i in 0..<4 {
            out |= UInt32(data[idx + i]) << UInt32(i * 8)
        }
        idx += 4
        return out
    }

    mutating func readSubdata(length: Int) throws -> Data {
        guard idx + length <= data.count else {
            throw ChatTemplateError.parseFailure("protobuf: length-delimited truncated")
        }
        let slice = data.subdata(in: idx..<(idx + length))
        idx += length
        return slice
    }

    mutating func skipField(wireType: Int) throws {
        switch wireType {
        case 0: _ = try readVarint()
        case 1:
            guard idx + 8 <= data.count else {
                throw ChatTemplateError.parseFailure("protobuf: fixed64 truncated")
            }
            idx += 8
        case 2:
            let len = Int(try readVarint())
            guard idx + len <= data.count else {
                throw ChatTemplateError.parseFailure("protobuf: length-delimited truncated")
            }
            idx += len
        case 5: _ = try readFixed32()
        default:
            throw ChatTemplateError.parseFailure("protobuf: unsupported wire type \(wireType)")
        }
    }
}
