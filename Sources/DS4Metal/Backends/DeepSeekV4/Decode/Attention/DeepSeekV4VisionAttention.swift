import Foundation

/// The language model's visual sentinels are outside its text vocabulary.
/// Mirrors ds4_image.c's next_image_span and attention_bounds; padding before
/// image-start remains causal, while the image itself sees 383 rows left and
/// 384 right (and at least its ordinary text sliding window on the left).
enum DeepSeekV4VisionAttention {
    struct Span: Equatable {
        let block: Range<Int>
        let image: ClosedRange<Int>
    }

    static func spans(tokens: [Int], vocabularySize: Int) throws -> [Span] {
        guard vocabularySize > 0, vocabularySize <= Int.max - 4 else {
            throw MetalError.unsupported("Vision: vocabolario non valido")
        }
        var result: [Span] = []
        var i = 0
        while i < tokens.count {
            guard tokens[i] >= 0 else { throw MetalError.unsupported("Vision: token negativo") }
            if tokens[i] < vocabularySize { i += 1; continue }
            let block = i
            while i < tokens.count && tokens[i] == vocabularySize + 1 { i += 1 }
            guard i < tokens.count, tokens[i] == vocabularySize else {
                throw MetalError.unsupported("Vision: manca image-start o blocco immagine tagliato")
            }
            let imageStart = i
            i += 1
            while i < tokens.count, tokens[i] != vocabularySize + 4 {
                guard tokens[i] > vocabularySize, tokens[i] < vocabularySize + 4 else {
                    throw MetalError.unsupported("Vision: sequenza di sentinelle non valida")
                }
                i += 1
            }
            guard i < tokens.count else { throw MetalError.unsupported("Vision: manca image-end") }
            result.append(Span(block: block..<(i + 1), image: imageStart...i))
            i += 1
        }
        return result
    }

    static func bounds(query: Int, startPosition: Int, image: ClosedRange<Int>,
                       firstRawPosition: Int, lastRawPosition: Int, window: Int) -> ClosedRange<Int> {
        let position = startPosition + query
        var low = window > 0 ? max(0, position + 1 - window) : 0
        var high = position
        if image.contains(query) {
            let left = min(query - image.lowerBound, 383)
            let right = min(image.upperBound - query, 384)
            let back = window > 0 ? max(window - 1, left) : position
            low = max(0, position - back)
            high = position + right
        }
        return max(low, firstRawPosition)...min(high, lastRawPosition)
    }
}
