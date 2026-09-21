import Foundation

/// Qwen35's ordered pre-tokenizer, shared by Bonsai and Qwen3.8 Flash Next.
/// Unicode classes are pinned to the upstream generated tables, independent
/// of the operating system's ICU version. Numeric scalars remain singletons.
enum QwenPretokenizer {
    private struct CharacterInfo {
        let codepoint: UInt32
        let next: Int
        var letter: Bool { QwenUnicode.contains(codepoint, in: QwenUnicode.letter) }
        var mark: Bool { QwenUnicode.contains(codepoint, in: QwenUnicode.mark) }
        var number: Bool { QwenUnicode.contains(codepoint, in: QwenUnicode.number) }
        var space: Bool { QwenUnicode.contains(codepoint, in: QwenUnicode.space) }
        var newline: Bool { codepoint == 10 || codepoint == 13 }
        var word: Bool { letter || mark }
        var punctuation: Bool { !space && !word && !number }
    }

    static func split(_ text: String) -> [[UInt8]] { split(Array(text.utf8)) }

    static func split(_ bytes: [UInt8]) -> [[UInt8]] {
        func at(_ position: Int) -> CharacterInfo? {
            guard position < bytes.count else { return nil }
            let decoded = ByteLevel.decodeOne(bytes, bytes.count, position)
            return CharacterInfo(codepoint: decoded.cp, next: decoded.next)
        }
        func lower(_ value: UInt32) -> UInt32 {
            (65...90).contains(value) ? value + 32 : value
        }
        var result: [[UInt8]] = []
        var position = 0
        while let current = at(position) {
            let start = position
            if current.codepoint == 39, let next = at(current.next) {
                let first = lower(next.codepoint)
                if [115, 116, 109, 100].contains(first) {
                    position = next.next
                    result.append(Array(bytes[start..<position])); continue
                }
                if let second = at(next.next) {
                    let last = lower(second.codepoint)
                    if (first == 114 && last == 101) || (first == 118 && last == 101)
                        || (first == 108 && last == 108) {
                        position = second.next
                        result.append(Array(bytes[start..<position])); continue
                    }
                }
            }
            var wordEnd: Int?
            if current.word { wordEnd = current.next }
            else if !current.newline && !current.number,
                    let next = at(current.next), next.word { wordEnd = next.next }
            if let wordEnd {
                position = wordEnd
                while let next = at(position), next.word { position = next.next }
                result.append(Array(bytes[start..<position])); continue
            }
            if current.number {
                position = current.next
                result.append(Array(bytes[start..<position])); continue
            }
            let punctuation = current.codepoint == 32 ? at(current.next) : current
            if let punctuation, punctuation.punctuation {
                position = current.codepoint == 32 ? current.next : start
                while let next = at(position), next.punctuation { position = next.next }
                while let next = at(position), next.newline { position = next.next }
                result.append(Array(bytes[start..<position])); continue
            }
            if current.space {
                var end = position, lastNewline: Int?, lastStart = position, count = 0
                while let next = at(end), next.space {
                    lastStart = end
                    if next.newline { lastNewline = next.next }
                    end = next.next; count += 1
                }
                position = lastNewline ?? ((count > 1 && end < bytes.count) ? lastStart : end)
                result.append(Array(bytes[start..<position])); continue
            }
            position = current.next
            result.append(Array(bytes[start..<position]))
        }
        return result
    }
}
