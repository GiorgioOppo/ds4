import SwiftUI
import DS4Engine
import DS4Core

// MARK: - Markdown rendering

/// Lightweight Markdown renderer for assistant messages: fenced code blocks,
/// headings, bullet/ordered lists, blockquotes, and paragraphs with inline
/// markdown (bold/italic/`code`/links). The entire answer is emitted as one
/// attributed `Text`, so AppKit can keep a selection across paragraph and block
/// boundaries instead of limiting it to one rendered Markdown block.
struct MarkdownView: View {
    /// Parsed and attributed ONCE per text value (in init): SwiftUI can re-evaluate `body`
    /// several times per update, and during streaming the view is recreated
    /// at every token — parsing there made the per-token UI cost grow with
    /// the message length.
    private let attributedText: AttributedString

    init(text: String) {
        self.attributedText = Self.render(Self.parse(text))
    }

    var body: some View {
        Text(attributedText)
            .textSelection(.enabled)
            .fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// Build one text storage for the full response. Besides enabling continuous
    /// selection, this retains the inline Markdown attributes while applying the
    /// block-level appearance that SwiftUI's `Text` does not infer by itself.
    private static func render(_ blocks: [Block]) -> AttributedString {
        var result = AttributedString()

        for (blockIndex, block) in blocks.enumerated() {
            if blockIndex > 0 { result += AttributedString("\n\n") }

            switch block {
            case .code(let code):
                var value = AttributedString(code)
                value.font = .system(.callout, design: .monospaced)
                value.backgroundColor = Color.black.opacity(0.06)
                result += value

            case .heading(let level, let string):
                var value = inline(string)
                value.font = level <= 1
                    ? .title3.bold()
                    : (level == 2 ? .headline : .subheadline.bold())
                result += value

            case .list(let items, let ordered):
                for (itemIndex, item) in items.enumerated() {
                    if itemIndex > 0 { result += AttributedString("\n") }
                    var marker = AttributedString(ordered ? "\(itemIndex + 1). " : "• ")
                    marker.font = .callout.monospacedDigit()
                    marker.foregroundColor = .secondary
                    result += marker
                    result += inline(item)
                }

            case .quote(let string):
                var marker = AttributedString("│ ")
                marker.foregroundColor = .secondary.opacity(0.5)
                var value = inline(string)
                value.foregroundColor = .secondary
                result += marker
                result += value

            case .paragraph(let string):
                result += inline(string)
            }
        }

        return result
    }

    /// Inline markdown (bold/italic/code/links), whitespace preserved, never throws.
    static func inline(_ s: String) -> AttributedString {
        (try? AttributedString(markdown: s, options: .init(
            interpretedSyntax: .inlineOnlyPreservingWhitespace,
            failurePolicy: .returnPartiallyParsedIfPossible))) ?? AttributedString(s)
    }

    enum Block { case code(String), heading(Int, String), list([String], ordered: Bool)
                 case quote(String), paragraph(String) }

    /// Split text into block-level elements (the part SwiftUI's Text won't do).
    static func parse(_ text: String) -> [Block] {
        var blocks: [Block] = []
        let lines = text.components(separatedBy: "\n")
        var para: [String] = []
        func flush() {
            if !para.isEmpty { blocks.append(.paragraph(para.joined(separator: "\n"))); para = [] }
        }
        var i = 0
        while i < lines.count {
            let line = lines[i]
            let t = line.trimmingCharacters(in: .whitespaces)

            if t.hasPrefix("```") {                         // fenced code block
                flush()
                var code: [String] = []
                i += 1
                while i < lines.count, !lines[i].trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                    code.append(lines[i]); i += 1
                }
                i += 1                                       // skip closing fence
                blocks.append(.code(code.joined(separator: "\n")))
                continue
            }
            if let h = headingLevel(t) {                     // # heading
                flush()
                let content = String(t.drop(while: { $0 == "#" })).trimmingCharacters(in: .whitespaces)
                blocks.append(.heading(h, content)); i += 1; continue
            }
            if isBullet(t) || isOrdered(t) {                 // list (grouped)
                flush()
                let ordered = isOrdered(t)
                var items: [String] = []
                while i < lines.count {
                    let tt = lines[i].trimmingCharacters(in: .whitespaces)
                    if ordered, isOrdered(tt) { items.append(stripOrdered(tt)) }
                    else if !ordered, isBullet(tt) { items.append(String(tt.dropFirst(2))) }
                    else { break }
                    i += 1
                }
                blocks.append(.list(items, ordered: ordered)); continue
            }
            if t.hasPrefix("> ") {                           // blockquote
                flush(); blocks.append(.quote(String(t.dropFirst(2)))); i += 1; continue
            }
            if t.isEmpty { flush() } else { para.append(line) }
            i += 1
        }
        flush()
        return blocks
    }

    private static func headingLevel(_ t: String) -> Int? {
        var n = 0
        for c in t { if c == "#" { n += 1 } else { break } }
        return (n >= 1 && n <= 6 && t.dropFirst(n).first == " ") ? n : nil
    }
    private static func isBullet(_ t: String) -> Bool {
        t.hasPrefix("- ") || t.hasPrefix("* ") || t.hasPrefix("+ ")
    }
    private static func isOrdered(_ t: String) -> Bool {
        guard let dot = t.firstIndex(of: ".") else { return false }
        let num = t[t.startIndex..<dot]
        return !num.isEmpty && num.allSatisfy(\.isNumber) && t[dot...].hasPrefix(". ")
    }
    private static func stripOrdered(_ t: String) -> String {
        guard let dot = t.firstIndex(of: ".") else { return t }
        return String(t[t.index(after: dot)...]).trimmingCharacters(in: .whitespaces)
    }
}
