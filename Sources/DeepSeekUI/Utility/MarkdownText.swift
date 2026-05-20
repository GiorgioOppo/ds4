import SwiftUI
#if canImport(AppKit)
import AppKit
#endif

/// Renders Markdown content into SwiftUI views with block-level support
/// (ATX/Setext headings, paragraphs, ordered and unordered lists,
/// blockquotes, fenced code blocks, horizontal rules, and GFM tables)
/// on top of inline formatting (bold, italic, inline code, links).
/// Used for finalized assistant messages once `.done` lands. In-flight
/// streaming text flows through `StreamingCaretText`, which renders
/// the stable prefix with this view and leaves only the trailing
/// in-progress block as plain text with a caret.
struct MarkdownText: View {
    let raw: String

    var body: some View {
        let blocks = MarkdownParser.parse(raw)
        VStack(alignment: .leading, spacing: 10) {
            ForEach(blocks.indices, id: \.self) { i in
                MarkdownBlockView(block: blocks[i])
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// MARK: - Block enum

/// One parsed Markdown block. Lists carry their items as inline strings
/// (rendered through `InlineMarkdownText`); blockquotes carry their
/// raw body and recursively re-parse it so nested blocks still render.
enum MarkdownBlock: Equatable {
    case heading(level: Int, text: String)
    case paragraph(String)
    case list(ordered: Bool, items: [String], startIndex: Int)
    case blockquote(String)
    case codeBlock(language: String, body: String)
    case horizontalRule
    case table(headers: [String], rows: [[String]])
}

// MARK: - Block view dispatch

private struct MarkdownBlockView: View {
    let block: MarkdownBlock

    var body: some View {
        switch block {
        case .heading(let level, let text):
            HeadingView(level: level, text: text)
        case .paragraph(let text):
            ProseView(text: text)
        case .list(let ordered, let items, let startIndex):
            ListView(ordered: ordered, items: items, startIndex: startIndex)
        case .blockquote(let content):
            BlockquoteView(content: content)
        case .codeBlock(let language, let body):
            ArtifactCodeBlock(language: language, source: body)
        case .horizontalRule:
            Divider().padding(.vertical, 2)
        case .table(let headers, let rows):
            MarkdownTableView(headers: headers, rows: rows)
        }
    }
}

// MARK: - Heading

private struct HeadingView: View {
    let level: Int
    let text: String

    var body: some View {
        InlineMarkdownText(raw: text)
            .font(font)
            .foregroundStyle(.primary)
            .padding(.top, topPadding)
    }

    private var font: Font {
        switch level {
        case 1: return .system(.title, design: .default).weight(.bold)
        case 2: return .system(.title2, design: .default).weight(.bold)
        case 3: return .system(.title3, design: .default).weight(.semibold)
        case 4: return .system(.headline, design: .default)
        case 5: return .system(.subheadline, design: .default).weight(.semibold)
        default: return .system(.callout, design: .default).weight(.semibold)
        }
    }

    private var topPadding: CGFloat {
        switch level {
        case 1, 2: return 6
        default: return 2
        }
    }
}

// MARK: - Paragraph

private struct ProseView: View {
    let text: String

    var body: some View {
        InlineMarkdownText(raw: text)
            .font(.callout)
            .lineSpacing(2)
    }
}

// MARK: - Inline markdown

/// Renders one prose chunk as an `AttributedString` so inline syntax
/// (`**bold**`, `*italic*`, `` `code` ``, `[link](url)`) is honored.
/// `inlineOnlyPreservingWhitespace` keeps line breaks visible inside a
/// paragraph and avoids the block-level interpretations the full
/// Markdown grammar applies (we already split blocks ourselves).
struct InlineMarkdownText: View {
    let raw: String

    var body: some View {
        if raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            EmptyView()
        } else if let attr = try? AttributedString(
            markdown: raw,
            options: AttributedString.MarkdownParsingOptions(
                allowsExtendedAttributes: false,
                interpretedSyntax: .inlineOnlyPreservingWhitespace,
                failurePolicy: .returnPartiallyParsedIfPossible,
                languageCode: nil))
        {
            Text(attr)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        } else {
            Text(raw)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

// MARK: - List

private struct ListView: View {
    let ordered: Bool
    let items: [String]
    let startIndex: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(items.indices, id: \.self) { i in
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    bullet(for: i)
                        .font(.callout.monospacedDigit())
                        .foregroundStyle(.secondary)
                        .frame(minWidth: 22, alignment: .trailing)
                    InlineMarkdownText(raw: items[i])
                        .font(.callout)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
        .padding(.leading, 4)
    }

    private func bullet(for i: Int) -> Text {
        if ordered {
            return Text("\(startIndex + i).")
        } else {
            return Text("•")
        }
    }
}

// MARK: - Blockquote

private struct BlockquoteView: View {
    let content: String

    var body: some View {
        let nested = MarkdownParser.parse(content)
        HStack(alignment: .top, spacing: 0) {
            RoundedRectangle(cornerRadius: 2)
                .fill(Color.accentColor.opacity(0.55))
                .frame(width: 3)
            VStack(alignment: .leading, spacing: 6) {
                ForEach(nested.indices, id: \.self) { i in
                    MarkdownBlockView(block: nested[i])
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .background(
            Color.accentColor.opacity(0.06),
            in: RoundedRectangle(cornerRadius: 6))
    }
}

// MARK: - Table

private struct MarkdownTableView: View {
    let headers: [String]
    let rows: [[String]]

    var body: some View {
        ScrollView(.horizontal, showsIndicators: true) {
            VStack(spacing: 0) {
                tableRow(headers, isHeader: true)
                Divider()
                ForEach(rows.indices, id: \.self) { i in
                    tableRow(rows[i], isHeader: false)
                    if i < rows.count - 1 {
                        Divider().opacity(0.35)
                    }
                }
            }
        }
        .overlay(
            RoundedRectangle(cornerRadius: 6)
                .strokeBorder(Color.secondary.opacity(0.25), lineWidth: 1))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    @ViewBuilder
    private func tableRow(_ cells: [String], isHeader: Bool) -> some View {
        HStack(alignment: .top, spacing: 0) {
            ForEach(cells.indices, id: \.self) { i in
                InlineMarkdownText(raw: cells[i])
                    .font(isHeader ? .callout.bold() : .callout)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .frame(minWidth: 80, alignment: .leading)
                if i < cells.count - 1 {
                    Divider().opacity(0.35)
                }
            }
        }
        .background(isHeader
                     ? Color.secondary.opacity(0.10)
                     : Color.clear)
    }
}

// MARK: - Parser

/// Splits a Markdown source into a sequence of block-level elements.
/// Lines are scanned once; paragraph text accumulates until a block
/// boundary (blank line or a recognised block opener) flushes it.
/// Fenced code follows the same rules as the previous segmenter so
/// half-streamed code never disappears: an unclosed fence falls back
/// to paragraph until the closing fence arrives.
enum MarkdownParser {
    static func parse(_ raw: String) -> [MarkdownBlock] {
        let lines = raw.components(separatedBy: "\n")
        var blocks: [MarkdownBlock] = []
        var paragraph: [String] = []
        var i = 0

        func flushParagraph() {
            guard !paragraph.isEmpty else { return }
            let joined = paragraph.joined(separator: "\n")
            if !joined.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                blocks.append(.paragraph(joined))
            }
            paragraph.removeAll(keepingCapacity: true)
        }

        while i < lines.count {
            let line = lines[i]
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            // Blank line: paragraph boundary.
            if trimmed.isEmpty {
                flushParagraph()
                i += 1
                continue
            }

            // Fenced code block.
            if let (fence, lang) = openingFence(line) {
                var j = i + 1
                var body: [String] = []
                var closed = false
                while j < lines.count {
                    if isClosingFence(lines[j], fence: fence) {
                        closed = true
                        break
                    }
                    body.append(lines[j])
                    j += 1
                }
                if closed {
                    flushParagraph()
                    blocks.append(.codeBlock(
                        language: lang,
                        body: body.joined(separator: "\n")))
                    i = j + 1
                    continue
                }
                // Unclosed: keep as paragraph so a later round closes it.
                paragraph.append(line)
                i += 1
                continue
            }

            // Setext heading: previous paragraph + === or ---. Tried
            // BEFORE the horizontal rule so `Hello\n---` becomes an
            // h2 instead of a paragraph + hr.
            if !paragraph.isEmpty, isSetextUnderline(trimmed) {
                let level = trimmed.first == "=" ? 1 : 2
                let text = paragraph.joined(separator: "\n")
                paragraph.removeAll(keepingCapacity: true)
                blocks.append(.heading(level: level, text: text))
                i += 1
                continue
            }

            // Horizontal rule (---, ***, ___).
            if isHorizontalRule(trimmed) {
                flushParagraph()
                blocks.append(.horizontalRule)
                i += 1
                continue
            }

            // ATX heading: # … ######
            if let (level, text) = parseAtxHeading(trimmed) {
                flushParagraph()
                blocks.append(.heading(level: level, text: text))
                i += 1
                continue
            }

            // Blockquote.
            if trimmed.hasPrefix(">") {
                flushParagraph()
                var quoteLines: [String] = []
                while i < lines.count {
                    let qt = lines[i].trimmingCharacters(in: .whitespaces)
                    if !qt.hasPrefix(">") { break }
                    var s = qt
                    s.removeFirst()
                    if s.hasPrefix(" ") { s.removeFirst() }
                    quoteLines.append(s)
                    i += 1
                }
                blocks.append(.blockquote(quoteLines.joined(separator: "\n")))
                continue
            }

            // Unordered list (-, *, +).
            if parseUnorderedItem(line) != nil {
                flushParagraph()
                var items: [String] = []
                while i < lines.count, let item = parseUnorderedItem(lines[i]) {
                    items.append(item)
                    i += 1
                }
                blocks.append(.list(ordered: false,
                                     items: items,
                                     startIndex: 1))
                continue
            }

            // Ordered list (1. / 1)).
            if let first = parseOrderedItem(line) {
                flushParagraph()
                var items: [String] = [first.text]
                let start = first.number
                i += 1
                while i < lines.count, let item = parseOrderedItem(lines[i]) {
                    items.append(item.text)
                    i += 1
                }
                blocks.append(.list(ordered: true,
                                     items: items,
                                     startIndex: start))
                continue
            }

            // GFM table: pipe-row followed by a separator row.
            if line.contains("|"),
               i + 1 < lines.count,
               isTableSeparator(lines[i + 1])
            {
                flushParagraph()
                let headers = parseTableRow(line)
                var rows: [[String]] = []
                var j = i + 2
                while j < lines.count,
                      lines[j].contains("|"),
                      !lines[j].trimmingCharacters(in: .whitespaces).isEmpty
                {
                    rows.append(parseTableRow(lines[j]))
                    j += 1
                }
                blocks.append(.table(headers: headers, rows: rows))
                i = j
                continue
            }

            // Default: accumulate into the current paragraph.
            paragraph.append(line)
            i += 1
        }
        flushParagraph()
        return blocks
    }

    // MARK: helpers

    private static func openingFence(_ line: String) -> (String, String)? {
        let trimmed = line.drop(while: { $0 == " " || $0 == "\t" })
        if trimmed.hasPrefix("```") {
            let lang = trimmed.dropFirst(3)
                .trimmingCharacters(in: .whitespaces)
            return ("```", lang)
        }
        if trimmed.hasPrefix("~~~") {
            let lang = trimmed.dropFirst(3)
                .trimmingCharacters(in: .whitespaces)
            return ("~~~", lang)
        }
        return nil
    }

    private static func isClosingFence(_ line: String, fence: String) -> Bool {
        let trimmed = String(line.drop(while: { $0 == " " || $0 == "\t" }))
            .trimmingCharacters(in: .whitespaces)
        return trimmed == fence
    }

    private static func isHorizontalRule(_ trimmed: String) -> Bool {
        let compact = trimmed.replacingOccurrences(of: " ", with: "")
        guard compact.count >= 3 else { return false }
        if compact.allSatisfy({ $0 == "-" }) { return true }
        if compact.allSatisfy({ $0 == "*" }) { return true }
        if compact.allSatisfy({ $0 == "_" }) { return true }
        return false
    }

    private static func parseAtxHeading(_ trimmed: String) -> (Int, String)? {
        guard trimmed.hasPrefix("#") else { return nil }
        var level = 0
        var rest = Substring(trimmed)
        while rest.first == "#" {
            level += 1
            rest = rest.dropFirst()
            if level > 6 { return nil }
        }
        guard level >= 1 else { return nil }
        if rest.isEmpty { return (level, "") }
        guard rest.first == " " else { return nil }
        let text = rest.trimmingCharacters(in: .whitespaces)
        return (level, stripTrailingHashes(text))
    }

    private static func stripTrailingHashes(_ s: String) -> String {
        var t = Substring(s)
        while t.last == "#" { t = t.dropLast() }
        return t.trimmingCharacters(in: .whitespaces)
    }

    private static func isSetextUnderline(_ trimmed: String) -> Bool {
        guard !trimmed.isEmpty else { return false }
        if trimmed.count >= 2, trimmed.allSatisfy({ $0 == "=" }) { return true }
        if trimmed.count >= 2, trimmed.allSatisfy({ $0 == "-" }) { return true }
        return false
    }

    private static func parseUnorderedItem(_ line: String) -> String? {
        let trimmed = line.drop(while: { $0 == " " || $0 == "\t" })
        guard let first = trimmed.first else { return nil }
        guard first == "-" || first == "*" || first == "+" else { return nil }
        let rest = trimmed.dropFirst()
        guard rest.first == " " else { return nil }
        return rest.dropFirst().trimmingCharacters(in: .whitespaces)
    }

    private static func parseOrderedItem(_ line: String) -> (number: Int, text: String)? {
        let trimmed = line.drop(while: { $0 == " " || $0 == "\t" })
        var digits = ""
        var rest = trimmed
        while let c = rest.first, c.isNumber {
            digits.append(c)
            rest = rest.dropFirst()
        }
        guard !digits.isEmpty, let num = Int(digits) else { return nil }
        guard rest.first == "." || rest.first == ")" else { return nil }
        let afterMark = rest.dropFirst()
        guard afterMark.first == " " else { return nil }
        let text = afterMark.dropFirst().trimmingCharacters(in: .whitespaces)
        return (num, text)
    }

    private static func isTableSeparator(_ line: String) -> Bool {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard trimmed.contains("-") else { return false }
        let cells = parseTableRow(line)
        guard !cells.isEmpty else { return false }
        return cells.allSatisfy { cell in
            let t = cell.trimmingCharacters(in: .whitespaces)
            guard !t.isEmpty else { return false }
            let stripped = t
                .replacingOccurrences(of: ":", with: "")
                .replacingOccurrences(of: " ", with: "")
            return !stripped.isEmpty && stripped.allSatisfy { $0 == "-" }
        }
    }

    private static func parseTableRow(_ line: String) -> [String] {
        var t = line.trimmingCharacters(in: .whitespaces)
        if t.hasPrefix("|") { t.removeFirst() }
        if t.hasSuffix("|") { t.removeLast() }
        return t.split(separator: "|", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespaces) }
    }
}

/// Visually-distinct code artifact: rounded background, monospaced font,
/// language badge in the header, and a Copy-to-clipboard action. Long
/// blocks scroll horizontally so we don't reflow code that relies on
/// indentation. Vertical content stays inline (no max-height) so users
/// can read the whole artifact without nested scrolling.
struct ArtifactCodeBlock: View {
    let language: String
    let source: String

    @State private var copied: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
            ScrollView(.horizontal, showsIndicators: true) {
                Text(source)
                    .font(.system(.callout, design: .monospaced))
                    .textSelection(.enabled)
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .background(Color(NSColor.textBackgroundColor).opacity(0.5))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .strokeBorder(Color.secondary.opacity(0.25), lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "curlybraces")
                .foregroundStyle(.secondary)
            Text(displayLanguage)
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
            Spacer()
            Button(action: copy) {
                Label(copied ? "Copied" : "Copy",
                       systemImage: copied ? "checkmark" : "doc.on.doc")
                    .labelStyle(.titleAndIcon)
                    .font(.caption)
            }
            .buttonStyle(.borderless)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(Color(NSColor.controlBackgroundColor))
    }

    private var displayLanguage: String {
        language.isEmpty ? "code" : language
    }

    private func copy() {
        #if canImport(AppKit)
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(source, forType: .string)
        #endif
        copied = true
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            copied = false
        }
    }
}
