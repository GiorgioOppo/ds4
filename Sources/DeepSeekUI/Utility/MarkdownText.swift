import SwiftUI
#if canImport(AppKit)
import AppKit
#endif

/// Renders Markdown via `AttributedString(markdown:options:)` for plain
/// prose and promotes fenced code blocks (```lang … ```) to dedicated
/// "artifact" boxes with a language badge and a Copy button. Used for
/// finalized assistant messages (after `.done`). Streaming in-progress
/// text still flows through plain `Text` because partial markdown
/// (e.g. an unclosed `**`) renders distractingly.
struct MarkdownText: View {
    let raw: String

    var body: some View {
        let segments = MarkdownSegmenter.segments(from: raw)
        VStack(alignment: .leading, spacing: 10) {
            ForEach(segments.indices, id: \.self) { i in
                switch segments[i] {
                case .prose(let text):
                    proseView(text)
                case .code(let language, let body):
                    ArtifactCodeBlock(language: language, body: body)
                }
            }
        }
    }

    @ViewBuilder
    private func proseView(_ text: String) -> some View {
        if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            EmptyView()
        } else if let attr = try? AttributedString(
            markdown: text,
            options: AttributedString.MarkdownParsingOptions(
                allowsExtendedAttributes: false,
                interpretedSyntax: .full,
                failurePolicy: .returnPartiallyParsedIfPossible,
                languageCode: nil))
        {
            Text(attr)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        } else {
            // Markdown parser only trips on truly malformed input; fall
            // back to plain text rather than swallowing the content.
            Text(text)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

/// One contiguous slice of a rendered message: either prose (passed to
/// the AttributedString markdown renderer) or a fenced code block we
/// promote to a distinct artifact card.
enum MarkdownSegment: Equatable {
    case prose(String)
    case code(language: String, body: String)
}

/// Splits a raw markdown string into prose + fenced-code segments.
/// Recognises ```lang … ``` and ~~~lang … ~~~ on their own lines. Any
/// fence without a matching closing fence stays inline as prose, so
/// half-streamed text never disappears.
enum MarkdownSegmenter {
    static func segments(from raw: String) -> [MarkdownSegment] {
        var out: [MarkdownSegment] = []
        let lines = raw.components(separatedBy: "\n")
        var i = 0
        var prose: [String] = []

        func flushProse() {
            guard !prose.isEmpty else { return }
            let joined = prose.joined(separator: "\n")
            out.append(.prose(joined))
            prose.removeAll(keepingCapacity: true)
        }

        while i < lines.count {
            let line = lines[i]
            if let (fence, lang) = openingFence(line) {
                // Look ahead for the matching closing fence on its own line.
                var j = i + 1
                var bodyLines: [String] = []
                var closed = false
                while j < lines.count {
                    if isClosingFence(lines[j], fence: fence) {
                        closed = true
                        break
                    }
                    bodyLines.append(lines[j])
                    j += 1
                }
                if closed {
                    flushProse()
                    out.append(.code(language: lang,
                                      body: bodyLines.joined(separator: "\n")))
                    i = j + 1
                    continue
                }
                // Unclosed fence: keep as prose so a partial parse can
                // still recover later.
            }
            prose.append(line)
            i += 1
        }
        flushProse()
        return out
    }

    /// Returns (fence-marker, language) if `line` opens a fenced block.
    /// Accepts ``` or ~~~ at the start of the line (after any spaces),
    /// optionally followed by a language tag.
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
}

/// Visually-distinct code artifact: rounded background, monospaced font,
/// language badge in the header, and a Copy-to-clipboard action. Long
/// blocks scroll horizontally so we don't reflow code that relies on
/// indentation. Vertical content stays inline (no max-height) so users
/// can read the whole artifact without nested scrolling.
struct ArtifactCodeBlock: View {
    let language: String
    let body: String

    @State private var copied: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
            ScrollView(.horizontal, showsIndicators: true) {
                Text(body)
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
        pb.setString(body, forType: .string)
        #endif
        copied = true
        // Reset the label after a beat so repeated copies still
        // animate the state change.
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            copied = false
        }
    }
}
