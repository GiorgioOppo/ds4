import SwiftUI

/// Renders Markdown via `AttributedString(markdown:options:)`. Used
/// for finalized assistant messages (after `.done`). Streaming
/// in-progress text still flows through plain `Text` because partial
/// markdown (e.g. an unclosed `**`) renders distractingly.
///
/// The Foundation initializer handles inline emphasis, links, lists,
/// and code spans. Fenced code blocks aren't rendered as separate
/// boxes — they appear as monospaced spans inline. Good enough for
/// a v1; a fully tokenised renderer can replace this without
/// touching the message bubble.
struct MarkdownText: View {
    let raw: String

    var body: some View {
        if let attr = try? AttributedString(
            markdown: raw,
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
            // Markdown parser will only fail on truly malformed input;
            // fall back to plain text rather than swallowing the
            // content.
            Text(raw)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
