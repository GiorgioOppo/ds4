import Foundation
import SwiftUI
import UniformTypeIdentifiers
#if canImport(AppKit)
import AppKit
#endif

/// One file pulled into the composer. Holds the decoded text in memory
/// — these are bounded by `AttachmentLimits.maxBytesPerFile`, so even
/// a few documents stay well under a typical context window.
struct DocumentAttachment: Identifiable, Equatable {
    let id: UUID = UUID()
    let name: String
    let byteCount: Int
    let text: String
}

enum AttachmentLimits {
    /// Per-file cap. Trims with an explanatory marker rather than
    /// rejecting the file outright, so a user attaching a long log
    /// still gets the head of it.
    static let maxBytesPerFile: Int = 256 * 1024
}

/// Reads a file URL into a `DocumentAttachment`. Tries UTF-8 first,
/// falls back to ISO-Latin-1 so "mostly-ASCII" logs / source files
/// from non-Unicode editors still come through. Returns nil for
/// content that looks binary (null byte in the first 4 KB).
enum AttachmentReader {
    static func read(_ url: URL) -> DocumentAttachment? {
        guard let raw = try? Data(contentsOf: url) else { return nil }
        if looksBinary(raw) { return nil }

        let limit = AttachmentLimits.maxBytesPerFile
        let truncated = raw.count > limit
        let slice = truncated ? raw.prefix(limit) : raw

        let decoded = String(data: slice, encoding: .utf8)
            ?? String(data: slice, encoding: .isoLatin1)
            ?? ""

        let text = truncated
            ? decoded + "\n\n[…truncated at \(limit) bytes…]"
            : decoded

        return DocumentAttachment(
            name: url.lastPathComponent,
            byteCount: raw.count,
            text: text)
    }

    private static func looksBinary(_ data: Data) -> Bool {
        let head = data.prefix(4096)
        return head.contains(0)
    }
}

/// Wraps NSOpenPanel for text-ish content types. The list intentionally
/// allows `.data` as a catch-all so users can still pick log files and
/// odd source extensions Finder hasn't classified; we filter out true
/// binaries at read time via `AttachmentReader.looksBinary`.
enum AttachmentPicker {
    static func present() -> [DocumentAttachment] {
        #if canImport(AppKit)
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.title = "Attach text documents"
        panel.message = "Select one or more text files to attach to your message"
        panel.allowedContentTypes = allowedContentTypes()

        guard panel.runModal() == .OK else { return [] }
        return panel.urls.compactMap(AttachmentReader.read)
        #else
        return []
        #endif
    }

    private static func allowedContentTypes() -> [UTType] {
        // Broad set so most "looks like text" extensions pick up.
        // Anything binary that sneaks through this filter still gets
        // rejected by `AttachmentReader.looksBinary`.
        [.plainText, .utf8PlainText, .text, .sourceCode, .json, .xml,
         .yaml, .html, .commaSeparatedText, .log, .data]
    }
}

/// Formats a batch of attachments into a single textual prefix that
/// gets prepended to the user message before it's sent to the model.
/// Each document gets a fenced block with a header line so the model
/// can distinguish multiple files and the user's actual prompt.
enum AttachmentFormatter {
    static func prefix(for attachments: [DocumentAttachment]) -> String {
        guard !attachments.isEmpty else { return "" }
        var out = ""
        for a in attachments {
            out += "Attached document: \(a.name)\n"
            out += "```\n\(a.text)\n```\n\n"
        }
        return out
    }
}
