import SwiftUI
import AppKit
import UniformTypeIdentifiers
import DS4Engine
import DS4Core

extension ChatStore {
    // MARK: - Text-file attachments

    /// Present an open panel for one or more text files and stage their contents.
    /// Honors the App Sandbox: each pick grants security-scoped access for the
    /// one-shot read (entitlement: files.user-selected.read-write).
    func pickAndAttachFiles() {
        attachmentNote = nil
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = true
        panel.title = "Import Text Files"
        panel.prompt = "Import"
        // Prefer text types; allow any file (.data) so odd extensions can still be
        // picked — non-text content simply fails to decode and is reported.
        panel.allowedContentTypes = [.text, .plainText, .sourceCode, .json, .xml,
                                     .commaSeparatedText, .log, .data]
        guard panel.runModal() == .OK else { return }
        importFiles(panel.urls)
    }

    /// Read each URL as text (UTF-8, then Latin-1) and stage it; collect failures.
    func importFiles(_ urls: [URL]) {
        var failed: [String] = []
        for url in urls {
            let scoped = url.startAccessingSecurityScopedResource()
            defer { if scoped { url.stopAccessingSecurityScopedResource() } }
            guard let text = Self.readText(url) else { failed.append(url.lastPathComponent); continue }
            let name = url.lastPathComponent
            // Re-importing the identical file is a no-op (avoid duplicate context).
            if !attachments.contains(where: { $0.name == name && $0.content == text }) {
                attachments.append(ChatAttachment(name: name, content: text))
            }
        }
        if !failed.isEmpty {
            attachmentNote = "Could not read as text: \(failed.joined(separator: ", "))"
        }
    }

    func removeAttachment(_ id: UUID) { attachments.removeAll { $0.id == id } }

    /// Decode a file as text: UTF-8 first, then Latin-1 (covers most legacy files).
    static func readText(_ url: URL) -> String? {
        if let s = try? String(contentsOf: url, encoding: .utf8) { return s }
        return try? String(contentsOf: url, encoding: .isoLatin1)
    }

    /// Fold staged attachments + the typed message into the text sent to the model.
    /// Each file is delimited so the model can tell content apart from the prompt.
    static func composeUserText(typed: String, attachments: [ChatAttachment]) -> String {
        guard !attachments.isEmpty else { return typed }
        var parts: [String] = attachments.map {
            "--- Attached file: \($0.name) ---\n\($0.content)\n--- end: \($0.name) ---"
        }
        if !typed.isEmpty { parts.append(typed) }
        return parts.joined(separator: "\n\n")
    }
}
