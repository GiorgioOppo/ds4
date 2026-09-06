import SwiftUI
import AppKit
import UniformTypeIdentifiers
import ImageIO
import DS4Engine
import DS4Core

extension ChatStore {
    // MARK: - File attachments

    /// Present an open panel for one or more text files and stage their contents.
    /// Honors the App Sandbox: each pick grants security-scoped access for the
    /// one-shot read (entitlement: files.user-selected.read-write).
    func pickAndAttachFiles() {
        attachmentNote = nil
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = true
        panel.title = "Allega documenti o immagini"
        panel.prompt = "Allega"
        // Prefer text types; allow any file (.data) so odd extensions can still be
        // picked — non-text content simply fails to decode and is reported.
        panel.allowedContentTypes = [.image, .text, .plainText, .sourceCode, .json, .xml,
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
            let name = url.lastPathComponent
            guard let size = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize,
                  size <= ChatImage.maximumFileBytes,
                  let data = try? Data(contentsOf: url), data.count <= ChatImage.maximumFileBytes else {
                failed.append("\(name): file non leggibile o maggiore di 20 MB")
                continue
            }
            if let source = CGImageSourceCreateWithData(data as CFData, nil) {
                guard canAttachImages else {
                    failed.append("\(name): carica DeepSeek Vision e il suo encoder in Impostazioni")
                    continue
                }
                guard let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
                      let width = properties[kCGImagePropertyPixelWidth] as? Int,
                      let height = properties[kCGImagePropertyPixelHeight] as? Int,
                      width > 0, height > 0, width <= 40_000_000 / height else {
                    failed.append("\(name): immagine non valida o maggiore di 40 megapixel")
                    continue
                }
                if attachments.contains(where: { $0.imageData == data }) { continue }
                guard attachments.filter({ $0.imageData != nil }).count < ChatImage.maximumImagesPerTurn else {
                    failed.append("\(name): massimo 4 immagini per messaggio")
                    continue
                }
                attachments.append(ChatAttachment(name: name, content: "", imageData: data))
                continue
            }
            guard let text = Self.decodeText(data) else {
                failed.append("\(name): formato non supportato; usa testo, PNG o JPEG")
                continue
            }
            // Re-importing the identical file is a no-op (avoid duplicate context).
            if !attachments.contains(where: { $0.name == name && $0.content == text }) {
                attachments.append(ChatAttachment(name: name, content: text))
            }
        }
        if !failed.isEmpty {
            attachmentNote = failed.joined(separator: "\n")
        }
    }

    func removeAttachment(_ id: UUID) { attachments.removeAll { $0.id == id } }

    /// Decode a file as text: UTF-8 first, then Latin-1 (covers most legacy files).
    static func readText(_ url: URL) -> String? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return decodeText(data)
    }

    private static func decodeText(_ data: Data) -> String? {
        // Latin-1 accepts every byte, including executables and images. Reject
        // binary control bytes before offering that legacy text fallback.
        guard !data.contains(where: { $0 == 0 || ($0 < 32 && $0 != 9 && $0 != 10 && $0 != 13) }) else { return nil }
        return String(data: data, encoding: .utf8) ?? String(data: data, encoding: .isoLatin1)
    }

    /// Fold staged attachments + the typed message into the text sent to the model.
    /// Each file is delimited so the model can tell content apart from the prompt.
    static func composeUserText(typed: String, attachments: [ChatAttachment]) -> String {
        guard !attachments.isEmpty else { return typed }
        var parts: [String] = attachments.filter { $0.imageData == nil }.map {
            "--- Attached file: \($0.name) ---\n\($0.content)\n--- end: \($0.name) ---"
        }
        if !typed.isEmpty { parts.append(typed) }
        return parts.joined(separator: "\n\n")
    }
}
