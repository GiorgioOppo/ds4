import AppKit
import DS4Core
import DS4Engine
import UniformTypeIdentifiers

extension ChatStore {
    var canAttachImages: Bool {
        isReady && settings.mode == .local && service?.visionEnabled == true
            && service?.loadedVisionEncoderPath == settings.visionEncoderPath
    }

    var visionConfigurationNote: String {
        if canAttachImages { return "Vision attivo · puoi allegare fino a 4 immagini per messaggio." }
        if settings.mode != .local { return "Le immagini sono disponibili nella chat locale." }
        if settings.visionEncoderPath.isEmpty {
            return "Seleziona un modello Vision Experimental e il suo encoder, poi carica il modello."
        }
        return "Encoder configurato. Carica o ricarica il modello Vision Experimental per attivare le immagini."
    }

    @discardableResult
    func selectVisionEncoder(path: String) -> Bool {
        guard phase != .loading, !isGenerating, !benchRunning,
              EngineActivityGate.shared.activeOwner == nil else { return false }
        do {
            try InferenceService.validateVisionEncoder(path: path)
            settings.visionEncoderPath = path
            // A catalog selection must never be replaced by a stale bookmark.
            UserDefaults.standard.removeObject(forKey: "DS4VisionEncoderBookmark")
            return true
        } catch {
            attachmentNote = "Encoder non valido: \(error)"
            return false
        }
    }

    func pickVisionEncoder() {
        let panel = NSOpenPanel()
        panel.title = "Seleziona l’encoder DeepSeek Vision"
        panel.prompt = "Usa encoder"
        panel.message = "Scegli DeepSeek-V4-Flash-Vision-Encoder.gguf, separato dal modello principale."
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [UTType(filenameExtension: "gguf") ?? .data]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        let scoped = url.startAccessingSecurityScopedResource()
        guard selectVisionEncoder(path: url.path) else {
            if scoped { url.stopAccessingSecurityScopedResource() }
            return
        }
        if let bookmark = try? url.bookmarkData(options: .withSecurityScope,
                                                includingResourceValuesForKeys: nil, relativeTo: nil) {
            UserDefaults.standard.set(bookmark, forKey: "DS4VisionEncoderBookmark")
        }
    }

    func restoreVisionEncoderBookmark() {
        guard let data = UserDefaults.standard.data(forKey: "DS4VisionEncoderBookmark") else { return }
        var stale = false
        guard let url = try? URL(resolvingBookmarkData: data, options: .withSecurityScope,
                                relativeTo: nil, bookmarkDataIsStale: &stale),
              url.path == settings.visionEncoderPath else { return }
        _ = url.startAccessingSecurityScopedResource()
        if stale, let refreshed = try? url.bookmarkData(options: .withSecurityScope,
                                                       includingResourceValuesForKeys: nil, relativeTo: nil) {
            UserDefaults.standard.set(refreshed, forKey: "DS4VisionEncoderBookmark")
        }
    }
}
