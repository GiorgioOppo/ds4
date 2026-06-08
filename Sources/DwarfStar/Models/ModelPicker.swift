import AppKit
import Foundation
import UniformTypeIdentifiers

/// Sandbox-friendly model selection. Under the App Sandbox the engine can only
/// `open()` files the user explicitly granted access to, so we pick the GGUF via
/// `NSOpenPanel`, begin security-scoped access (held for the app session — the
/// model stays mmap'd), and persist a security-scoped bookmark so the same file
/// re-opens on the next launch without re-prompting.
@MainActor
enum ModelPicker {
    private static let bookmarkKey = "ds4.modelBookmark"

    /// Present an open panel to choose a `.gguf` file. Returns its path and starts
    /// security-scoped access; persists a bookmark for next launch.
    static func pickGGUF() -> String? {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.title = "Scegli un modello GGUF"
        panel.prompt = "Apri"
        if let gguf = UTType(filenameExtension: "gguf") {
            panel.allowedContentTypes = [gguf, .data]   // prefer .gguf, still allow any file
        }
        guard panel.runModal() == .OK, let url = panel.url else { return nil }
        _ = url.startAccessingSecurityScopedResource()
        saveBookmark(url)
        return url.path
    }

    /// Resolve a previously-picked model, starting security-scoped access. Returns
    /// its path if the bookmark still resolves.
    static func restoreBookmark() -> String? {
        guard let data = UserDefaults.standard.data(forKey: bookmarkKey) else { return nil }
        var stale = false
        guard let url = try? URL(resolvingBookmarkData: data, options: .withSecurityScope,
                                 relativeTo: nil, bookmarkDataIsStale: &stale) else { return nil }
        guard url.startAccessingSecurityScopedResource() else { return nil }
        if stale { saveBookmark(url) }   // refresh a stale bookmark
        return url.path
    }

    private static func saveBookmark(_ url: URL) {
        if let data = try? url.bookmarkData(options: .withSecurityScope,
                                            includingResourceValuesForKeys: nil, relativeTo: nil) {
            UserDefaults.standard.set(data, forKey: bookmarkKey)
        }
    }
}
