import AppKit
import Foundation
import UniformTypeIdentifiers
import DS4Engine

/// Sandbox-friendly model selection. Under the App Sandbox the engine can only
/// `open()` files the user explicitly granted access to, so we pick the GGUF via
/// `NSOpenPanel`, begin security-scoped access (held for the app session — the
/// model stays mmap'd), and persist a security-scoped bookmark so the same file
/// re-opens on the next launch without re-prompting.
@MainActor
enum ModelPicker {
    private static let bookmarkKey = "ds4.modelBookmark"
    private static let pathKey = "ds4.modelPath"   // plain-path fallback (unsandboxed `make app` build)

    /// Present an open panel to choose a `.gguf` file. Returns its path and starts
    /// security-scoped access; persists a bookmark for next launch.
    static func pickGGUF() -> String? {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.title = "Choose a GGUF Model"
        panel.prompt = "Open"
        if let gguf = UTType(filenameExtension: "gguf") {
            panel.allowedContentTypes = [gguf, .data]   // prefer .gguf, still allow any file
        }
        guard panel.runModal() == .OK, let url = panel.url else { return nil }
        let scoped = url.startAccessingSecurityScopedResource()
        do {
            _ = try validateRunnableGGUF(at: url)
        } catch {
            if scoped { url.stopAccessingSecurityScopedResource() }
            presentValidationError(error)
            return nil
        }
        saveBookmark(url)
        return url.path
    }

    /// Metadata-only runtime validation used by Browse. Catalog download-only
    /// artifacts (PRO, split shards, MTP) and future unsupported architectures
    /// must not silently become the active local model.
    @discardableResult
    static func validateRunnableGGUF(at url: URL) throws -> RuntimeModelDescriptor {
        let descriptor = try InferenceService.inspectModel(path: url.path)
        _ = try BackendSelector.select(descriptor)
        return descriptor
    }

    /// Validate an already-accessible path and present a concise error when it
    /// is not runnable. This is used for legacy catalog files before selection;
    /// it inspects metadata only and never hashes hundreds of GB.
    static func acceptRunnableGGUF(path: String) -> Bool {
        do {
            _ = try validateRunnableGGUF(at: URL(fileURLWithPath: path))
            return true
        } catch {
            presentValidationError(error)
            return false
        }
    }

    /// Resolve a previously-picked model, starting security-scoped access. Returns
    /// its path if the bookmark still resolves.
    static func restoreBookmark() -> String? {
        if let data = UserDefaults.standard.data(forKey: bookmarkKey) {
            var stale = false
            if let url = try? URL(resolvingBookmarkData: data, options: .withSecurityScope,
                                  relativeTo: nil, bookmarkDataIsStale: &stale),
               url.startAccessingSecurityScopedResource() {
                if stale { saveBookmark(url) }   // refresh a stale bookmark
                return url.path
            }
        }
        // Plain-path fallback (unsandboxed build has full file access — no scope needed).
        if let path = UserDefaults.standard.string(forKey: pathKey),
           FileManager.default.isReadableFile(atPath: path) {
            return path
        }
        return nil
    }

    /// Persist a GGUF owned by DwarfStar (for example a catalog download under
    /// Application Support). It needs no security-scoped grant, so discard an
    /// older external-file bookmark before saving the plain path; otherwise the
    /// stale bookmark would win again on the next launch.
    static func rememberManagedGGUF(path: String) {
        guard FileManager.default.isReadableFile(atPath: path) else { return }
        UserDefaults.standard.set(path, forKey: pathKey)
        UserDefaults.standard.removeObject(forKey: bookmarkKey)
    }

    /// Commit a catalog selection found outside the managed directory. Preserve
    /// the security-scoped bookmark only when it grants this exact file; an old
    /// bookmark for another GGUF must not override the new catalog path later.
    static func rememberCatalogGGUF(path: String) {
        guard FileManager.default.isReadableFile(atPath: path) else { return }
        if AppEnvironment.isManagedModelPath(path) {
            rememberManagedGGUF(path: path)
            return
        }
        if bookmarkMatches(path: path) {
            // `saveBookmark` already persisted the matching grant. Keep it.
            return
        }
        UserDefaults.standard.set(path, forKey: pathKey)
        UserDefaults.standard.removeObject(forKey: bookmarkKey)
    }

    // MARK: Model-FOLDER grant (sandbox sibling caches)
    //
    // The file grant above covers ONLY the picked .gguf: its sidecar caches
    // (<model>.q4dense, <model>.expbundle — typically built by the demo/CLI
    // next to the model) stay unreadable under the sandbox, so the app
    // requantizes from scratch and rebuilds the bundle INSIDE its container
    // (gigabytes of duplication, or a skipped bundle when disk is short).
    // Granting the model's FOLDER makes the siblings readable and the engine
    // reuses them as-is.
    private static let dirBookmarkKey = "ds4.modelDirBookmark"

    /// Ask the user to grant read access to the model's folder. Returns true
    /// on grant; persists a security-scoped bookmark for later launches.
    static func pickModelFolder(near modelPath: String?) -> Bool {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.title = "Grant Access to the Model's Folder"
        panel.prompt = "Grant"
        panel.message = "Select the folder that contains the GGUF: existing sidecar caches (\u{201C}.q4dense\u{201D}, \u{201C}.expbundle\u{201D}) become reusable and nothing is rebuilt inside the app container."
        if let p = modelPath, !p.isEmpty {
            panel.directoryURL = URL(fileURLWithPath: (p as NSString).deletingLastPathComponent,
                                     isDirectory: true)
        }
        guard panel.runModal() == .OK, let url = panel.url else { return false }
        _ = url.startAccessingSecurityScopedResource()
        if let data = try? url.bookmarkData(options: .withSecurityScope,
                                            includingResourceValuesForKeys: nil, relativeTo: nil) {
            UserDefaults.standard.set(data, forKey: dirBookmarkKey)
        }
        return true
    }

    /// Re-arm the folder grant on launch (no-op without a stored bookmark).
    /// Access is held for the app session, like the model file's.
    static func restoreFolderBookmark() {
        guard let data = UserDefaults.standard.data(forKey: dirBookmarkKey) else { return }
        var stale = false
        if let url = try? URL(resolvingBookmarkData: data, options: .withSecurityScope,
                              relativeTo: nil, bookmarkDataIsStale: &stale),
           url.startAccessingSecurityScopedResource() {
            if stale, let d = try? url.bookmarkData(options: .withSecurityScope,
                                                    includingResourceValuesForKeys: nil, relativeTo: nil) {
                UserDefaults.standard.set(d, forKey: dirBookmarkKey)
            }
        }
    }

    private static func saveBookmark(_ url: URL) {
        if let data = try? url.bookmarkData(options: .withSecurityScope,
                                            includingResourceValuesForKeys: nil, relativeTo: nil) {
            UserDefaults.standard.set(data, forKey: bookmarkKey)
            UserDefaults.standard.removeObject(forKey: pathKey)
        } else {
            // Unsandboxed (make app): security-scoped bookmarks unavailable — the
            // engine has full file access, so remember the plain path to re-open it.
            UserDefaults.standard.set(url.path, forKey: pathKey)
            UserDefaults.standard.removeObject(forKey: bookmarkKey)
        }
    }

    private static func bookmarkMatches(path: String) -> Bool {
        guard let data = UserDefaults.standard.data(forKey: bookmarkKey) else { return false }
        var stale = false
        guard let bookmarked = try? URL(
            resolvingBookmarkData: data,
            options: .withSecurityScope,
            relativeTo: nil,
            bookmarkDataIsStale: &stale
        ) else { return false }
        return bookmarked.resolvingSymlinksInPath().standardizedFileURL.path
            == URL(fileURLWithPath: path).resolvingSymlinksInPath().standardizedFileURL.path
    }

    private static func presentValidationError(_ error: Error) {
        let alert = NSAlert()
        alert.alertStyle = .warning
        alert.messageText = "Modello non supportato"
        alert.informativeText = "Il GGUF scelto non può essere eseguito dal runtime corrente.\n\n\(error)"
        alert.addButton(withTitle: "OK")
        alert.runModal()
    }
}
