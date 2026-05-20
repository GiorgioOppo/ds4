import Foundation

/// Holds active security-scoped access to a project's source folders.
///
/// The macOS app runs sandboxed (see `DeepSeekUI.entitlements`). When the
/// user picks a folder via `NSOpenPanel`, macOS extends sandbox access for
/// the process lifetime — but that grant is lost across launches. The
/// `com.apple.security.files.bookmarks.app-scope` entitlement lets us
/// persist that access via app-scoped bookmarks resolved at launch.
///
/// Without this, every tool read in a project chat fails with "couldn't
/// be opened because you don't have permission to view it", including
/// reads that go through the per-project symlink farm at
/// `~/Library/Application Support/<app>/projects/<id>/`: following a
/// symlink in the farm lands on the user's real path, and the sandbox
/// blocks the underlying open() because no bookmark has been activated
/// for it.
///
/// Lifecycle: `ProjectLibrary` owns one vault per project, calls
/// `acquire(bookmarks:)` after loading from disk (and after each
/// `update`), and `release()` (implicit via `deinit`) when a project is
/// deleted or replaced.
final class ProjectSourceAccess {
    /// Outcome of resolving a single bookmark blob.
    struct Resolved {
        /// URL that the bookmark resolves to (post-redirection if the
        /// user moved the folder).
        let url: URL
        /// True when macOS marked the bookmark stale — still resolvable,
        /// but should be regenerated from `url` and re-persisted on the
        /// next save so it stays usable.
        let isStale: Bool
        /// True when `startAccessingSecurityScopedResource()` succeeded.
        /// False means the resolve worked but the sandbox refused the
        /// grant — the URL is unusable for I/O until the user re-picks
        /// the folder.
        let didStartAccess: Bool
    }

    private var accessed: [URL] = []

    deinit { release() }

    /// Stop accessing every URL the vault currently holds. Idempotent:
    /// safe to call multiple times. Called automatically on `deinit`.
    func release() {
        for url in accessed {
            url.stopAccessingSecurityScopedResource()
        }
        accessed.removeAll()
    }

    /// Resolve `bookmarks`, start security-scoped access on each, and
    /// return one optional `Resolved` per input — positions are
    /// preserved, with `nil` marking entries that failed to resolve
    /// (corrupted data, folder deleted, sandbox refused). Any
    /// previously held URLs are released first so the vault can be
    /// reused across project edits.
    @discardableResult
    func acquire(bookmarks: [Data]) -> [Resolved?] {
        release()
        var out: [Resolved?] = []
        out.reserveCapacity(bookmarks.count)
        for data in bookmarks {
            // Treat the sentinel empty Data() (used when bookmark
            // generation failed at pick time) as a non-resolvable entry
            // without going through `URL(resolvingBookmarkData:)`, which
            // would log an opaque error for every empty blob on launch.
            guard !data.isEmpty else {
                out.append(nil)
                continue
            }
            var isStale = false
            guard let url = try? URL(
                resolvingBookmarkData: data,
                options: [.withSecurityScope],
                relativeTo: nil,
                bookmarkDataIsStale: &isStale)
            else {
                out.append(nil)
                continue
            }
            let started = url.startAccessingSecurityScopedResource()
            if started { accessed.append(url) }
            out.append(Resolved(url: url, isStale: isStale,
                                didStartAccess: started))
        }
        return out
    }

    /// Create an app-scoped bookmark for a URL the user just picked via
    /// `NSOpenPanel`. Returns `nil` if the bookmark can't be produced
    /// (extremely rare — usually means the URL isn't actually
    /// user-selected). The caller persists the resulting `Data` in
    /// `Project.sourceBookmarks` next to its path string.
    static func makeBookmark(for url: URL) -> Data? {
        try? url.bookmarkData(
            options: [.withSecurityScope],
            includingResourceValuesForKeys: nil,
            relativeTo: nil)
    }
}
