import Foundation

/// Where a model lives. The chat layer talks to whichever endpoint
/// is currently `loaded` on `ModelState` without caring about the
/// flavour — the inference plumbing on the other side maps the
/// endpoint to a concrete backend (today: local Metal transformer;
/// future: HTTP/SSE remote inference).
///
/// Codable so the `ModelLibrary` can persist a recents list under
/// `Application Support/.../models.json`, and Hashable so SwiftUI
/// pickers / dictionaries can use it directly.
enum ModelEndpoint: Codable, Hashable {
    /// On-disk converted V4 directory: contains tokenizer.json,
    /// config.json, and `model-*.safetensors` shards. The whole
    /// load path (`InferenceService.loadModel(at:)`) operates on
    /// the wrapped URL.
    case localDirectory(path: String)

    // case remoteAPI(baseURL: URL, modelName: String,
    //                 apiKeyKeychainID: String?)
    //
    // Reserved for the next iteration. Each remote endpoint will
    // need an inference backend that translates `generateForConversation`
    // events into HTTP/SSE chunks, plus a keychain hookup for the
    // API key. Keeping the case as a comment so the surrounding
    // code already accepts a Codable representation without a
    // migration when the case lands.

    /// Stable identifier for SwiftUI ForEach / set membership.
    /// Built from the case payload so two `.localDirectory` values
    /// for the same path compare equal.
    var id: String {
        switch self {
        case .localDirectory(let path): return "local::\(path)"
        }
    }

    /// User-facing label. For local directories it's the trailing
    /// folder name (the part the user actually picked in Finder).
    var displayName: String {
        switch self {
        case .localDirectory(let path):
            let url = URL(fileURLWithPath: path)
            let name = url.lastPathComponent
            return name.isEmpty ? path : name
        }
    }

    /// Long-form description used in tooltips and the picker
    /// secondary label. For local directories shows the full path
    /// so the user can disambiguate two folders with the same
    /// trailing name.
    var subtitle: String {
        switch self {
        case .localDirectory(let path): return path
        }
    }

    /// SF Symbol the toolbar picker uses for this endpoint kind.
    /// Distinct symbols make the (eventual) local-vs-remote
    /// distinction obvious at a glance.
    var iconName: String {
        switch self {
        case .localDirectory: return "internaldrive"
        }
    }
}
